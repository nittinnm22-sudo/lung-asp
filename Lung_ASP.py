"""
Lung_ASP.py  v6.5
FDG PET/CT Primary Lung Tumor Segmentation Pipeline
Integrates TotalSegmentator, Random Walker segmentation, exclusion masks,
tumor protection zone, hilar zone, node separation.
Calls advanced_metrics.compute_all_metrics() and Mask_QC overlay generation.
"""

import os
import sys
import logging
import argparse
import tempfile
import numpy as np
import nibabel as nib
from scipy import ndimage
from pathlib import Path

logger = logging.getLogger(__name__)

VERSION = "6.5"
DEFAULT_DPI = 150
TOTALSEGMENTATOR_TIMEOUT = 600  # seconds; increase for large volumes or slow systems


def _load_nifti(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path}")
    nii = nib.load(str(path))
    return nii.get_fdata().astype(np.float32), nii.affine, nii.header


def _resample_to_reference(source_nii, reference_nii):
    from scipy.ndimage import zoom
    src_data = source_nii.get_fdata().astype(np.float32)
    src_spacing = np.abs(np.diag(source_nii.affine)[:3])
    ref_spacing = np.abs(np.diag(reference_nii.affine)[:3])
    ref_shape = reference_nii.shape[:3]
    zoom_factors = (
        src_spacing[0] / ref_spacing[0] * src_data.shape[0] / ref_shape[0],
        src_spacing[1] / ref_spacing[1] * src_data.shape[1] / ref_shape[1],
        src_spacing[2] / ref_spacing[2] * src_data.shape[2] / ref_shape[2],
    )
    resampled = zoom(src_data, zoom_factors, order=1)
    # Pad or crop to exact reference shape
    result = np.zeros(ref_shape, dtype=np.float32)
    s = tuple(slice(0, min(resampled.shape[i], ref_shape[i])) for i in range(3))
    result[s] = resampled[s]
    return result


def _get_spacing(affine):
    return tuple(float(np.abs(affine[i, i])) for i in range(3))


def _normalize_pet(pet_data):
    return np.clip(pet_data, 0, None).astype(np.float32)


def _run_totalsegmentator(ct_nifti_path, output_dir):
    import subprocess
    cmd = [
        sys.executable, "-m", "totalsegmentator",
        "-i", str(ct_nifti_path),
        "-o", str(output_dir),
        "--task", "total",
        "--fast",
        "--ml"
    ]
    logger.info(f"Running TotalSegmentator: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=TOTALSEGMENTATOR_TIMEOUT)
    if result.returncode != 0:
        logger.warning(f"TotalSegmentator failed: {result.stderr[:500]}")
        return None
    return output_dir


def _load_lung_mask(ts_dir, reference_shape):
    ts_dir = Path(ts_dir)
    combined = np.zeros(reference_shape, dtype=bool)
    found = False
    for fname in ["lung_left.nii.gz", "lung_right.nii.gz",
                  "lung_upper_lobe_left.nii.gz", "lung_upper_lobe_right.nii.gz",
                  "lung_lower_lobe_left.nii.gz", "lung_lower_lobe_right.nii.gz"]:
        fpath = ts_dir / fname
        if fpath.exists():
            try:
                nii = nib.load(str(fpath))
                data = nii.get_fdata() > 0.5
                if data.shape == reference_shape:
                    combined |= data
                    found = True
            except Exception as e:
                logger.warning(f"Failed to load {fname}: {e}")
    return combined if found else None


def _create_hilar_zone(lung_mask, spacing, dilation_mm=25.0):
    s = ndimage.generate_binary_structure(3, 1)
    n = max(int(round(dilation_mm / min(spacing))), 2)
    # Erode lung mask to get core
    eroded = ndimage.binary_erosion(lung_mask, structure=s, iterations=n)
    # Hilar zone = dilated lung minus eroded lung (ring around inner lung)
    dilated = ndimage.binary_dilation(lung_mask, structure=s, iterations=n // 2)
    hilar = dilated & ~eroded
    return hilar.astype(bool)


def _random_walker_segment(pet_data, initial_mask, spacing):
    try:
        from skimage.segmentation import random_walker
        if initial_mask.sum() == 0:
            return initial_mask
        markers = np.zeros(pet_data.shape, dtype=int)
        # Foreground markers: high PET within initial mask
        s = ndimage.generate_binary_structure(3, 1)
        eroded_fg = ndimage.binary_erosion(initial_mask, structure=s, iterations=2)
        if eroded_fg.sum() < 10:
            eroded_fg = initial_mask.copy()
        markers[eroded_fg] = 1
        # Background markers: dilated outside of mask
        dilated = ndimage.binary_dilation(initial_mask, structure=s, iterations=5)
        bg = ~dilated
        markers[bg] = 2
        # Only run on bounding box region
        coords = np.argwhere(dilated)
        if len(coords) == 0:
            return initial_mask
        mn = coords.min(axis=0)
        mx = coords.max(axis=0) + 1
        pet_crop = pet_data[mn[0]:mx[0], mn[1]:mx[1], mn[2]:mx[2]]
        mark_crop = markers[mn[0]:mx[0], mn[1]:mx[1], mn[2]:mx[2]]
        if mark_crop.max() < 2:
            return initial_mask
        result = random_walker(pet_crop, mark_crop, beta=100, mode='cg_mg')
        seg = np.zeros_like(initial_mask, dtype=bool)
        seg[mn[0]:mx[0], mn[1]:mx[1], mn[2]:mx[2]] = (result == 1)
        return seg
    except Exception as e:
        logger.warning(f"Random Walker failed ({e}), using initial mask")
        return initial_mask


def _create_exclusion_mask(ct_data, lung_mask, spacing):
    # Exclude: bone (> 150 HU), air (< -700 HU) outside lung
    bone = ct_data > 150
    air = ct_data < -700
    excl = bone | air
    # If we have lung mask, only exclude outside of lung region
    if lung_mask is not None:
        s = ndimage.generate_binary_structure(3, 1)
        exp_lung = ndimage.binary_dilation(
            lung_mask, structure=s,
            iterations=max(int(round(10.0 / min(spacing))), 2)
        )
        # Air exclusion only outside lung
        excl = bone | (air & ~exp_lung)
    return excl.astype(bool)


def _apply_tumor_protection_zone(mask, pet_data, spacing, protection_mm=5.0):
    if mask.sum() == 0:
        return mask
    s = ndimage.generate_binary_structure(3, 1)
    n = max(int(round(protection_mm / min(spacing))), 1)
    protected = ndimage.binary_dilation(mask, structure=s, iterations=n)
    # Only keep voxels with positive PET signal in the protected zone
    # that are connected to the original mask
    expanded = protected & (pet_data > 0)
    # Fill holes in the expanded region
    expanded = ndimage.binary_fill_holes(expanded)
    # Reconnect to original: keep expanded voxels that are within the protected zone or original mask
    result = expanded & (mask | protected)
    # Ensure original mask is fully included
    result = result | mask
    return result.astype(bool)


def _separate_nodes(mask, primary_mask, spacing, min_node_gap_mm=10.0):
    s = ndimage.generate_binary_structure(3, 1)
    lc, nc = ndimage.label(mask, structure=s)
    if nc <= 1:
        return mask.astype(bool), []
    primary_label = lc[np.argwhere(primary_mask)[0]] if primary_mask.sum() > 0 else 0
    primary_comp = (lc == primary_label) if primary_label > 0 else primary_mask.copy()
    nodes = []
    for i in range(1, nc + 1):
        if i == primary_label:
            continue
        comp = (lc == i)
        # Check if this component is far enough from primary
        comp_coords = np.argwhere(comp)
        prim_coords = np.argwhere(primary_comp)
        if len(prim_coords) == 0:
            nodes.append(comp)
            continue
        from scipy.spatial import cKDTree
        prim_tree = cKDTree(prim_coords * np.array(spacing))
        comp_phys = comp_coords * np.array(spacing)
        min_dist, _ = prim_tree.query(comp_phys, k=1, workers=1)
        min_dist = float(min_dist.min())
        if min_dist > min_node_gap_mm:
            nodes.append(comp)
        else:
            primary_comp = primary_comp | comp
    return primary_comp.astype(bool), nodes


def _pet_threshold_segment(pet_data, initial_mask, threshold_fraction=0.42):
    if initial_mask.sum() == 0:
        return initial_mask
    suv_max = float(pet_data[initial_mask].max())
    threshold = threshold_fraction * suv_max
    return (pet_data >= threshold) & initial_mask


def segment_lung_tumor(
    pet_nifti_path,
    ct_nifti_path,
    output_dir,
    use_totalseg=True,
    dpi=150,
    progress_callback=None,
):
    """Run the Lung ASP segmentation pipeline.

    Parameters
    ----------
    pet_nifti_path : str or Path
        Path to input PET NIfTI file (SUV-normalized)
    ct_nifti_path : str or Path
        Path to input CT NIfTI file
    output_dir : str or Path
        Directory for all outputs
    use_totalseg : bool
        Whether to use TotalSegmentator for lung masks
    dpi : int
        DPI for QC image output
    progress_callback : callable, optional
        Called with (step_name, fraction_complete) for progress updates

    Returns
    -------
    dict
        Results: mask_path, metrics, output_dir, status
    """
    import advanced_metrics
    import Mask_QC

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _progress(step, frac):
        if progress_callback is not None:
            try:
                progress_callback(step, frac)
            except Exception:
                pass
        logger.info(f"[{int(frac*100):3d}%] {step}")

    _progress("Loading NIfTI files", 0.02)
    pet_nii = nib.load(str(pet_nifti_path))
    ct_nii = nib.load(str(ct_nifti_path))
    ct_data = ct_nii.get_fdata().astype(np.float32)
    ct_affine = ct_nii.affine
    spacing = _get_spacing(ct_affine)
    logger.info(f"CT shape: {ct_data.shape}, spacing: {spacing}")

    # Resample PET to CT space if needed
    pet_data_orig = pet_nii.get_fdata().astype(np.float32)
    if pet_data_orig.shape != ct_data.shape:
        _progress("Resampling PET to CT space", 0.05)
        pet_data = _resample_to_reference(pet_nii, ct_nii)
    else:
        pet_data = pet_data_orig
    pet_data = _normalize_pet(pet_data)

    # TotalSegmentator lung masks
    lung_mask = None
    hilar_mask = None
    if use_totalseg:
        _progress("Running TotalSegmentator", 0.10)
        try:
            ts_dir = output_dir / "totalsegmentator"
            ts_dir.mkdir(exist_ok=True)
            ts_out = _run_totalsegmentator(ct_nifti_path, ts_dir)
            if ts_out is not None:
                lung_mask = _load_lung_mask(ts_dir, ct_data.shape)
                if lung_mask is not None:
                    hilar_mask = _create_hilar_zone(lung_mask, spacing)
                    logger.info(f"Lung mask loaded: {lung_mask.sum():,} voxels")
                else:
                    logger.warning("Could not load lung mask from TotalSegmentator")
        except Exception as e:
            logger.warning(f"TotalSegmentator step failed: {e}")

    _progress("Initial PET segmentation", 0.30)
    # Body background estimate: mean PET outside high-uptake regions
    body_mask = ct_data > -200  # rough body boundary
    if body_mask.sum() > 100:
        bg_vals = pet_data[body_mask & (pet_data < np.percentile(pet_data[body_mask], 80))]
        background = float(np.mean(bg_vals)) if len(bg_vals) > 0 else 1.0
    else:
        background = 1.0
    background = max(background, 0.1)

    threshold = 2.5 * background
    logger.info(f"PET threshold: {threshold:.3f} (background={background:.3f})")

    # Initial mask: PET >= threshold
    initial_mask = (pet_data >= threshold).astype(bool)

    # Restrict to lung region if available
    if lung_mask is not None:
        # Expand lung mask slightly
        s = ndimage.generate_binary_structure(3, 1)
        expanded_lung = ndimage.binary_dilation(
            lung_mask, structure=s,
            iterations=max(int(round(20.0 / min(spacing))), 3)
        )
        initial_mask = initial_mask & expanded_lung

    _progress("Applying exclusion mask", 0.40)
    excl_mask = _create_exclusion_mask(ct_data, lung_mask, spacing)
    initial_mask = initial_mask & ~excl_mask

    _progress("Cleaning initial mask", 0.45)
    s = ndimage.generate_binary_structure(3, 1)
    # Fill holes
    initial_mask = ndimage.binary_fill_holes(initial_mask)
    # Remove small components
    lc, nc = ndimage.label(initial_mask, structure=s)
    if nc > 0:
        sizes = ndimage.sum(initial_mask, lc, range(1, nc + 1))
        min_size = 50
        keep = np.zeros_like(initial_mask)
        for i, sz in enumerate(sizes, 1):
            if sz >= min_size:
                keep |= (lc == i)
        initial_mask = keep.astype(bool)

    if initial_mask.sum() == 0:
        logger.warning("Empty mask after cleaning, trying fallback threshold")
        threshold_fallback = 1.5 * background
        initial_mask = (pet_data >= threshold_fallback).astype(bool)
        initial_mask = ndimage.binary_fill_holes(initial_mask)

    _progress("Random Walker refinement", 0.55)
    tumor_mask = _random_walker_segment(pet_data, initial_mask, spacing)

    _progress("Applying tumor protection zone", 0.60)
    tumor_mask = _apply_tumor_protection_zone(tumor_mask, pet_data, spacing)

    # Exclude hilar zone from mask if available
    if hilar_mask is not None:
        # Don't exclude hilar region that has high PET uptake (possible tumor)
        hilar_high_pet = hilar_mask & (pet_data >= threshold * 1.5)
        safe_hilar = hilar_mask & ~hilar_high_pet
        tumor_mask = tumor_mask & ~safe_hilar

    _progress("Final mask cleanup", 0.65)
    tumor_mask = ndimage.binary_fill_holes(tumor_mask)
    lc, nc = ndimage.label(tumor_mask, structure=s)
    if nc > 0:
        sizes = ndimage.sum(tumor_mask, lc, range(1, nc + 1))
        keep = np.zeros_like(tumor_mask)
        for i, sz in enumerate(sizes, 1):
            if sz >= 50:
                keep |= (lc == i)
        tumor_mask = keep.astype(bool)

    if tumor_mask.sum() == 0:
        logger.error("Final tumor mask is empty!")
        return {"status": "error", "message": "Empty tumor mask", "output_dir": str(output_dir)}

    # Save tumor mask
    _progress("Saving tumor mask", 0.70)
    mask_nii = nib.Nifti1Image(tumor_mask.astype(np.uint8), ct_affine, ct_nii.header)
    mask_path = output_dir / "tumor_mask.nii.gz"
    nib.save(mask_nii, str(mask_path))
    logger.info(f"Saved tumor mask: {mask_path} ({tumor_mask.sum():,} voxels)")

    # Save constraint mask (hilar + exclusion) for QC
    constraint_mask = np.zeros_like(tumor_mask, dtype=np.uint8)
    if hilar_mask is not None:
        constraint_mask |= hilar_mask.astype(np.uint8)
    constraint_mask_path = output_dir / "constraint_mask.nii.gz"
    constraint_nii = nib.Nifti1Image(constraint_mask, ct_affine)
    nib.save(constraint_nii, str(constraint_mask_path))

    # Compute metrics
    _progress("Computing radiomics metrics", 0.75)
    try:
        metrics = advanced_metrics.compute_all_metrics(
            pet_data=pet_data,
            tumor_mask=tumor_mask,
            spacing=spacing,
            ct_data=ct_data,
            ct_nifti_path=str(ct_nifti_path),
        )
        logger.info(f"Metrics computed: SUVmax={metrics.get('SUVmax', 0):.2f}, "
                    f"MTV={metrics.get('MTV', 0):.1f}cc, "
                    f"Dmax={metrics.get('Dmax_mm', 0):.1f}mm")
    except Exception as e:
        logger.error(f"Metrics computation failed: {e}", exc_info=True)
        metrics = {}

    # Generate QC overlays
    _progress("Generating QC overlays", 0.85)
    try:
        Mask_QC.generate_qc_overlays(
            pet_nifti_path=str(pet_nifti_path),
            ct_nifti_path=str(ct_nifti_path),
            tumor_mask_path=str(mask_path),
            constraint_mask_path=str(constraint_mask_path),
            output_path=str(output_dir / "Mask_QC.png"),
            dpi=dpi,
        )
        logger.info("Mask QC overlay generated")
    except Exception as e:
        logger.error(f"Mask QC overlay failed: {e}", exc_info=True)

    _progress("Generating metric QC overlays", 0.92)
    try:
        Mask_QC.generate_metric_qc_overlays(
            pet_nifti_path=str(pet_nifti_path),
            ct_nifti_path=str(ct_nifti_path),
            tumor_mask_path=str(mask_path),
            output_dir=str(output_dir),
            metrics=metrics,
            dpi=dpi,
        )
        logger.info("Metric QC overlays generated")
    except Exception as e:
        logger.error(f"Metric QC overlays failed: {e}", exc_info=True)

    _progress("Complete", 1.0)
    return {
        "status": "success",
        "mask_path": str(mask_path),
        "metrics": metrics,
        "output_dir": str(output_dir),
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description=f"Lung ASP v{VERSION} — FDG PET/CT Lung Tumor Segmentation"
    )
    parser.add_argument("--pet", required=True, help="PET NIfTI file path")
    parser.add_argument("--ct", required=True, help="CT NIfTI file path")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--no-totalseg", action="store_true",
                        help="Skip TotalSegmentator")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI,
                        help=f"QC image DPI (default: {DEFAULT_DPI})")
    args = parser.parse_args()

    result = segment_lung_tumor(
        pet_nifti_path=args.pet,
        ct_nifti_path=args.ct,
        output_dir=args.out,
        use_totalseg=not args.no_totalseg,
        dpi=args.dpi,
    )
    if result.get("status") == "success":
        print(f"SUCCESS: mask saved to {result['mask_path']}")
        m = result.get("metrics", {})
        for k, v in m.items():
            if k != "qc_coords":
                print(f"  {k}: {v}")
    else:
        print(f"ERROR: {result.get('message', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
