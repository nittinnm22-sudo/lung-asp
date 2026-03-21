"""Lung_ASP.py  v6.5
FDG PET/CT Primary Lung Tumor Segmentation Pipeline.

Pipeline steps:
  1. Run TotalSegmentator on CT for anatomical structures.
  2. Build constraint / exclusion masks from lobe and vessel segmentations.
  3. Create hilar exclusion zone.
  4. Run random walker segmentation on PET within lung region.
  5. Post-process and clean tumor mask.
  6. Optionally separate lymph nodes from primary tumor.
  7. Save tumor mask as NIfTI.
  8. Compute radiomics metrics (advanced_metrics.compute_all_metrics).
  9. Generate 3×3 QC grid (Mask_QC.generate_qc_overlays).
 10. Generate per-metric QC images (Mask_QC.generate_metric_qc_overlays).
 11. Save metrics as JSON and CSV.
 12. Return metrics dict.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import nibabel as nib
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    label as ndi_label,
    zoom,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> Path:
    """Create directory (and parents) if it does not exist. Returns Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _voxel_spacing(affine: np.ndarray) -> np.ndarray:
    """Extract voxel spacing from NIfTI affine."""
    return np.sqrt((affine[:3, :3] ** 2).sum(axis=0))


def _resample_to(
    vol: np.ndarray,
    source_shape: tuple,
    target_shape: tuple,
    order: int = 1,
) -> np.ndarray:
    """Resample *vol* (whose shape is *source_shape*) to *target_shape*."""
    if vol.shape == target_shape:
        return vol
    factors = tuple(t / s for t, s in zip(target_shape, source_shape))
    return zoom(vol.astype(np.float32), factors, order=order)


# ---------------------------------------------------------------------------
# TotalSegmentator integration
# ---------------------------------------------------------------------------

def _run_totalsegmentator(
    ct_path: str,
    seg_dir: str,
    device: str = "cpu",
    fast: bool = False,
) -> bool:
    """Run TotalSegmentator on the CT image.

    Saves individual NIfTI segmentation files into *seg_dir*.
    Returns True on success, False on failure.
    """
    try:
        from totalsegmentator.python_api import totalsegmentator
        totalsegmentator(
            input=ct_path,
            output=seg_dir,
            device=device,
            fast=fast,
            ml=True,
        )
        logger.info("TotalSegmentator completed → %s", seg_dir)
        return True
    except ImportError:
        logger.warning("TotalSegmentator not installed; skipping anatomical segmentation")
        return False
    except Exception as exc:
        logger.error("TotalSegmentator failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Lung lobe loading & constraint mask
# ---------------------------------------------------------------------------

LOBE_FILENAMES = {
    "right_upper_lobe": "lung_upper_lobe_right.nii.gz",
    "right_middle_lobe": "lung_middle_lobe_right.nii.gz",
    "right_lower_lobe": "lung_lower_lobe_right.nii.gz",
    "left_upper_lobe":  "lung_upper_lobe_left.nii.gz",
    "left_lower_lobe":  "lung_lower_lobe_left.nii.gz",
}

SIDE_LOBES = {
    "left":  ["left_upper_lobe", "left_lower_lobe"],
    "right": ["right_upper_lobe", "right_middle_lobe", "right_lower_lobe"],
}

HILAR_STRUCTURES = [
    "pulmonary_artery.nii.gz",
    "pulmonary_vein.nii.gz",
    "heart.nii.gz",
]


def _load_lobe_mask(
    seg_dir: str,
    selected_lobes: Sequence[str],
    target_shape: tuple,
) -> np.ndarray:
    """Load and combine lobe segmentations into a single binary mask."""
    combined = np.zeros(target_shape, dtype=bool)
    for lobe_key in selected_lobes:
        fname = LOBE_FILENAMES.get(lobe_key)
        if fname is None:
            continue
        fpath = os.path.join(seg_dir, fname)
        if not os.path.exists(fpath):
            logger.debug("Lobe file not found: %s", fpath)
            continue
        try:
            img = nib.load(fpath)
            data = np.asarray(img.get_fdata(), dtype=bool)
            if data.shape != target_shape:
                data = _resample_to(data.astype(np.float32),
                                     data.shape, target_shape, order=0) > 0.5
            combined |= data
        except Exception as exc:
            logger.warning("Could not load lobe mask %s: %s", fpath, exc)
    return combined


def _build_constraint_mask(
    seg_dir: str,
    target_shape: tuple,
    selected_lobes: Sequence[str],
    exclusion_dilation_mm: float,
    ct_affine: np.ndarray,
) -> np.ndarray:
    """Build the constraint mask = lung lobes minus exclusion structures."""
    lobe_mask = _load_lobe_mask(seg_dir, selected_lobes, target_shape)

    if not lobe_mask.any():
        logger.warning("No lobe masks found; using full volume as constraint")
        lobe_mask = np.ones(target_shape, dtype=bool)

    # Exclusion structures (large vessels, trachea, oesophagus)
    exclusion_files = [
        "trachea.nii.gz",
        "esophagus.nii.gz",
        "aorta.nii.gz",
    ]
    exclusion = np.zeros(target_shape, dtype=bool)
    spacing = _voxel_spacing(ct_affine)
    dilation_vox = max(1, int(round(exclusion_dilation_mm / spacing.mean())))

    for fname in exclusion_files:
        fpath = os.path.join(seg_dir, fname)
        if not os.path.exists(fpath):
            continue
        try:
            img = nib.load(fpath)
            data = np.asarray(img.get_fdata(), dtype=bool)
            if data.shape != target_shape:
                data = _resample_to(data.astype(np.float32),
                                     data.shape, target_shape, order=0) > 0.5
            if exclusion_dilation_mm > 0:
                data = binary_dilation(data, iterations=dilation_vox)
            exclusion |= data
        except Exception as exc:
            logger.debug("Could not load exclusion structure %s: %s", fname, exc)

    constraint = lobe_mask & ~exclusion
    return constraint.astype(bool)


# ---------------------------------------------------------------------------
# Hilar exclusion zone
# ---------------------------------------------------------------------------

def _build_hilar_exclusion(
    seg_dir: str,
    target_shape: tuple,
    hilar_radius_mm: float,
    ct_affine: np.ndarray,
) -> np.ndarray:
    """Create a hilar exclusion zone by dilating pulmonary hilum structures."""
    spacing = _voxel_spacing(ct_affine)
    dilation_vox = max(1, int(round(hilar_radius_mm / spacing.mean())))
    hilar = np.zeros(target_shape, dtype=bool)

    for fname in HILAR_STRUCTURES:
        fpath = os.path.join(seg_dir, fname)
        if not os.path.exists(fpath):
            continue
        try:
            img = nib.load(fpath)
            data = np.asarray(img.get_fdata(), dtype=bool)
            if data.shape != target_shape:
                data = _resample_to(data.astype(np.float32),
                                     data.shape, target_shape, order=0) > 0.5
            hilar |= data
        except Exception as exc:
            logger.debug("Could not load hilar structure %s: %s", fname, exc)

    if hilar.any():
        hilar = binary_dilation(hilar, iterations=dilation_vox)

    return hilar.astype(bool)


# ---------------------------------------------------------------------------
# Random walker segmentation
# ---------------------------------------------------------------------------

def _run_random_walker(
    pet_data: np.ndarray,
    constraint_mask: np.ndarray,
    fg_frac: float = 0.7,
    bg_frac: float = 0.1,
    beta: float = 130.0,
    tolerance: float = 1e-3,
) -> np.ndarray:
    """Run random walker segmentation on the PET image within the constraint.

    Seeds are set based on PET intensity percentiles within the constraint mask.

    Returns binary tumor mask.
    """
    try:
        from skimage.segmentation import random_walker
    except ImportError:
        logger.warning("scikit-image not available; using threshold segmentation")
        return _threshold_segment(pet_data, constraint_mask, fg_frac)

    pet_in = pet_data * constraint_mask
    if not constraint_mask.any() or pet_in.max() == 0:
        return np.zeros_like(pet_data, dtype=bool)

    vals = pet_in[constraint_mask]
    fg_threshold = float(np.percentile(vals, 100 * (1 - fg_frac)))
    bg_threshold = float(np.percentile(vals, 100 * bg_frac))

    labels = np.zeros_like(pet_data, dtype=np.int32)
    # Foreground seeds = high PET within constraint
    labels[(pet_in >= fg_threshold) & constraint_mask] = 2
    # Background seeds = low PET within constraint
    labels[(pet_in <= bg_threshold) & constraint_mask] = 1
    # Outside constraint = background
    labels[~constraint_mask] = 1

    if not (labels == 2).any():
        logger.warning("No foreground seeds found; falling back to threshold")
        return _threshold_segment(pet_data, constraint_mask, fg_frac)

    try:
        logger.info("Running random walker (beta=%.1f, tol=%.1e)…", beta, tolerance)
        result = random_walker(
            pet_in.astype(np.float64),
            labels,
            beta=beta,
            tol=tolerance,
            mode="bf",
        )
        tumor_mask = (result == 2) & constraint_mask
        return tumor_mask.astype(bool)
    except Exception as exc:
        logger.warning("Random walker failed (%s); falling back to threshold", exc)
        return _threshold_segment(pet_data, constraint_mask, fg_frac)


def _threshold_segment(
    pet_data: np.ndarray,
    constraint_mask: np.ndarray,
    fg_frac: float = 0.7,
) -> np.ndarray:
    """Simple threshold segmentation as fallback."""
    pet_in = pet_data * constraint_mask
    if not constraint_mask.any() or pet_in.max() == 0:
        return np.zeros_like(pet_data, dtype=bool)
    vals = pet_in[constraint_mask]
    threshold = float(np.percentile(vals, 100 * (1 - fg_frac)))
    return (pet_in >= threshold) & constraint_mask


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def _postprocess_mask(
    raw_mask: np.ndarray,
    min_vol_mm3: float = 100.0,
    spacing: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Clean up segmentation mask: fill holes, remove small components."""
    if not raw_mask.any():
        return raw_mask.astype(bool)

    # Close small holes
    closed = binary_dilation(raw_mask, iterations=2)
    closed = binary_erosion(closed, iterations=2)

    if spacing is None:
        spacing = np.ones(3)

    voxel_vol = float(np.prod(spacing))
    min_vox = max(1, int(min_vol_mm3 / voxel_vol))

    labeled, n = ndi_label(closed)
    cleaned = np.zeros_like(raw_mask, dtype=bool)
    for idx in range(1, n + 1):
        comp = labeled == idx
        if comp.sum() >= min_vox:
            cleaned |= comp

    if not cleaned.any():
        return raw_mask.astype(bool)
    return cleaned


# ---------------------------------------------------------------------------
# Node / primary separation
# ---------------------------------------------------------------------------

def _separate_nodes(
    tumor_mask: np.ndarray,
    pet_data: np.ndarray,
    spacing: np.ndarray,
    min_sep_dist_mm: float = 10.0,
) -> np.ndarray:
    """Attempt to separate lymph nodes from the primary tumor.

    Returns the component most likely to be the primary tumor
    (largest volume × highest SUVmax).
    """
    labeled, n = ndi_label(tumor_mask)
    if n <= 1:
        return tumor_mask.astype(bool)

    voxel_vol = float(np.prod(spacing))
    best_comp = None
    best_score = -1.0

    for idx in range(1, n + 1):
        comp = labeled == idx
        vol_cm3 = comp.sum() * voxel_vol / 1000.0
        suv_max = float(pet_data[comp].max())
        # Score: volume^0.5 × SUVmax (favour large + hot)
        score = (vol_cm3 ** 0.5) * suv_max
        if score > best_score:
            best_score = score
            best_comp = comp

    if best_comp is None:
        return tumor_mask.astype(bool)

    logger.info("Node separation: retained primary component (score=%.2f)", best_score)
    return best_comp.astype(bool)


# ---------------------------------------------------------------------------
# Side determination
# ---------------------------------------------------------------------------

def _determine_side(
    tumor_mask: np.ndarray,
    ct_affine: np.ndarray,
    requested_side: str = "auto",
) -> str:
    """Determine which lung side the tumor is on."""
    if requested_side in ("left", "right"):
        return requested_side

    if not tumor_mask.any():
        return "right"

    coords = np.argwhere(tumor_mask)
    centroid_i = float(coords[:, 0].mean())
    mid_i = tumor_mask.shape[0] / 2.0

    # Determine L/R from affine orientation
    # If affine[0,0] > 0: increasing i → increasing x (left in LPS)
    if ct_affine[0, 0] > 0:
        side = "left" if centroid_i < mid_i else "right"
    else:
        side = "right" if centroid_i < mid_i else "left"

    logger.info("Auto-detected tumor side: %s", side)
    return side


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def process_case(
    pet_path: str,
    ct_path: str,
    out_dir: str,
    side: str = "auto",
    device: str = "cpu",
    fast: bool = False,
    exclusion_dilation_mm: float = 5.0,
    hilar_radius_mm: float = 20.0,
    separate_nodes: bool = True,
    fg_frac: float = 0.7,
    bg_frac: float = 0.1,
    beta: float = 130.0,
    tolerance: float = 1e-3,
    selected_lobes: Optional[List[str]] = None,
) -> Dict:
    """Run the complete lung tumor segmentation and radiomics pipeline.

    Parameters
    ----------
    pet_path : str
        Path to PET NIfTI file (SUV-normalised).
    ct_path : str
        Path to CT NIfTI file (HU).
    out_dir : str
        Output directory for all results.
    side : str
        Tumour side: 'left', 'right', or 'auto'.
    device : str
        TotalSegmentator device: 'cpu', 'cuda', 'mps'.
    fast : bool
        Use fast TotalSegmentator mode.
    exclusion_dilation_mm : float
        Dilation (mm) applied to exclusion structures.
    hilar_radius_mm : float
        Radius (mm) of the hilar exclusion zone.
    separate_nodes : bool
        Whether to separate lymph nodes from the primary tumor.
    fg_frac : float
        Fraction of highest-PET voxels used as foreground seeds.
    bg_frac : float
        Fraction of lowest-PET voxels used as background seeds.
    beta : float
        Random walker beta parameter.
    tolerance : float
        Random walker solver tolerance.
    selected_lobes : list of str, optional
        Lobe keys to include. If None, all lobes are used.

    Returns
    -------
    metrics : dict
        All computed radiomics metrics.
    """
    out_dir_path = ensure_dir(out_dir)
    seg_dir = str(out_dir_path / "totalseg")
    ensure_dir(seg_dir)

    # -------------------------------------------------------------------
    # Load PET and CT
    # -------------------------------------------------------------------
    logger.info("Loading PET: %s", pet_path)
    logger.info("Loading CT:  %s", ct_path)
    pet_img = nib.load(pet_path)
    ct_img = nib.load(ct_path)
    pet_data = np.asarray(pet_img.get_fdata(), dtype=np.float32)
    ct_data_raw = np.asarray(ct_img.get_fdata(), dtype=np.float32)
    pet_affine = pet_img.affine
    ct_affine = ct_img.affine
    pet_spacing = _voxel_spacing(pet_affine)

    # Resample CT to PET space
    ct_data = _resample_to(ct_data_raw, ct_data_raw.shape, pet_data.shape, order=1)

    # -------------------------------------------------------------------
    # Step 1 — TotalSegmentator
    # -------------------------------------------------------------------
    logger.info("Step 1: TotalSegmentator…")
    ts_ok = _run_totalsegmentator(ct_path, seg_dir, device=device, fast=fast)

    # -------------------------------------------------------------------
    # Step 2 — Constraint mask
    # -------------------------------------------------------------------
    if selected_lobes is None:
        if side == "auto":
            # Use all lobes; we'll filter by side later
            selected_lobes = list(LOBE_FILENAMES.keys())
        else:
            selected_lobes = SIDE_LOBES.get(side, list(LOBE_FILENAMES.keys()))

    logger.info("Step 2: Building constraint mask (lobes: %s)…", selected_lobes)
    if ts_ok:
        constraint_mask = _build_constraint_mask(
            seg_dir, pet_data.shape, selected_lobes,
            exclusion_dilation_mm, ct_affine,
        )
        hilar_excl = _build_hilar_exclusion(
            seg_dir, pet_data.shape, hilar_radius_mm, ct_affine
        )
    else:
        # Fallback: use lung-intensity HU region as constraint
        logger.warning("Using CT intensity for constraint (TotalSegmentator unavailable)")
        constraint_mask = (ct_data >= -900) & (ct_data <= -100)
        hilar_excl = np.zeros_like(constraint_mask, dtype=bool)

    # Apply hilar exclusion
    constraint_mask = constraint_mask & ~hilar_excl

    # -------------------------------------------------------------------
    # Step 3 — Determine side & filter by side
    # -------------------------------------------------------------------
    logger.info("Step 3: Determining side…")
    if side == "auto":
        # Quick pre-segmentation to find centroid
        quick_mask = (pet_data * constraint_mask >= np.percentile(
            pet_data[constraint_mask], 90
        ) if constraint_mask.any() else np.zeros_like(pet_data, dtype=bool))
        side = _determine_side(quick_mask, ct_affine, "auto")
        # Re-filter lobes by detected side
        side_lobes = SIDE_LOBES.get(side, list(LOBE_FILENAMES.keys()))
        if ts_ok:
            constraint_mask = _build_constraint_mask(
                seg_dir, pet_data.shape, side_lobes,
                exclusion_dilation_mm, ct_affine,
            )
            constraint_mask = constraint_mask & ~hilar_excl

    # -------------------------------------------------------------------
    # Step 4 — Random walker segmentation
    # -------------------------------------------------------------------
    logger.info("Step 4: Random walker segmentation…")
    raw_tumor = _run_random_walker(
        pet_data, constraint_mask,
        fg_frac=fg_frac, bg_frac=bg_frac,
        beta=beta, tolerance=tolerance,
    )

    # -------------------------------------------------------------------
    # Step 5 — Post-processing
    # -------------------------------------------------------------------
    logger.info("Step 5: Post-processing…")
    tumor_mask = _postprocess_mask(raw_tumor, min_vol_mm3=100.0, spacing=pet_spacing)

    # -------------------------------------------------------------------
    # Step 6 — Node separation
    # -------------------------------------------------------------------
    if separate_nodes and tumor_mask.any():
        logger.info("Step 6: Node separation…")
        tumor_mask = _separate_nodes(tumor_mask, pet_data, pet_spacing)
    else:
        logger.info("Step 6: Node separation skipped")

    # -------------------------------------------------------------------
    # Step 7 — Save tumor mask
    # -------------------------------------------------------------------
    tumor_nifti_path = str(out_dir_path / "tumor_mask.nii.gz")
    logger.info("Step 7: Saving tumor mask → %s", tumor_nifti_path)
    tumor_img = nib.Nifti1Image(
        tumor_mask.astype(np.uint8), pet_affine, pet_img.header
    )
    nib.save(tumor_img, tumor_nifti_path)

    # Save constraint mask for QC
    constraint_nifti_path = str(out_dir_path / "constraint_mask.nii.gz")
    constraint_img = nib.Nifti1Image(
        constraint_mask.astype(np.uint8), pet_affine
    )
    nib.save(constraint_img, constraint_nifti_path)

    # -------------------------------------------------------------------
    # Step 8 — Compute metrics
    # -------------------------------------------------------------------
    logger.info("Step 8: Computing radiomics metrics…")
    try:
        import advanced_metrics
        metrics, qc_coords = advanced_metrics.compute_all_metrics(
            pet_nifti_path=pet_path,
            tumor_mask_path=tumor_nifti_path,
            ct_nifti_path=ct_path,
        )
    except Exception as exc:
        logger.error("Metrics computation failed: %s", exc)
        metrics = {}
        qc_coords = {}

    # -------------------------------------------------------------------
    # Step 9 — 3×3 QC grid
    # -------------------------------------------------------------------
    qc_grid_path = str(out_dir_path / "Mask_QC.png")
    logger.info("Step 9: Generating 3×3 QC grid → %s", qc_grid_path)
    try:
        import Mask_QC
        Mask_QC.generate_qc_overlays(
            pet_nifti_path=pet_path,
            ct_nifti_path=ct_path,
            tumor_mask_path=tumor_nifti_path,
            constraint_mask_path=constraint_nifti_path,
            output_path=qc_grid_path,
        )
    except Exception as exc:
        logger.error("generate_qc_overlays failed: %s", exc)

    # -------------------------------------------------------------------
    # Step 10 — Per-metric QC images
    # -------------------------------------------------------------------
    logger.info("Step 10: Generating per-metric QC images…")
    try:
        import Mask_QC  # noqa: F811 (re-import OK)
        Mask_QC.generate_metric_qc_overlays(
            pet_nifti_path=pet_path,
            ct_nifti_path=ct_path,
            tumor_mask_path=tumor_nifti_path,
            output_dir=str(out_dir_path),
            metrics=metrics,
            qc_coords=qc_coords,
        )
    except Exception as exc:
        logger.error("generate_metric_qc_overlays failed: %s", exc)

    # -------------------------------------------------------------------
    # Step 11 — Save metrics JSON + CSV
    # -------------------------------------------------------------------
    metrics_json_path = str(out_dir_path / "metrics.json")
    metrics_csv_path = str(out_dir_path / "metrics.csv")
    logger.info("Step 11: Saving metrics → %s", metrics_json_path)

    try:
        with open(metrics_json_path, "w") as fh:
            json.dump(metrics, fh, indent=2)
    except Exception as exc:
        logger.error("Could not save metrics JSON: %s", exc)

    try:
        with open(metrics_csv_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["metric", "value"])
            for k, v in metrics.items():
                writer.writerow([k, v])
    except Exception as exc:
        logger.error("Could not save metrics CSV: %s", exc)

    # -------------------------------------------------------------------
    # Step 12 — Return
    # -------------------------------------------------------------------
    logger.info("Pipeline complete.  Output directory: %s", out_dir)
    return metrics


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Lung-ASP: Lung Tumor Segmentation & Radiomics Pipeline"
    )
    parser.add_argument("pet", help="PET NIfTI path (SUV)")
    parser.add_argument("ct", help="CT NIfTI path (HU)")
    parser.add_argument("out_dir", help="Output directory")
    parser.add_argument("--side", default="auto",
                        choices=["auto", "left", "right"])
    parser.add_argument("--device", default="cpu",
                        choices=["cpu", "cuda", "mps"])
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--excl-dilation", type=float, default=5.0,
                        dest="exclusion_dilation_mm")
    parser.add_argument("--hilar-radius", type=float, default=20.0,
                        dest="hilar_radius_mm")
    parser.add_argument("--no-node-sep", action="store_false", dest="separate_nodes")
    parser.add_argument("--fg-frac", type=float, default=0.7, dest="fg_frac")
    parser.add_argument("--bg-frac", type=float, default=0.1, dest="bg_frac")
    parser.add_argument("--beta", type=float, default=130.0)
    parser.add_argument("--tol", type=float, default=1e-3, dest="tolerance")

    args = parser.parse_args()

    result = process_case(
        pet_path=args.pet,
        ct_path=args.ct,
        out_dir=args.out_dir,
        side=args.side,
        device=args.device,
        fast=args.fast,
        exclusion_dilation_mm=args.exclusion_dilation_mm,
        hilar_radius_mm=args.hilar_radius_mm,
        separate_nodes=args.separate_nodes,
        fg_frac=args.fg_frac,
        bg_frac=args.bg_frac,
        beta=args.beta,
        tolerance=args.tolerance,
    )

    print(json.dumps(result, indent=2))
