"""
advanced_metrics.py  v3.7
FDG PET/CT Lung Tumor Radiomics — Primary Isolation + Full-Mask SUV + Geometric Metrics
"""

import logging
import numpy as np
from scipy import ndimage
from scipy.ndimage import uniform_filter

logger = logging.getLogger(__name__)


def _bottleneck(clean, pet_data, ct_data, spacing, smi, smv, os):
    s = ndimage.generate_binary_structure(3, 1)
    ni = max(int(round(1.0 / float(min(spacing)))), 1)
    er = ndimage.binary_erosion(clean, structure=s, iterations=ni)
    if er.sum() < 5:
        return None
    le, ne = ndimage.label(er, structure=s)
    if ne < 2:
        return None
    sl = le[smi]
    if sl == 0:
        cc = np.argwhere(er)
        dd = np.sum((cc - np.array(smi)) ** 2, axis=1)
        sl = le[tuple(cc[np.argmin(dd)])]
    if sl == 0:
        return None
    pe = (le == sl)
    pr = ndimage.binary_dilation(pe, structure=s, iterations=ni) & clean
    pr = ndimage.binary_fill_holes(pr) & clean
    if pr.sum() >= 20 and pr.sum() < os * 0.88 and pr[smi]:
        logger.info(f"  S0 (bottleneck): kept={pr.sum():,}")
        return pr.astype(bool)
    return None


def _convex_hull_trim(clean, ct_data, pet_data, spacing, smi, smv, os):
    try:
        from skimage.morphology import convex_hull_image
    except ImportError:
        return None
    s = ndimage.generate_binary_structure(3, 1)
    thr = smv * 0.40
    hot = clean & (pet_data >= thr)
    if hot.sum() < 5:
        hot = clean.copy()
    hull = np.zeros_like(clean, dtype=bool)
    for k in range(clean.shape[2]):
        sl = hot[:, :, k]
        if sl.sum() < 3:
            hull[:, :, k] = sl
            continue
        try:
            hull[:, :, k] = convex_hull_image(sl)
        except Exception:
            hull[:, :, k] = sl
    pr = hull & clean
    le, ne = ndimage.label(pr, structure=s)
    if ne > 1:
        sl2 = le[smi]
        if sl2 == 0:
            cc = np.argwhere(pr)
            dd = np.sum((cc - np.array(smi)) ** 2, axis=1)
            sl2 = le[tuple(cc[np.argmin(dd)])]
        if sl2 > 0:
            pr = (le == sl2)
    pr = ndimage.binary_fill_holes(pr) & clean
    if pr.sum() >= 20 and pr.sum() < os * 0.88 and pr[smi]:
        logger.info(f"  S1 (convex hull): kept={pr.sum():,}")
        return pr.astype(bool)
    return None


def _lung_adjacency_filter(clean, ct_data, pet_data, spacing, smi, os, pm):
    s = ndimage.generate_binary_structure(3, 1)
    rv = max(int(round(15.0 / float(min(spacing)))), 2)
    lung_air = (ct_data < -300.0).astype(np.float32)
    lf = uniform_filter(lung_air, size=2 * rv + 1, mode='constant')
    lung_adj = clean & (lf > 0.02)
    sv = np.zeros_like(clean, dtype=bool)
    sv[smi] = True
    ng = max(int(round(8.0 / float(min(spacing)))), 2)
    sv = ndimage.binary_dilation(sv, structure=s, iterations=ng) & clean
    keep = lung_adj | sv
    if keep.sum() < 20:
        return None
    le, ne = ndimage.label(keep, structure=s)
    sl = le[smi]
    if sl == 0:
        cc = np.argwhere(keep)
        dd = np.sum((cc - np.array(smi)) ** 2, axis=1)
        sl = le[tuple(cc[np.argmin(dd)])]
    if sl == 0:
        return None
    pr = (le == sl).astype(bool)
    ni = max(int(round(3.0 / float(min(spacing)))), 1)
    pr = ndimage.binary_dilation(pr, structure=s, iterations=ni) & clean
    pr = ndimage.binary_fill_holes(pr) & clean
    rm = os - pr.sum()
    if rm < 20 or pr.sum() < os * 0.20:
        return None
    if not pr[smi]:
        return None
    logger.info(f"  S2 (lung adj): kept={pr.sum():,}, removed={rm:,}")
    return pr.astype(bool)


def _erosion_isolation(clean, spacing, smi, os):
    s = ndimage.generate_binary_structure(3, 1)
    for em in (2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20, 25):
        ni = max(int(round(float(em) / float(min(spacing)))), 1)
        er = ndimage.binary_erosion(clean, structure=s, iterations=ni)
        if er.sum() < 20:
            continue
        le, ne = ndimage.label(er, structure=s)
        if ne < 2:
            continue
        sl = le[smi]
        if sl == 0:
            cc = np.argwhere(er)
            dd = np.sum((cc - np.array(smi)) ** 2, axis=1)
            sl = le[tuple(cc[np.argmin(dd)])]
        if sl == 0:
            continue
        pe = (le == sl)
        if pe.sum() / max(er.sum(), 1) < 0.15:
            continue
        pr = ndimage.binary_dilation(pe, structure=s, iterations=ni) & clean
        pr = ndimage.binary_fill_holes(pr) & clean
        if pr.sum() >= 20 and pr.sum() < os * 0.88:
            logger.info(f"  S3 (erosion={em}mm): kept={pr.sum():,}")
            return pr.astype(bool)
    return None


def _pet_distance_isolation(clean, pet_data, spacing, smi, smv, os):
    s = ndimage.generate_binary_structure(3, 1)
    vm = float(os) * float(np.prod(spacing))
    R = (3.0 * vm / (4.0 * np.pi)) ** (1.0 / 3.0)
    mc = np.argwhere(clean)
    sa = np.array(smi, dtype=float)
    dm = np.sqrt(np.sum(((mc - sa) * spacing) ** 2, axis=1))
    pv = pet_data[mc[:, 0], mc[:, 1], mc[:, 2]]
    pw = np.clip(pv / max(smv, 1.0), 0, 1)
    ed = dm * (2.0 - pw)
    keep = ed <= R * 1.0
    if keep.sum() < 20 or keep.sum() >= os * 0.88:
        return None
    pd = np.zeros(clean.shape, dtype=bool)
    kc = mc[keep]
    pd[kc[:, 0], kc[:, 1], kc[:, 2]] = True
    ld, nd = ndimage.label(pd, structure=s)
    if nd > 1:
        dl = ld[smi]
        if dl == 0:
            dc = np.argwhere(pd)
            dd = np.sum((dc - sa) ** 2, axis=1)
            dl = ld[tuple(dc[np.argmin(dd)])]
        if dl > 0:
            pd = (ld == dl)
    pd = ndimage.binary_fill_holes(pd) & clean
    if pd.sum() >= 20:
        logger.info(f"  S4 (distance): kept={pd.sum():,}")
        return pd.astype(bool)
    return None


def _ct_density_filter(clean, ct_data, spacing, smi, os):
    s = ndimage.generate_binary_structure(3, 1)
    rv = max(int(round(10.0 / float(min(spacing)))), 1)
    lf = uniform_filter(
        (ct_data < -500.0).astype(np.float32), size=2 * rv + 1, mode='constant')
    keep = clean & (lf >= 0.05)
    sv = np.zeros_like(clean, dtype=bool)
    sv[smi] = True
    ng = max(int(round(10.0 / float(min(spacing)))), 2)
    sv = ndimage.binary_dilation(sv, structure=s, iterations=ng) & clean
    keep = keep | sv
    if keep.sum() < 20:
        return None
    lk, _ = ndimage.label(keep, structure=s)
    sl = lk[smi]
    if sl == 0:
        cc = np.argwhere(keep)
        dd = np.sum((cc - np.array(smi)) ** 2, axis=1)
        sl = lk[tuple(cc[np.argmin(dd)])]
    if sl == 0:
        return None
    pk = (lk == sl).astype(bool)
    ni = max(int(round(2.0 / float(min(spacing)))), 1)
    pk = ndimage.binary_dilation(pk, structure=s, iterations=ni) & clean
    pk = ndimage.binary_fill_holes(pk) & clean
    rm = os - pk.sum()
    if rm < 20 or pk.sum() < os * 0.20:
        return None
    if not pk[smi]:
        return None
    logger.info(f"  S5 (CT filter): kept={pk.sum():,}, removed={rm:,}")
    return pk.astype(bool)


def _isolate_primary_inner(tumor_mask, pet_data, ct_data, spacing):
    s = ndimage.generate_binary_structure(3, 1)
    lc, nc = ndimage.label(tumor_mask, structure=s)
    if nc == 0:
        return tumor_mask.astype(bool)
    sizes = ndimage.sum(tumor_mask, lc, range(1, nc + 1))
    clean = (lc == (np.argmax(sizes) + 1)).astype(bool)
    if nc == 1:
        return clean
    masked_pet = pet_data * tumor_mask
    smi = tuple(np.unravel_index(np.argmax(masked_pet), masked_pet.shape))
    smv = float(masked_pet[smi])
    os = int(clean.sum())
    full_clean = tumor_mask.astype(bool)
    pm = tumor_mask.astype(bool)
    for name, fn, args in [
        ("S0", _bottleneck,            (full_clean, pet_data, ct_data, spacing, smi, smv, os)),
        ("S1", _convex_hull_trim,      (full_clean, ct_data, pet_data, spacing, smi, smv, os)),
        ("S2", _lung_adjacency_filter, (full_clean, ct_data, pet_data, spacing, smi, os, pm)),
        ("S3", _erosion_isolation,     (full_clean, spacing, smi, os)),
        ("S4", _pet_distance_isolation,(full_clean, pet_data, spacing, smi, smv, os)),
        ("S5", _ct_density_filter,     (full_clean, ct_data, spacing, smi, os)),
    ]:
        try:
            result = fn(*args)
            if result is not None and result.sum() >= 20:
                logger.info(f"Primary isolation succeeded with {name}")
                return result
        except Exception as e:
            logger.warning(f"  {name} failed: {e}")
    logger.info("  All strategies failed, using largest component")
    return clean


def _isolate_primary(tumor_mask, pet_data, ct_data, spacing):
    try:
        result = _isolate_primary_inner(tumor_mask, pet_data, ct_data, spacing)
        if result is not None and result.sum() >= 20:
            return result
    except Exception as e:
        logger.warning(f"Primary isolation failed: {e}")
    return tumor_mask.astype(bool)


def _compute_suv_metrics(pet_data, tumor_mask, spacing):
    if tumor_mask.sum() == 0:
        return {"SUVmax": 0.0, "SUVmean": 0.0, "SUVpeak": 0.0, "MTV": 0.0, "TLG": 0.0}
    vox_vol_cc = float(np.prod(spacing)) / 1000.0
    vals = pet_data[tumor_mask > 0]
    suvmax = float(np.max(vals))
    suvmean = float(np.mean(vals))
    mtv = float(tumor_mask.sum()) * vox_vol_cc
    tlg = suvmean * mtv
    masked_pet = pet_data * tumor_mask
    smi = np.unravel_index(np.argmax(masked_pet), masked_pet.shape)
    nz, ny, nx = np.ogrid[:pet_data.shape[0], :pet_data.shape[1], :pet_data.shape[2]]
    dist = np.sqrt(
        ((nz - smi[0]) * spacing[0]) ** 2 +
        ((ny - smi[1]) * spacing[1]) ** 2 +
        ((nx - smi[2]) * spacing[2]) ** 2
    )
    sphere = (dist <= 10.0) & (tumor_mask > 0)
    if sphere.sum() > 0:
        suvpeak = float(np.mean(pet_data[sphere]))
    else:
        suvpeak = suvmax
    return {
        "SUVmax": round(suvmax, 4),
        "SUVmean": round(suvmean, 4),
        "SUVpeak": round(suvpeak, 4),
        "MTV": round(mtv, 4),
        "TLG": round(tlg, 4),
    }


def _compute_geometric_metrics(primary_mask, pet_data, spacing):
    s = ndimage.generate_binary_structure(3, 1)
    if primary_mask.sum() < 2:
        return {
            "Dmax_mm": 0.0, "NHOCmax_mm": 0.0, "NHOPmax_mm": 0.0,
            "Sphericity": 0.0, "Asphericity": 0.0,
            "qc_coords": {}
        }
    vox_vol_cc = float(np.prod(spacing)) / 1000.0
    mtv = float(primary_mask.sum()) * vox_vol_cc
    coords = np.argwhere(primary_mask)
    centroid_ijk = tuple(np.round(np.mean(coords, axis=0)).astype(int))
    masked_pet = pet_data * primary_mask
    hotspot_ijk = tuple(np.unravel_index(np.argmax(masked_pet), masked_pet.shape))
    hotspot_suv = float(masked_pet[hotspot_ijk])
    eroded = ndimage.binary_erosion(primary_mask, structure=s)
    boundary = primary_mask & ~eroded
    boundary_coords = np.argwhere(boundary)
    max_bc = 500
    if len(boundary_coords) > max_bc:
        # Deterministic uniform stride sampling to keep Dmax reproducible
        stride = max(len(boundary_coords) // max_bc, 1)
        bc_sample = boundary_coords[::stride][:max_bc]
    else:
        bc_sample = boundary_coords
    bc_phys = bc_sample * np.array(spacing)
    if len(bc_phys) >= 2:
        diffs = bc_phys[:, None, :] - bc_phys[None, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
        dmax_idx = np.unravel_index(np.argmax(dists), dists.shape)
        dmax_mm = float(dists[dmax_idx])
        dmax_pt_a = tuple(bc_sample[dmax_idx[0]])
        dmax_pt_b = tuple(bc_sample[dmax_idx[1]])
    else:
        dmax_mm = 0.0
        dmax_pt_a = hotspot_ijk
        dmax_pt_b = hotspot_ijk
    centroid_phys = np.array(centroid_ijk) * np.array(spacing)
    hotspot_phys = np.array(hotspot_ijk) * np.array(spacing)
    nhocmax_mm = float(np.linalg.norm(hotspot_phys - centroid_phys))
    if len(boundary_coords) > 0:
        bp_phys = boundary_coords * np.array(spacing)
        dists_to_boundary = np.sqrt(np.sum((bp_phys - hotspot_phys) ** 2, axis=1))
        nearest_idx = np.argmin(dists_to_boundary)
        nhopmax_mm = float(dists_to_boundary[nearest_idx])
        nearest_boundary_ijk = tuple(boundary_coords[nearest_idx])
    else:
        nhopmax_mm = 0.0
        nearest_boundary_ijk = hotspot_ijk
    sa_voxels = float(boundary.sum())
    # Approximate face area per boundary voxel: voxel volume^(2/3) gives a characteristic
    # surface area element (geometric mean of face areas for anisotropic voxels).
    face_area = float(np.prod(spacing)) ** (2.0 / 3.0)
    sa_mm2 = sa_voxels * face_area
    vol_mm3 = float(primary_mask.sum()) * float(np.prod(spacing))
    if sa_mm2 > 0:
        sphericity = (np.pi ** (1.0 / 3.0) * (6.0 * vol_mm3) ** (2.0 / 3.0)) / sa_mm2
    else:
        sphericity = 0.0
    asphericity = (1.0 / sphericity - 1.0) if sphericity > 0 else 0.0
    qc_coords = {
        "hotspot_ijk": hotspot_ijk,
        "hotspot_suv": hotspot_suv,
        "centroid_ijk": centroid_ijk,
        "nearest_boundary_ijk": nearest_boundary_ijk,
        "dmax_pt_a": dmax_pt_a,
        "dmax_pt_b": dmax_pt_b,
        "dmax_mm": dmax_mm,
        "nhopmax_mm": nhopmax_mm,
        "nhocmax_mm": nhocmax_mm,
    }
    return {
        "Dmax_mm": round(dmax_mm, 2),
        "NHOCmax_mm": round(nhocmax_mm, 2),
        "NHOPmax_mm": round(nhopmax_mm, 2),
        "Sphericity": round(sphericity, 4),
        "Asphericity": round(asphericity, 4),
        "qc_coords": qc_coords,
    }


def _compute_getu(pet_data, tumor_mask, spacing, suv_metrics, geometric_metrics):
    dmax = geometric_metrics.get("Dmax_mm", 0.0)
    mtv = suv_metrics.get("MTV", 0.0)
    suvmean = suv_metrics.get("SUVmean", 0.0)
    suvmax = suv_metrics.get("SUVmax", 0.0)
    tlg = suv_metrics.get("TLG", 0.0)
    # mtv_threshold: 40% SUVmax threshold used as the MTV display cutoff for gETU
    mtv_threshold = suvmax * 0.40
    if dmax > 0 and mtv > 0:
        sphere_vol = (4.0 / 3.0) * np.pi * (dmax / 2.0) ** 3 / 1000.0  # mm³ → cc
        getu_index = tlg / max(sphere_vol, 1e-6)
        getu_raw = suvmean * (mtv / max(sphere_vol, 1e-6))
    else:
        getu_index = 0.0
        getu_raw = 0.0
    return {
        "gETU_index": round(getu_index, 4),
        "gETU_raw": round(getu_raw, 4),
        "mtv_threshold": round(mtv_threshold, 4),
    }


def compute_all_metrics(pet_data, tumor_mask, spacing, ct_data=None, ct_nifti_path=None):
    """Compute all PET/CT radiomics metrics for lung tumor segmentation.

    Parameters
    ----------
    pet_data : np.ndarray
        PET SUV data array
    tumor_mask : np.ndarray
        Binary tumor mask
    spacing : tuple/list
        Voxel spacing in mm (i, j, k)
    ct_data : np.ndarray, optional
        CT Hounsfield unit data array
    ct_nifti_path : str, optional
        Path to CT NIfTI file (used if ct_data not provided)

    Returns
    -------
    dict
        All computed metrics including qc_coords for QC overlay generation
    """
    logger.info("Computing all metrics...")
    metrics = {}

    if ct_data is None and ct_nifti_path is not None:
        try:
            import nibabel as nib
            ct_nii = nib.load(ct_nifti_path)
            ct_data = ct_nii.get_fdata().astype(np.float32)
        except Exception as e:
            logger.warning(f"Could not load CT from {ct_nifti_path}: {e}")

    tumor_mask = (tumor_mask > 0)

    if tumor_mask.sum() == 0:
        logger.warning("Empty tumor mask, returning zero metrics")
        return {
            "SUVmax": 0.0, "SUVmean": 0.0, "SUVpeak": 0.0,
            "MTV": 0.0, "TLG": 0.0,
            "Dmax_mm": 0.0, "NHOCmax_mm": 0.0, "NHOPmax_mm": 0.0,
            "Sphericity": 0.0, "Asphericity": 0.0,
            "gETU_index": 0.0, "gETU_raw": 0.0, "mtv_threshold": 0.0,
        }

    logger.info("Computing SUV metrics on full mask...")
    suv_metrics = _compute_suv_metrics(pet_data, tumor_mask, spacing)
    metrics.update(suv_metrics)

    logger.info("Isolating primary tumor...")
    if ct_data is not None:
        primary_mask = _isolate_primary(tumor_mask, pet_data, ct_data, spacing)
    else:
        primary_mask = tumor_mask.copy()

    logger.info("Computing geometric metrics on primary...")
    geo_metrics = _compute_geometric_metrics(primary_mask, pet_data, spacing)
    metrics.update(geo_metrics)

    logger.info("Computing gETU...")
    getu_metrics = _compute_getu(pet_data, tumor_mask, spacing, suv_metrics, geo_metrics)
    metrics.update(getu_metrics)

    logger.info(f"Metrics: {', '.join(f'{k}={v}' for k, v in metrics.items() if k != 'qc_coords')}")
    return metrics
