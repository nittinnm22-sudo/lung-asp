"""advanced_metrics.py  v3.7
Complete FDG PET/CT Lung Tumor Radiomics — all metrics + primary isolation.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.ndimage import (
    binary_erosion,
    binary_dilation,
    distance_transform_edt,
    label as ndi_label,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _voxel_spacing_from_affine(affine: np.ndarray) -> np.ndarray:
    """Return voxel spacing (mm) from a NIfTI affine matrix."""
    return np.sqrt((affine[:3, :3] ** 2).sum(axis=0))


def _ijk_to_world(ijk: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Convert voxel indices to world coordinates."""
    ijk_h = np.append(np.asarray(ijk, dtype=float), 1.0)
    return (affine @ ijk_h)[:3]


def _surface_voxels(mask: np.ndarray) -> np.ndarray:
    """Return binary mask of surface (boundary) voxels."""
    eroded = binary_erosion(mask, iterations=1)
    return mask & ~eroded


def _largest_cc(mask: np.ndarray) -> np.ndarray:
    """Return the largest connected component of a binary mask."""
    labeled, n = ndi_label(mask)
    if n == 0:
        return mask.copy()
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    largest = int(np.argmax(sizes)) + 1
    return labeled == largest


def _validate_point_in_mask(
    pt: Tuple[int, int, int], mask: np.ndarray, label: str = ""
) -> Tuple[int, int, int]:
    """Ensure point is inside mask; snap to nearest mask voxel if not."""
    pi, pj, pk = int(round(pt[0])), int(round(pt[1])), int(round(pt[2]))
    shape = mask.shape
    pi = int(np.clip(pi, 0, shape[0] - 1))
    pj = int(np.clip(pj, 0, shape[1] - 1))
    pk = int(np.clip(pk, 0, shape[2] - 1))
    if mask[pi, pj, pk]:
        return (pi, pj, pk)
    # snap to nearest mask voxel
    coords = np.argwhere(mask)
    if coords.size == 0:
        return (pi, pj, pk)
    dists = np.sum((coords - np.array([pi, pj, pk])) ** 2, axis=1)
    nearest = coords[np.argmin(dists)]
    if label:
        logger.debug("%s snapped to nearest mask voxel %s", label, nearest)
    return (int(nearest[0]), int(nearest[1]), int(nearest[2]))


# ---------------------------------------------------------------------------
# Strategy 0 — Bottleneck Separation
# ---------------------------------------------------------------------------

def _bottleneck_separation(
    mask: np.ndarray, spacing: np.ndarray, min_size_vox: int = 50
) -> Optional[np.ndarray]:
    """Detect and cut bottleneck connections between tumor components.

    Uses distance-transform thinning to find narrow bridges and removes them.
    Returns the largest component after separation, or None if no useful split.
    """
    dt = distance_transform_edt(mask, sampling=spacing)
    # Threshold at a fraction of the max distance to find bottlenecks
    threshold = 0.35 * dt.max()
    bottleneck_mask = mask & (dt < threshold)
    # Remove the bottleneck region
    candidate = mask & ~bottleneck_mask
    labeled, n = ndi_label(candidate)
    if n < 2:
        return None
    sizes = ndimage.sum(candidate, labeled, range(1, n + 1))
    # Keep only regions large enough
    kept = np.zeros_like(mask)
    for idx, sz in enumerate(sizes, 1):
        if sz >= min_size_vox:
            kept |= labeled == idx
    if not kept.any():
        return None
    return _largest_cc(kept)


# ---------------------------------------------------------------------------
# Primary Tumor Isolation — Strategies S0–S5
# ---------------------------------------------------------------------------

def _convex_hull_trim_am(
    mask: np.ndarray,
    ct_data: np.ndarray,
    pet_data: np.ndarray,
    spacing: np.ndarray,
    min_size_vox: int,
    min_size_mm3: float,
    original_size: int,
) -> Optional[np.ndarray]:
    """S1 — Trim non-convex lobes using convex-hull-based approach."""
    try:
        from scipy.spatial import ConvexHull
    except ImportError:
        return None

    coords = np.argwhere(mask)
    if len(coords) < 4:
        return None
    try:
        hull = ConvexHull(coords * spacing)
    except Exception:
        return None

    # Find voxels inside convex hull
    from scipy.spatial import Delaunay
    try:
        tri = Delaunay(coords * spacing)
    except Exception:
        return None

    all_coords = np.argwhere(mask)
    all_world = all_coords * spacing
    inside = tri.find_simplex(all_world) >= 0
    hull_mask = np.zeros_like(mask)
    for c, flag in zip(all_coords, inside):
        if flag:
            hull_mask[c[0], c[1], c[2]] = True

    # Use PET signal to select the dominant lobe
    pet_in_hull = pet_data * hull_mask
    labeled, n = ndi_label(hull_mask)
    if n == 0:
        return None

    best_comp = None
    best_suv = -1.0
    for idx in range(1, n + 1):
        comp = labeled == idx
        sz = comp.sum()
        if sz < min_size_vox:
            continue
        mean_pet = float(pet_data[comp].mean())
        if mean_pet > best_suv:
            best_suv = mean_pet
            best_comp = comp

    if best_comp is None or best_comp.sum() < min_size_vox:
        return None
    return best_comp


def _lung_adjacency_am(
    mask: np.ndarray,
    ct_data: np.ndarray,
    pet_data: np.ndarray,
    spacing: np.ndarray,
    min_size_vox: int,
    original_size: int,
    pet_max: float,
) -> Optional[np.ndarray]:
    """S2 — Filter based on proximity to lung air (CT HU < -400)."""
    air_mask = ct_data < -400
    air_dt = distance_transform_edt(~air_mask, sampling=spacing)

    labeled, n = ndi_label(mask)
    if n == 0:
        return None

    best_comp = None
    best_score = -1.0
    for idx in range(1, n + 1):
        comp = labeled == idx
        sz = comp.sum()
        if sz < min_size_vox:
            continue
        # Score: mean PET × inverse mean air-distance
        mean_pet = float(pet_data[comp].mean())
        mean_air_dist = float(air_dt[comp].mean())
        score = mean_pet / (1.0 + mean_air_dist * 0.1)
        if score > best_score:
            best_score = score
            best_comp = comp

    if best_comp is None or best_comp.sum() < min_size_vox:
        return None
    return best_comp


def _erosion_am(
    mask: np.ndarray,
    spacing: np.ndarray,
    min_size_vox: int,
    original_size: int,
) -> Optional[np.ndarray]:
    """S3 — Progressive erosion to separate touching components."""
    eroded = mask.copy()
    for iterations in range(1, 6):
        eroded = binary_erosion(eroded, iterations=1)
        labeled, n = ndi_label(eroded)
        if n >= 2:
            sizes = ndimage.sum(eroded, labeled, range(1, n + 1))
            largest_idx = int(np.argmax(sizes)) + 1
            core = labeled == largest_idx
            # Dilate back
            grown = binary_dilation(core, iterations=iterations)
            result = mask & grown
            if result.sum() >= min_size_vox:
                return result
    return None


def _pet_distance_am(
    mask: np.ndarray,
    pet_data: np.ndarray,
    spacing: np.ndarray,
    min_size_vox: int,
    min_size_mm3: float,
    original_size: int,
) -> Optional[np.ndarray]:
    """S4 — PET-weighted distance-based isolation."""
    pet_in = pet_data * mask
    if pet_in.max() == 0:
        return None

    hotspot = np.unravel_index(pet_in.argmax(), pet_in.shape)
    dt = distance_transform_edt(mask, sampling=spacing)
    # Weight: high PET AND close to hotspot
    hotspot_dist = np.sqrt(
        ((np.indices(mask.shape).T - np.array(hotspot)) ** 2 * spacing ** 2).sum(axis=-1)
    )
    score = pet_in / (1.0 + hotspot_dist * 0.05)
    threshold = score[mask].mean()
    selected = mask & (score >= threshold)

    labeled, n = ndi_label(selected)
    if n == 0:
        return None
    sizes = ndimage.sum(selected, labeled, range(1, n + 1))
    largest_idx = int(np.argmax(sizes)) + 1
    result = labeled == largest_idx
    if result.sum() < min_size_vox:
        return None
    return result


def _ct_density_am(
    mask: np.ndarray,
    ct_data: np.ndarray,
    spacing: np.ndarray,
    min_size_vox: int,
    original_size: int,
) -> Optional[np.ndarray]:
    """S5 — CT density-based filtering (tumor tissue > -100 HU)."""
    solid_tissue = ct_data > -100
    candidate = mask & solid_tissue
    labeled, n = ndi_label(candidate)
    if n == 0:
        return None
    sizes = ndimage.sum(candidate, labeled, range(1, n + 1))
    largest_idx = int(np.argmax(sizes)) + 1
    result = labeled == largest_idx
    if result.sum() < min_size_vox:
        return None
    return result


def _isolate_primary_tumor(
    tumor_mask: np.ndarray,
    pet_data: np.ndarray,
    ct_data: np.ndarray,
    spacing: np.ndarray,
) -> np.ndarray:
    """Isolate the primary tumor from a possibly multi-component mask.

    Tries strategies S0–S5 in order; falls back to largest CC.
    """
    voxel_vol_mm3 = float(np.prod(spacing))
    n_vox = int(tumor_mask.sum())
    min_size_mm3 = 50.0  # minimum 50 mm³
    min_size_vox = max(10, int(min_size_mm3 / voxel_vol_mm3))
    pet_max = float(pet_data[tumor_mask > 0].max()) if tumor_mask.any() else 1.0

    # Morphological cleanup first
    clean = binary_erosion(tumor_mask, iterations=1)
    clean = binary_dilation(clean, iterations=1)
    clean = clean.astype(bool)
    if not clean.any():
        clean = tumor_mask.astype(bool)

    labeled, n = ndi_label(clean)
    if n == 1:
        return clean

    logger.debug("Isolating primary tumor: %d components found", n)

    # S0 — Bottleneck separation
    result = _bottleneck_separation(clean, spacing, min_size_vox)
    if result is not None and result.sum() >= min_size_vox:
        logger.debug("S0 (bottleneck) succeeded")
        return result.astype(bool)

    # S1 — Convex hull trim
    result = _convex_hull_trim_am(clean, ct_data, pet_data, spacing,
                                   min_size_vox, min_size_mm3, n_vox)
    if result is not None and result.sum() >= min_size_vox:
        logger.debug("S1 (convex hull) succeeded")
        return result.astype(bool)

    # S2 — Lung adjacency
    result = _lung_adjacency_am(clean, ct_data, pet_data, spacing,
                                 min_size_vox, n_vox, pet_max)
    if result is not None and result.sum() >= min_size_vox:
        logger.debug("S2 (lung adjacency) succeeded")
        return result.astype(bool)

    # S3 — Erosion isolation
    result = _erosion_am(clean, spacing, min_size_vox, n_vox)
    if result is not None and result.sum() >= min_size_vox:
        logger.debug("S3 (erosion) succeeded")
        return result.astype(bool)

    # S4 — PET distance isolation
    result = _pet_distance_am(clean, pet_data, spacing, min_size_vox,
                               min_size_mm3, n_vox)
    if result is not None and result.sum() >= min_size_vox:
        logger.debug("S4 (PET distance) succeeded")
        return result.astype(bool)

    # S5 — CT density filter
    result = _ct_density_am(clean, ct_data, spacing, min_size_vox, n_vox)
    if result is not None and result.sum() >= min_size_vox:
        logger.debug("S5 (CT density) succeeded")
        return result.astype(bool)

    # Fallback — largest CC by PET mean
    logger.debug("All strategies failed; falling back to highest-PET component")
    labeled, n = ndi_label(clean)
    best_comp = None
    best_pet = -1.0
    for idx in range(1, n + 1):
        comp = labeled == idx
        if comp.sum() < min_size_vox:
            continue
        mean_pet = float(pet_data[comp].mean())
        if mean_pet > best_pet:
            best_pet = mean_pet
            best_comp = comp
    if best_comp is not None:
        return best_comp.astype(bool)
    return _largest_cc(clean).astype(bool)


# ---------------------------------------------------------------------------
# SUVpeak — 1 cc sphere
# ---------------------------------------------------------------------------

def compute_suv_peak(
    pet_data: np.ndarray,
    tumor_mask: np.ndarray,
    spacing: np.ndarray,
) -> float:
    """Compute SUVpeak as mean SUV in a 1 cc sphere centred on the max voxel."""
    pet_in = pet_data * tumor_mask
    if not tumor_mask.any() or pet_in.max() == 0:
        return 0.0

    hotspot = np.unravel_index(pet_in.argmax(), pet_in.shape)
    voxel_vol_mm3 = float(np.prod(spacing))
    sphere_vol_mm3 = 1000.0  # 1 cc
    radius_mm = (3.0 * sphere_vol_mm3 / (4.0 * np.pi)) ** (1.0 / 3.0)

    # Build sphere mask
    grid = np.indices(pet_data.shape)
    dist_sq = sum(
        ((grid[k] - hotspot[k]) * spacing[k]) ** 2 for k in range(3)
    )
    sphere = dist_sq <= radius_mm ** 2
    inside = sphere & tumor_mask

    if not inside.any():
        return float(pet_data[hotspot])
    return float(pet_data[inside].mean())


# ---------------------------------------------------------------------------
# Dmax — maximum intra-tumoral distance
# ---------------------------------------------------------------------------

def compute_dmax_with_endpoints(
    tumor_mask: np.ndarray,
    spacing: np.ndarray,
) -> Tuple[float, Tuple[int, int, int], Tuple[int, int, int]]:
    """Return (Dmax_mm, endpoint_A, endpoint_B) — max diameter across surface voxels."""
    surface = _surface_voxels(tumor_mask)
    coords = np.argwhere(surface)
    if len(coords) < 2:
        coords = np.argwhere(tumor_mask)
    if len(coords) < 2:
        return 0.0, (0, 0, 0), (0, 0, 0)

    world = coords * spacing

    # Fast approximate: start from arbitrary point, find farthest, repeat once
    def _farthest(world_pts, ref_idx):
        dists = np.sum((world_pts - world_pts[ref_idx]) ** 2, axis=1)
        return int(np.argmax(dists))

    a_idx = _farthest(world, 0)
    b_idx = _farthest(world, a_idx)
    a_idx = _farthest(world, b_idx)

    dmax = float(np.sqrt(np.sum((world[a_idx] - world[b_idx]) ** 2)))
    pt_a = tuple(int(x) for x in coords[a_idx])
    pt_b = tuple(int(x) for x in coords[b_idx])
    return dmax, pt_a, pt_b  # type: ignore[return-value]


def compute_dmax(
    tumor_mask: np.ndarray,
    spacing: np.ndarray,
) -> float:
    """Return Dmax in mm."""
    dmax, _, _ = compute_dmax_with_endpoints(tumor_mask, spacing)
    return dmax


# ---------------------------------------------------------------------------
# NHOCmax — Normalised Hotspot-to-Centroid distance
# ---------------------------------------------------------------------------

def compute_nhoc_max_with_coords(
    pet_data: np.ndarray,
    tumor_mask: np.ndarray,
    spacing: np.ndarray,
) -> Tuple[float, Tuple[int, int, int], Tuple[float, float, float]]:
    """Return (NHOCmax, hotspot_voxel, centroid_voxel)."""
    pet_in = pet_data * tumor_mask
    if not tumor_mask.any() or pet_in.max() == 0:
        return 0.0, (0, 0, 0), (0.0, 0.0, 0.0)

    hotspot = np.unravel_index(pet_in.argmax(), pet_in.shape)
    coords = np.argwhere(tumor_mask).astype(float)
    centroid = coords.mean(axis=0)

    dmax = compute_dmax(tumor_mask, spacing)
    if dmax == 0:
        return 0.0, hotspot, tuple(centroid)  # type: ignore[return-value]

    dist_mm = float(np.sqrt(np.sum(
        ((np.array(hotspot) - centroid) * spacing) ** 2
    )))
    nhoc = dist_mm / dmax
    return nhoc, hotspot, tuple(centroid)  # type: ignore[return-value]


def compute_nhoc_max(
    pet_data: np.ndarray,
    tumor_mask: np.ndarray,
    spacing: np.ndarray,
) -> float:
    """Return NHOCmax."""
    val, _, _ = compute_nhoc_max_with_coords(pet_data, tumor_mask, spacing)
    return val


# ---------------------------------------------------------------------------
# NHOPmax — Normalised Hotspot-to-Periphery distance
# ---------------------------------------------------------------------------

def compute_nhop_max_with_coords(
    pet_data: np.ndarray,
    tumor_mask: np.ndarray,
    spacing: np.ndarray,
) -> Tuple[float, Tuple[int, int, int], Tuple[int, int, int]]:
    """Return (NHOPmax, hotspot_voxel, boundary_voxel)."""
    pet_in = pet_data * tumor_mask
    if not tumor_mask.any() or pet_in.max() == 0:
        return 0.0, (0, 0, 0), (0, 0, 0)

    hotspot = np.unravel_index(pet_in.argmax(), pet_in.shape)
    surface = _surface_voxels(tumor_mask)
    surf_coords = np.argwhere(surface)
    if surf_coords.size == 0:
        surf_coords = np.argwhere(tumor_mask)

    dists = np.sum(
        ((surf_coords - np.array(hotspot)) * spacing) ** 2, axis=1
    )
    farthest_idx = int(np.argmax(dists))
    boundary_pt = tuple(int(x) for x in surf_coords[farthest_idx])

    dmax = compute_dmax(tumor_mask, spacing)
    if dmax == 0:
        return 0.0, hotspot, boundary_pt  # type: ignore[return-value]

    dist_mm = float(np.sqrt(dists[farthest_idx]))
    nhop = dist_mm / dmax
    return nhop, hotspot, boundary_pt  # type: ignore[return-value]


def compute_nhop_max(
    pet_data: np.ndarray,
    tumor_mask: np.ndarray,
    spacing: np.ndarray,
) -> float:
    """Return NHOPmax."""
    val, _, _ = compute_nhop_max_with_coords(pet_data, tumor_mask, spacing)
    return val


# ---------------------------------------------------------------------------
# gETU — Generalised Effective Tumor Uptake
# ---------------------------------------------------------------------------

def compute_getu(
    pet_data: np.ndarray,
    tumor_mask: np.ndarray,
    spacing: np.ndarray,
    threshold_fraction: float = 0.41,
) -> Dict[str, float]:
    """Compute gETU (generalised Effective Tumor Uptake).

    gETU = SUVmean(MTV) × MTV_cm3

    Returns dict with keys: gETU, MTV_cm3, SUVmean_MTV, SUVmax, threshold_used.
    """
    pet_in = pet_data * tumor_mask
    if not tumor_mask.any() or pet_in.max() == 0:
        return {"gETU": 0.0, "MTV_cm3": 0.0, "SUVmean_MTV": 0.0,
                "SUVmax": 0.0, "threshold_used": 0.0}

    suv_max = float(pet_in.max())
    threshold = threshold_fraction * suv_max
    mtv_mask = tumor_mask & (pet_data >= threshold)

    voxel_vol_mm3 = float(np.prod(spacing))
    mtv_cm3 = float(mtv_mask.sum()) * voxel_vol_mm3 / 1000.0

    if not mtv_mask.any():
        return {"gETU": 0.0, "MTV_cm3": 0.0, "SUVmean_MTV": 0.0,
                "SUVmax": suv_max, "threshold_used": threshold}

    suv_mean_mtv = float(pet_data[mtv_mask].mean())
    getu = suv_mean_mtv * mtv_cm3

    return {
        "gETU": getu,
        "MTV_cm3": mtv_cm3,
        "SUVmean_MTV": suv_mean_mtv,
        "SUVmax": suv_max,
        "threshold_used": threshold,
    }


# ---------------------------------------------------------------------------
# Sphericity / Asphericity
# ---------------------------------------------------------------------------

def compute_sphericity_asphericity(
    tumor_mask: np.ndarray,
    spacing: np.ndarray,
) -> Tuple[float, float]:
    """Compute sphericity and asphericity using marching cubes surface mesh.

    Returns (sphericity, asphericity).
    """
    try:
        from skimage.measure import marching_cubes, mesh_surface_area
    except ImportError:
        logger.warning("scikit-image not available; returning approximate sphericity")
        # Fallback: isoperimetric approximation
        vol_mm3 = float(tumor_mask.sum()) * float(np.prod(spacing))
        r = (3.0 * vol_mm3 / (4.0 * np.pi)) ** (1.0 / 3.0)
        surface_mm2 = 4.0 * np.pi * r ** 2
        sphericity = (np.pi ** (1.0 / 3.0) * (6.0 * vol_mm3) ** (2.0 / 3.0)) / surface_mm2
        return float(np.clip(sphericity, 0.0, 1.0)), float(1.0 - np.clip(sphericity, 0.0, 1.0))

    if not tumor_mask.any():
        return 0.0, 1.0

    # Pad to avoid boundary issues
    padded = np.pad(tumor_mask.astype(np.float32), 1, mode="constant")
    try:
        verts, faces, normals, values = marching_cubes(padded, level=0.5,
                                                        spacing=tuple(spacing))
        surface_mm2 = float(mesh_surface_area(verts, faces))
    except Exception as exc:
        logger.debug("marching_cubes failed: %s", exc)
        return 0.0, 1.0

    vol_mm3 = float(tumor_mask.sum()) * float(np.prod(spacing))
    if surface_mm2 == 0:
        return 0.0, 1.0

    sphericity = (np.pi ** (1.0 / 3.0) * (6.0 * vol_mm3) ** (2.0 / 3.0)) / surface_mm2
    sphericity = float(np.clip(sphericity, 0.0, 1.0))
    asphericity = 1.0 - sphericity
    return sphericity, asphericity


# ---------------------------------------------------------------------------
# Zero-value fallback
# ---------------------------------------------------------------------------

def _zero_metrics() -> Dict[str, float]:
    """Return a dict of zero-value metrics for error fallback."""
    return {
        "SUVmax": 0.0,
        "SUVmean": 0.0,
        "SUVpeak": 0.0,
        "TLG": 0.0,
        "MTV_cm3": 0.0,
        "volume_cm3": 0.0,
        "Dmax_mm": 0.0,
        "NHOCmax": 0.0,
        "NHOPmax": 0.0,
        "gETU": 0.0,
        "gETU_MTV_cm3": 0.0,
        "gETU_SUVmean": 0.0,
        "sphericity": 0.0,
        "asphericity": 1.0,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_all_metrics(
    pet_nifti_path: str,
    tumor_mask_path: str,
    ct_nifti_path: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict]:
    """Compute all radiomics metrics for a lung tumor.

    Parameters
    ----------
    pet_nifti_path : str
        Path to the PET NIfTI file (SUV-normalised).
    tumor_mask_path : str
        Path to the binary tumor mask NIfTI file.
    ct_nifti_path : str, optional
        Path to the CT NIfTI file (HU). Required for CT-based strategies.

    Returns
    -------
    metrics : dict
        All computed metrics.
    qc_coords : dict
        QC coordinates for overlay generation, with keys:
        dmax_A, dmax_B, nhoc_hotspot, nhoc_centroid,
        nhop_hotspot, nhop_boundary, getu_centroid.
    """
    try:
        pet_img = nib.load(pet_nifti_path)
        mask_img = nib.load(tumor_mask_path)
    except Exception as exc:
        logger.error("Failed to load NIfTI files: %s", exc)
        return _zero_metrics(), {}

    pet_data = np.asarray(pet_img.get_fdata(), dtype=np.float32)
    tumor_mask = np.asarray(mask_img.get_fdata(), dtype=bool)
    spacing = _voxel_spacing_from_affine(pet_img.affine)

    # Optionally load CT
    ct_data = None
    if ct_nifti_path is not None:
        try:
            ct_img = nib.load(ct_nifti_path)
            ct_data_raw = np.asarray(ct_img.get_fdata(), dtype=np.float32)
            # Resample CT to PET space if shapes differ
            if ct_data_raw.shape != pet_data.shape:
                logger.warning("CT shape %s ≠ PET shape %s; using PET spacing",
                               ct_data_raw.shape, pet_data.shape)
                # Simple nearest-neighbour crop/pad
                ct_data = np.full(pet_data.shape, -1000.0, dtype=np.float32)
                slices = tuple(slice(0, min(s, t)) for s, t in
                               zip(ct_data_raw.shape, pet_data.shape))
                ct_data[slices] = ct_data_raw[slices]
            else:
                ct_data = ct_data_raw
        except Exception as exc:
            logger.warning("Could not load CT: %s", exc)
            ct_data = None

    if ct_data is None:
        ct_data = np.zeros_like(pet_data)

    if not tumor_mask.any():
        logger.warning("Tumor mask is empty; returning zero metrics")
        return _zero_metrics(), {}

    # Resample mask to PET space if needed
    if tumor_mask.shape != pet_data.shape:
        from scipy.ndimage import zoom
        zoom_factors = tuple(p / m for p, m in zip(pet_data.shape, tumor_mask.shape))
        tumor_mask = zoom(tumor_mask.astype(float), zoom_factors, order=0) > 0.5

    # Isolate primary tumor
    primary = _isolate_primary_tumor(tumor_mask, pet_data, ct_data, spacing)

    voxel_vol_mm3 = float(np.prod(spacing))

    # Basic PET metrics
    pet_in_primary = pet_data * primary
    suv_max = float(pet_in_primary.max()) if primary.any() else 0.0
    suv_mean = float(pet_data[primary].mean()) if primary.any() else 0.0
    suv_peak = compute_suv_peak(pet_data, primary, spacing)

    # Volume
    volume_cm3 = float(primary.sum()) * voxel_vol_mm3 / 1000.0

    # TLG = SUVmean × MTV (with 41% threshold)
    getu_dict = compute_getu(pet_data, primary, spacing)
    mtv_cm3 = getu_dict["MTV_cm3"]
    tlg = getu_dict["SUVmean_MTV"] * mtv_cm3

    # Dmax
    dmax, pt_a, pt_b = compute_dmax_with_endpoints(primary, spacing)

    # NHOCmax
    nhoc, hotspot_nhoc, centroid = compute_nhoc_max_with_coords(
        pet_data, primary, spacing
    )

    # NHOPmax
    nhop, hotspot_nhop, boundary_pt = compute_nhop_max_with_coords(
        pet_data, primary, spacing
    )

    # Sphericity
    sphericity, asphericity = compute_sphericity_asphericity(primary, spacing)

    # gETU centroid (same as tumor centroid)
    coords = np.argwhere(primary).astype(float)
    centroid_arr = coords.mean(axis=0) if coords.size > 0 else np.zeros(3)

    metrics: Dict[str, float] = {
        "SUVmax": suv_max,
        "SUVmean": suv_mean,
        "SUVpeak": suv_peak,
        "TLG": tlg,
        "MTV_cm3": mtv_cm3,
        "volume_cm3": volume_cm3,
        "Dmax_mm": dmax,
        "NHOCmax": nhoc,
        "NHOPmax": nhop,
        "gETU": getu_dict["gETU"],
        "gETU_MTV_cm3": getu_dict["MTV_cm3"],
        "gETU_SUVmean": getu_dict["SUVmean_MTV"],
        "sphericity": sphericity,
        "asphericity": asphericity,
    }

    qc_coords: Dict = {
        "dmax_A": pt_a,
        "dmax_B": pt_b,
        "nhoc_hotspot": hotspot_nhoc,
        "nhoc_centroid": centroid,
        "nhop_hotspot": hotspot_nhop,
        "nhop_boundary": boundary_pt,
        "getu_centroid": tuple(centroid_arr),
        "getu_threshold": getu_dict["threshold_used"],
        "getu_suv_max": suv_max,
    }

    logger.info(
        "Metrics computed — SUVmax=%.2f  Dmax=%.1f mm  gETU=%.2f  MTV=%.2f cm³",
        suv_max, dmax, getu_dict["gETU"], mtv_cm3,
    )
    return metrics, qc_coords
