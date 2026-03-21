"""Mask_QC.py  v5.9
QC overlay generation for lung tumor segmentation.

Generates:
  • Mask_QC.png          — 3×3 PET/CT/Fusion × Axial/Coronal/Sagittal grid
  • Dmax_QC.png          — Maximum intra-tumoral diameter QC
  • NHOCmax_QC.png       — Normalised Hotspot-to-Centroid QC
  • NHOPmax_QC.png       — Normalised Hotspot-to-Periphery QC
  • gETU_QC.png          — Generalised Effective Tumor Uptake QC
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from scipy.ndimage import (
    binary_erosion,
    binary_dilation,
    distance_transform_edt,
    label as ndi_label,
    zoom,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _voxel_spacing_from_affine(affine: np.ndarray) -> np.ndarray:
    """Extract voxel spacing (mm) from a NIfTI affine matrix."""
    return np.sqrt((affine[:3, :3] ** 2).sum(axis=0))


def _get_aspect_ratio(
    spacing: np.ndarray, view: str
) -> float:
    """Compute display aspect ratio for the given view.

    Parameters
    ----------
    spacing : array-like, shape (3,)
        Voxel spacing in (i, j, k) order.
    view : str
        One of 'axial', 'coronal', 'sagittal'.
    """
    si, sj, sk = float(spacing[0]), float(spacing[1]), float(spacing[2])
    if view == "axial":
        # rows = j-axis, cols = i-axis → aspect = sj/si
        return sj / si
    elif view == "coronal":
        # rows = k-axis, cols = i-axis → aspect = sk/si
        return sk / si
    else:  # sagittal
        # rows = k-axis, cols = j-axis → aspect = sk/sj
        return sk / sj


def _orient_for_display(slice_2d: np.ndarray, view: str) -> np.ndarray:
    """Rotate/flip a 2-D slice for anatomical display orientation."""
    if view == "axial":
        return np.rot90(slice_2d, k=1)
    elif view == "coronal":
        return np.rot90(slice_2d, k=1)
    else:  # sagittal
        return np.rot90(slice_2d, k=1)


def _add_contour(
    ax: plt.Axes,
    slice_2d: np.ndarray,
    color: str,
    lw: float = 1.5,
    ls: str = "-",
) -> None:
    """Draw contour lines on a matplotlib axes from a binary 2-D slice."""
    if slice_2d.any():
        ax.contour(slice_2d.astype(float), levels=[0.5], colors=[color],
                   linewidths=[lw], linestyles=[ls])


# ---------------------------------------------------------------------------
# 3×3 Grid plot helpers
# ---------------------------------------------------------------------------

def _plot_pet(
    ax: plt.Axes,
    ps: np.ndarray,
    ms: np.ndarray,
    cs: np.ndarray,
    title: str,
    vmax: float,
    view: str,
    aspect: float,
) -> None:
    """PET overlay with tumor and constraint contours."""
    pet_disp = _orient_for_display(ps, view)
    ax.imshow(pet_disp, cmap="hot", vmin=0, vmax=vmax, aspect=aspect, origin="lower")
    _add_contour(ax, _orient_for_display(ms, view), "cyan", lw=1.5)
    _add_contour(ax, _orient_for_display(cs, view), "yellow", lw=1.0, ls="--")
    ax.set_title(title, fontsize=8, color="white")
    ax.axis("off")


def _plot_ct(
    ax: plt.Axes,
    cts: np.ndarray,
    ms: np.ndarray,
    cs: np.ndarray,
    title: str,
    view: str,
    aspect: float,
) -> None:
    """CT with mask overlay and contours."""
    ct_disp = _orient_for_display(cts, view)
    ax.imshow(ct_disp, cmap="gray", vmin=-200, vmax=300, aspect=aspect, origin="lower")
    # Soft overlay for mask
    mask_disp = _orient_for_display(ms.astype(float), view)
    ax.imshow(np.ma.masked_where(mask_disp < 0.5, mask_disp),
              cmap="Blues", alpha=0.35, vmin=0, vmax=1,
              aspect=aspect, origin="lower")
    _add_contour(ax, _orient_for_display(ms, view), "cyan", lw=1.5)
    _add_contour(ax, _orient_for_display(cs, view), "yellow", lw=1.0, ls="--")
    ax.set_title(title, fontsize=8, color="white")
    ax.axis("off")


def _plot_fusion(
    ax: plt.Axes,
    cts: np.ndarray,
    ps: np.ndarray,
    ms: np.ndarray,
    cs: np.ndarray,
    title: str,
    vmax: float,
    view: str,
    aspect: float,
) -> None:
    """CT+PET fusion with contours."""
    ct_disp = _orient_for_display(cts, view)
    pet_disp = _orient_for_display(ps, view)
    ax.imshow(ct_disp, cmap="gray", vmin=-200, vmax=300, aspect=aspect, origin="lower")
    ax.imshow(np.ma.masked_where(pet_disp <= 0, pet_disp),
              cmap="hot", alpha=0.55, vmin=0, vmax=vmax,
              aspect=aspect, origin="lower")
    _add_contour(ax, _orient_for_display(ms, view), "cyan", lw=1.5)
    _add_contour(ax, _orient_for_display(cs, view), "yellow", lw=1.0, ls="--")
    ax.set_title(title, fontsize=8, color="white")
    ax.axis("off")


# ---------------------------------------------------------------------------
# Public: 3×3 QC Grid (Mask_QC.png)
# ---------------------------------------------------------------------------

def generate_qc_overlays(
    pet_nifti_path: str,
    ct_nifti_path: str,
    tumor_mask_path: str,
    constraint_mask_path: str,
    output_path: str,
    dpi: int = 300,
) -> None:
    """Generate a 3×3 QC grid (PET/CT/Fusion × Axial/Coronal/Sagittal).

    The slices are centred on the tumor centroid.  Output is saved to
    *output_path* (e.g., ``<out_dir>/Mask_QC.png``).
    """
    try:
        pet_img = nib.load(pet_nifti_path)
        ct_img = nib.load(ct_nifti_path)
        mask_img = nib.load(tumor_mask_path)
        constraint_img = nib.load(constraint_mask_path)
    except Exception as exc:
        logger.error("generate_qc_overlays: failed to load NIfTI — %s", exc)
        return

    pet = np.asarray(pet_img.get_fdata(), dtype=np.float32)
    ct = np.asarray(ct_img.get_fdata(), dtype=np.float32)
    tumor = np.asarray(mask_img.get_fdata(), dtype=bool)
    constraint = np.asarray(constraint_img.get_fdata(), dtype=bool)
    spacing = _voxel_spacing_from_affine(pet_img.affine)

    # Resample CT/mask/constraint to PET space if shapes differ
    def _resamp(vol: np.ndarray, target_shape: tuple, order: int = 1) -> np.ndarray:
        if vol.shape == target_shape:
            return vol
        factors = tuple(t / s for t, s in zip(target_shape, vol.shape))
        return zoom(vol, factors, order=order)

    tshape = pet.shape
    ct = _resamp(ct, tshape, order=1)
    tumor = _resamp(tumor.astype(np.float32), tshape, order=0) > 0.5
    constraint = _resamp(constraint.astype(np.float32), tshape, order=0) > 0.5

    if not tumor.any():
        logger.warning("generate_qc_overlays: tumor mask is empty")
        return

    # Tumor centroid
    coords = np.argwhere(tumor)
    ci, cj, ck = coords.mean(axis=0).astype(int)
    pet_vmax = float(np.percentile(pet[tumor], 99)) if tumor.any() else 1.0

    views = ["axial", "coronal", "sagittal"]
    slice_indices = {"axial": ck, "coronal": cj, "sagittal": ci}

    def _get_slice(vol: np.ndarray, view: str) -> np.ndarray:
        idx = slice_indices[view]
        if view == "axial":
            return vol[:, :, idx]
        elif view == "coronal":
            return vol[:, idx, :]
        else:
            return vol[idx, :, :]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12), facecolor="black")
    fig.suptitle("Lung Tumor QC — PET / CT / Fusion", color="white", fontsize=12)

    row_labels = ["PET", "CT", "Fusion"]
    col_labels = ["Axial", "Coronal", "Sagittal"]

    for col_idx, view in enumerate(views):
        aspect = _get_aspect_ratio(spacing, view)
        ps = _get_slice(pet, view)
        cts = _get_slice(ct, view)
        ms = _get_slice(tumor.astype(float), view)
        cs = _get_slice(constraint.astype(float), view)

        _plot_pet(axes[0, col_idx], ps, ms, cs,
                  f"PET — {col_labels[col_idx]}", pet_vmax, view, aspect)
        _plot_ct(axes[1, col_idx], cts, ms, cs,
                 f"CT — {col_labels[col_idx]}", view, aspect)
        _plot_fusion(axes[2, col_idx], cts, ps, ms, cs,
                     f"Fusion — {col_labels[col_idx]}", pet_vmax, view, aspect)

    for ax in axes.ravel():
        ax.set_facecolor("black")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    try:
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="black")
        logger.info("Saved Mask_QC → %s", output_path)
    except Exception as exc:
        logger.error("Failed to save Mask_QC: %s", exc)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Primary isolation helpers (for metric QC)
# ---------------------------------------------------------------------------

def _bottleneck_qc(
    clean: np.ndarray,
    pet_data: np.ndarray,
    ct_data: np.ndarray,
    spacing: np.ndarray,
    smi: int,
    smv: float,
    os: int,
) -> Optional[np.ndarray]:
    """S0 — Distance-transform bottleneck detection."""
    dt = distance_transform_edt(clean, sampling=spacing)
    threshold = 0.35 * dt.max()
    candidate = clean & (dt >= threshold)
    labeled, n = ndi_label(candidate)
    if n < 2:
        return None
    sizes = ndimage.sum(candidate, labeled, range(1, n + 1))
    kept = np.zeros_like(clean)
    for idx, sz in enumerate(sizes, 1):
        if sz >= smi:
            kept |= labeled == idx
    if not kept.any():
        return None
    from scipy.ndimage import sum as ndi_sum
    labeled2, n2 = ndi_label(kept)
    if n2 == 0:
        return None
    best_idx = int(np.argmax(
        [float(pet_data[labeled2 == i].mean()) for i in range(1, n2 + 1)]
    )) + 1
    return (labeled2 == best_idx).astype(bool)


def _convex_hull_trim(
    clean: np.ndarray,
    ct_data: np.ndarray,
    pet_data: np.ndarray,
    spacing: np.ndarray,
    smi: int,
    smv: float,
    os: int,
) -> Optional[np.ndarray]:
    """S1 — Convex hull based trimming."""
    try:
        from scipy.spatial import Delaunay
    except ImportError:
        return None

    coords = np.argwhere(clean)
    if len(coords) < 4:
        return None
    try:
        tri = Delaunay(coords * spacing)
    except Exception:
        return None

    labeled, n = ndi_label(clean)
    if n == 0:
        return None

    best_comp = None
    best_pet = -1.0
    for idx in range(1, n + 1):
        comp = labeled == idx
        if comp.sum() < smi:
            continue
        mean_pet = float(pet_data[comp].mean())
        if mean_pet > best_pet:
            best_pet = mean_pet
            best_comp = comp

    return best_comp


def _lung_adjacency_filter(
    clean: np.ndarray,
    ct_data: np.ndarray,
    pet_data: np.ndarray,
    spacing: np.ndarray,
    smi: int,
    os: int,
    pm: float,
) -> Optional[np.ndarray]:
    """S2 — Lung air adjacency filtering."""
    air_mask = ct_data < -400
    air_dt = distance_transform_edt(~air_mask, sampling=spacing)
    labeled, n = ndi_label(clean)
    if n == 0:
        return None

    best_comp = None
    best_score = -1.0
    for idx in range(1, n + 1):
        comp = labeled == idx
        if comp.sum() < smi:
            continue
        score = float(pet_data[comp].mean()) / (1.0 + float(air_dt[comp].mean()) * 0.1)
        if score > best_score:
            best_score = score
            best_comp = comp
    return best_comp


def _erosion_isolation(
    clean: np.ndarray,
    spacing: np.ndarray,
    smi: int,
    os: int,
) -> Optional[np.ndarray]:
    """S3 — Progressive erosion separation."""
    eroded = clean.copy()
    for iters in range(1, 6):
        eroded = binary_erosion(eroded, iterations=1)
        labeled, n = ndi_label(eroded)
        if n >= 2:
            sizes = ndimage.sum(eroded, labeled, range(1, n + 1))
            largest_idx = int(np.argmax(sizes)) + 1
            core = labeled == largest_idx
            grown = binary_dilation(core, iterations=iters)
            result = clean & grown
            if result.sum() >= smi:
                return result.astype(bool)
    return None


def _pet_distance_isolation(
    clean: np.ndarray,
    pet_data: np.ndarray,
    spacing: np.ndarray,
    smi: int,
    smv: float,
    os: int,
) -> Optional[np.ndarray]:
    """S4 — PET-weighted distance isolation."""
    pet_in = pet_data * clean
    if pet_in.max() == 0:
        return None
    hotspot = np.unravel_index(pet_in.argmax(), pet_in.shape)
    score = pet_in / (1.0 + np.sqrt(
        sum(((np.indices(clean.shape)[k] - hotspot[k]) * spacing[k]) ** 2
            for k in range(3))
    ) * 0.05)
    threshold = score[clean].mean()
    selected = clean & (score >= threshold)
    labeled, n = ndi_label(selected)
    if n == 0:
        return None
    sizes = ndimage.sum(selected, labeled, range(1, n + 1))
    largest_idx = int(np.argmax(sizes)) + 1
    result = (labeled == largest_idx).astype(bool)
    return result if result.sum() >= smi else None


def _ct_density_filter(
    clean: np.ndarray,
    ct_data: np.ndarray,
    spacing: np.ndarray,
    smi: int,
    os: int,
) -> Optional[np.ndarray]:
    """S5 — CT density based filtering."""
    solid = ct_data > -100
    candidate = clean & solid
    labeled, n = ndi_label(candidate)
    if n == 0:
        return None
    sizes = ndimage.sum(candidate, labeled, range(1, n + 1))
    largest_idx = int(np.argmax(sizes)) + 1
    result = (labeled == largest_idx).astype(bool)
    return result if result.sum() >= smi else None


def _isolate_primary_inner(
    tumor_mask: np.ndarray,
    pet_data: np.ndarray,
    ct_data: np.ndarray,
    spacing: np.ndarray,
) -> np.ndarray:
    """Run 6 isolation strategies (S0-S5) returning the primary mask."""
    from scipy.ndimage import sum as ndi_sum

    voxel_vol = float(np.prod(spacing))
    n_vox = int(tumor_mask.sum())
    min_mm3 = 50.0
    smi = max(10, int(min_mm3 / voxel_vol))
    smv = min_mm3
    os = n_vox
    pm = float(pet_data[tumor_mask].max()) if tumor_mask.any() else 1.0

    clean = binary_erosion(tumor_mask, iterations=1)
    clean = binary_dilation(clean, iterations=1).astype(bool)
    if not clean.any():
        clean = tumor_mask.astype(bool)

    labeled, n = ndi_label(clean)
    if n == 1:
        return clean

    strategies = [
        lambda: _bottleneck_qc(clean, pet_data, ct_data, spacing, smi, smv, os),
        lambda: _convex_hull_trim(clean, ct_data, pet_data, spacing, smi, smv, os),
        lambda: _lung_adjacency_filter(clean, ct_data, pet_data, spacing, smi, os, pm),
        lambda: _erosion_isolation(clean, spacing, smi, os),
        lambda: _pet_distance_isolation(clean, pet_data, spacing, smi, smv, os),
        lambda: _ct_density_filter(clean, ct_data, spacing, smi, os),
    ]

    for i, strategy in enumerate(strategies):
        try:
            result = strategy()
            if result is not None and result.sum() >= smi:
                logger.debug("Mask_QC isolation S%d succeeded", i)
                return result.astype(bool)
        except Exception as exc:
            logger.debug("Mask_QC isolation S%d failed: %s", i, exc)

    # Fallback: highest-PET component
    labeled, n = ndi_label(clean)
    best_comp = None
    best_pet = -1.0
    for idx in range(1, n + 1):
        comp = labeled == idx
        if comp.sum() < smi:
            continue
        mp = float(pet_data[comp].mean())
        if mp > best_pet:
            best_pet = mp
            best_comp = comp
    if best_comp is not None:
        return best_comp.astype(bool)
    from scipy.ndimage import label as ndi_label2
    labeled2, _ = ndi_label2(clean)
    sizes = ndimage.sum(clean, labeled2, range(1, labeled2.max() + 1))
    if not sizes:
        return clean.astype(bool)
    best = int(np.argmax(sizes)) + 1
    return (labeled2 == best).astype(bool)


def _isolate_primary_for_qc(
    tumor_mask: np.ndarray,
    pet_data: np.ndarray,
    ct_data: np.ndarray,
    spacing: np.ndarray,
) -> np.ndarray:
    """Wrapper with morphological cleanup before isolation."""
    if not tumor_mask.any():
        return tumor_mask.astype(bool)
    try:
        return _isolate_primary_inner(tumor_mask, pet_data, ct_data, spacing)
    except Exception as exc:
        logger.warning("_isolate_primary_for_qc failed: %s; returning largest CC", exc)
        labeled, n = ndi_label(tumor_mask)
        if n == 0:
            return tumor_mask.astype(bool)
        sizes = ndimage.sum(tumor_mask, labeled, range(1, n + 1))
        best = int(np.argmax(sizes)) + 1
        return (labeled == best).astype(bool)


# ---------------------------------------------------------------------------
# Coordinate transforms & drawing
# ---------------------------------------------------------------------------

def _transform_point(
    pt: Tuple[int, int, int],
    view: str,
    vs: np.ndarray,
) -> Tuple[float, float]:
    """Transform a 3-D voxel point to 2-D display coordinates.

    Coordinate convention (matching _orient_for_display rot90 k=1):
    - axial   : ro = pi, co = pj, nc = vs[1];  flip nr = (nc-1)-co
    - coronal : ro = pi, co = pk, nc = vs[2]
    - sagittal: ro = pj, co = pk, nc = vs[2]

    Returns (col_display, row_display) as (float, float).
    """
    pi, pj, pk = int(pt[0]), int(pt[1]), int(pt[2])

    if view == "axial":
        ro, co = pi, pj
        nc = int(vs[1])
        nr = (nc - 1) - co
        nr = (nc - 1) - nr  # axial flip
        ncl = ro
    elif view == "coronal":
        ro, co = pi, pk
        nc = int(vs[2])
        nr = (nc - 1) - co
        ncl = ro
    else:  # sagittal
        ro, co = pj, pk
        nc = int(vs[2])
        nr = (nc - 1) - co
        ncl = ro

    return float(ncl), float(nr)


def _slice_for_point(pt: Tuple[int, int, int], view: str) -> int:
    """Return the slice index for a given view."""
    pi, pj, pk = int(pt[0]), int(pt[1]), int(pt[2])
    if view == "axial":
        return pk
    elif view == "coronal":
        return pj
    else:  # sagittal
        return pi


def _clamp(val: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    return float(max(lo, min(hi, val)))


def _get_slice(vol: np.ndarray, view: str, idx: int) -> np.ndarray:
    """Extract a 2-D slice from a 3-D volume."""
    idx = int(idx)
    if view == "axial":
        idx = int(_clamp(idx, 0, vol.shape[2] - 1))
        return vol[:, :, idx]
    elif view == "coronal":
        idx = int(_clamp(idx, 0, vol.shape[1] - 1))
        return vol[:, idx, :]
    else:  # sagittal
        idx = int(_clamp(idx, 0, vol.shape[0] - 1))
        return vol[idx, :, :]


def _tumor_bbox_2d(
    ts: np.ndarray, margin: int = 30
) -> Tuple[int, int, int, int]:
    """Compute 2-D bounding box around non-zero region with margin.

    Returns (row_min, row_max, col_min, col_max).
    """
    rows = np.any(ts, axis=1)
    cols = np.any(ts, axis=0)
    if not rows.any():
        h, w = ts.shape
        return 0, h, 0, w
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    h, w = ts.shape
    r0 = int(_clamp(r0 - margin, 0, h - 1))
    r1 = int(_clamp(r1 + margin, 0, h - 1))
    c0 = int(_clamp(c0 - margin, 0, w - 1))
    c1 = int(_clamp(c1 + margin, 0, w - 1))
    return r0, r1, c0, c1


def _transform_cropped(
    pt: Tuple[int, int, int],
    view: str,
    vs: np.ndarray,
    crop: Tuple[int, int, int, int],
) -> Tuple[float, float]:
    """Transform point then adjust for crop offset (r0, r1, c0, c1)."""
    x, y = _transform_point(pt, view, vs)
    r0, r1, c0, c1 = crop
    return float(x - c0), float(y - r0)


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def _plot_base(
    ax: plt.Axes,
    cs: np.ndarray,
    ps: np.ndarray,
    ms: np.ndarray,
    view: str,
    aspect: float,
    pvmax: float,
    crop: Optional[Tuple[int, int, int, int]] = None,
) -> None:
    """Base plot: CT gray + PET hot overlay + contour."""
    def _maybe_crop(img2d):
        if crop is None:
            return img2d
        r0, r1, c0, c1 = crop
        return img2d[r0:r1 + 1, c0:c1 + 1]

    ct_disp = _orient_for_display(_maybe_crop(cs), view)
    pet_disp = _orient_for_display(_maybe_crop(ps), view)
    mask_disp = _orient_for_display(_maybe_crop(ms), view)

    ax.imshow(ct_disp, cmap="gray", vmin=-200, vmax=300, aspect=aspect, origin="lower")
    ax.imshow(np.ma.masked_where(pet_disp <= 0, pet_disp),
              cmap="hot", alpha=0.55, vmin=0, vmax=pvmax,
              aspect=aspect, origin="lower")
    if mask_disp.any():
        ax.contour(mask_disp.astype(float), levels=[0.5], colors=["cyan"],
                   linewidths=[1.0])
    ax.set_facecolor("black")
    ax.axis("off")


def _draw_marker(
    ax: plt.Axes,
    xy: Tuple[float, float],
    color: str,
    marker: str = "o",
    size: float = 60.0,
    edge: str = "white",
    ew: float = 0.8,
    zorder: int = 10,
) -> None:
    """Draw a scatter marker on axes."""
    ax.scatter([xy[0]], [xy[1]], c=color, marker=marker, s=size,
               edgecolors=edge, linewidths=ew, zorder=zorder)


def _draw_line(
    ax: plt.Axes,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    color: str,
    lw: float = 1.5,
    ls: str = "-",
    zorder: int = 9,
) -> None:
    """Draw a line between two 2-D points on axes."""
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=lw,
            ls=ls, zorder=zorder)


# ---------------------------------------------------------------------------
# 4-panel figure builder
# ---------------------------------------------------------------------------

def _make_4panel(
    pet: np.ndarray,
    ct: np.ndarray,
    dm: np.ndarray,
    spacing: np.ndarray,
    pvmax: float,
    pp: Tuple[int, int, int],
    views: Sequence[str],
    zv: str,
    zm: np.ndarray,
) -> Tuple[plt.Figure, np.ndarray]:
    """Create a 1×4 subplot (axial, coronal, sagittal, zoomed) with black bg.

    Parameters
    ----------
    pet, ct, dm : 3-D arrays
        PET data, CT data, and display mask (primary tumor).
    spacing : array-like
        Voxel spacing.
    pvmax : float
        PET display max.
    pp : 3-tuple
        Reference point (i, j, k) for slice selection.
    views : sequence of str
        The three view names for panels 0-2.
    zv : str
        View for the zoomed panel (panel 3).
    zm : ndarray
        Mask used for zoom bounding box.

    Returns
    -------
    fig, axes : Figure and 1-D axes array of length 4.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), facecolor="black")
    fig.patch.set_facecolor("black")

    vs = np.array(pet.shape, dtype=float)

    for panel_idx, view in enumerate(views[:3]):
        ax = axes[panel_idx]
        s_idx = _slice_for_point(pp, view)
        aspect = _get_aspect_ratio(spacing, view)
        cs_slice = _get_slice(ct, view, s_idx)
        ps_slice = _get_slice(pet, view, s_idx)
        ms_slice = _get_slice(dm.astype(float), view, s_idx)
        _plot_base(ax, cs_slice, ps_slice, ms_slice, view, aspect, pvmax)
        ax.set_title(view.capitalize(), fontsize=8, color="white")

    # Zoomed panel
    ax_zoom = axes[3]
    z_idx = _slice_for_point(pp, zv)
    aspect_z = _get_aspect_ratio(spacing, zv)
    cs_z = _get_slice(ct, zv, z_idx)
    ps_z = _get_slice(pet, zv, z_idx)
    ms_z = _get_slice(dm.astype(float), zv, z_idx)
    ms_z_disp = _orient_for_display(ms_z, zv)
    r0, r1, c0, c1 = _tumor_bbox_2d(ms_z_disp > 0.5)
    crop = (r0, r1, c0, c1)
    _plot_base(ax_zoom, cs_z, ps_z, ms_z, zv, aspect_z, pvmax, crop=crop)
    ax_zoom.set_title(f"Zoom ({zv.capitalize()})", fontsize=8, color="white")

    for ax in axes:
        ax.set_facecolor("black")

    return fig, axes


# ---------------------------------------------------------------------------
# Point validation & boundary
# ---------------------------------------------------------------------------

def _validate_point(
    pt: Tuple,
    pm: np.ndarray,
    label: str = "",
) -> Tuple[int, int, int]:
    """Ensure point is inside mask; snap to nearest mask voxel if not."""
    pi, pj, pk = int(round(float(pt[0]))), int(round(float(pt[1]))), int(round(float(pt[2])))
    shape = pm.shape
    pi = int(np.clip(pi, 0, shape[0] - 1))
    pj = int(np.clip(pj, 0, shape[1] - 1))
    pk = int(np.clip(pk, 0, shape[2] - 1))
    if pm[pi, pj, pk]:
        return (pi, pj, pk)
    coords = np.argwhere(pm)
    if coords.size == 0:
        return (pi, pj, pk)
    dists = np.sum((coords - np.array([pi, pj, pk])) ** 2, axis=1)
    nearest = coords[np.argmin(dists)]
    if label:
        logger.debug("_validate_point '%s': snapped to %s", label, nearest)
    return (int(nearest[0]), int(nearest[1]), int(nearest[2]))


def _find_boundary_point_in_primary(
    pm: np.ndarray,
    hi: Tuple[int, int, int],
    spacing: np.ndarray,
) -> Tuple[int, int, int]:
    """Find the nearest boundary voxel to a hotspot within the primary mask."""
    eroded = binary_erosion(pm, iterations=1)
    surface = pm & ~eroded
    surf_coords = np.argwhere(surface)
    if surf_coords.size == 0:
        surf_coords = np.argwhere(pm)
    dists = np.sum(
        ((surf_coords - np.array(hi)) * spacing) ** 2, axis=1
    )
    nearest = surf_coords[np.argmin(dists)]
    return (int(nearest[0]), int(nearest[1]), int(nearest[2]))


def _find_dmax_endpoints_in_primary(
    pm: np.ndarray,
    spacing: np.ndarray,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Find the two boundary points with maximum distance (Dmax endpoints)."""
    eroded = binary_erosion(pm, iterations=1)
    surface = pm & ~eroded
    coords = np.argwhere(surface)
    if len(coords) < 2:
        coords = np.argwhere(pm)
    if len(coords) < 2:
        c = np.argwhere(pm)
        if len(c) == 0:
            return (0, 0, 0), (0, 0, 0)
        return tuple(c[0]), tuple(c[0])  # type: ignore[return-value]

    world = coords * spacing

    def _farthest(pts, ref_idx):
        d = np.sum((pts - pts[ref_idx]) ** 2, axis=1)
        return int(np.argmax(d))

    a_idx = _farthest(world, 0)
    b_idx = _farthest(world, a_idx)
    a_idx = _farthest(world, b_idx)

    return (
        (int(coords[a_idx][0]), int(coords[a_idx][1]), int(coords[a_idx][2])),
        (int(coords[b_idx][0]), int(coords[b_idx][1]), int(coords[b_idx][2])),
    )


# ---------------------------------------------------------------------------
# Metric QC generators
# ---------------------------------------------------------------------------

def _generate_dmax_qc(
    pet: np.ndarray,
    ct: np.ndarray,
    tumor: np.ndarray,
    spacing: np.ndarray,
    qc: Dict,
    met: Dict,
    op: str,
    dpi: int,
    pvmax: float,
    primary: np.ndarray,
) -> None:
    """Generate Dmax QC figure.

    Shows orange line between endpoints A and B, yellow markers, annotations.
    """
    raw_a = qc.get("dmax_A", (0, 0, 0))
    raw_b = qc.get("dmax_B", (0, 0, 0))
    pt_a = _validate_point(raw_a, primary, "dmax_A")
    pt_b = _validate_point(raw_b, primary, "dmax_B")

    # Recompute from primary if needed
    if pt_a == pt_b:
        pt_a, pt_b = _find_dmax_endpoints_in_primary(primary, spacing)

    dmax_mm = float(met.get("Dmax_mm", 0.0))
    views = ["axial", "coronal", "sagittal"]
    zoom_view = "axial"
    vs = np.array(pet.shape, dtype=float)

    fig, axes = _make_4panel(pet, ct, primary, spacing, pvmax,
                              pt_a, views, zoom_view, primary)

    for panel_idx, view in enumerate(views):
        ax = axes[panel_idx]
        s_idx = _slice_for_point(pt_a, view)
        ms = _get_slice(primary.astype(float), view, s_idx)
        ms_disp = _orient_for_display(ms, view)

        xa, ya = _transform_point(pt_a, view, vs)
        xb, yb = _transform_point(pt_b, view, vs)

        _draw_line(ax, (xa, ya), (xb, yb), "orange", lw=2.0)
        _draw_marker(ax, (xa, ya), "yellow", marker="o", size=60)
        _draw_marker(ax, (xb, yb), "yellow", marker="o", size=60)
        ax.annotate("A", (xa, ya), color="white", fontsize=7,
                    xytext=(5, 5), textcoords="offset points")
        ax.annotate("B", (xb, yb), color="white", fontsize=7,
                    xytext=(5, 5), textcoords="offset points")

    # Zoomed panel — annotate with Dmax value
    ax_zoom = axes[3]
    z_idx = _slice_for_point(pt_a, zoom_view)
    ms_z = _get_slice(primary.astype(float), zoom_view, z_idx)
    ms_z_disp = _orient_for_display(ms_z, zoom_view)
    r0, r1, c0, c1 = _tumor_bbox_2d(ms_z_disp > 0.5)
    crop = (r0, r1, c0, c1)

    xac, yac = _transform_cropped(pt_a, zoom_view, vs, crop)
    xbc, ybc = _transform_cropped(pt_b, zoom_view, vs, crop)
    _draw_line(ax_zoom, (xac, yac), (xbc, ybc), "orange", lw=2.0)
    _draw_marker(ax_zoom, (xac, yac), "yellow", marker="o", size=80)
    _draw_marker(ax_zoom, (xbc, ybc), "yellow", marker="o", size=80)
    ax_zoom.annotate("A", (xac, yac), color="white", fontsize=8,
                     xytext=(5, 5), textcoords="offset points")
    ax_zoom.annotate("B", (xbc, ybc), color="white", fontsize=8,
                     xytext=(5, 5), textcoords="offset points")

    fig.suptitle(
        f"Dmax QC  |  Dmax = {dmax_mm:.1f} mm",
        color="white", fontsize=11,
    )

    a_patch = mpatches.Patch(color="orange", label=f"Dmax line ({dmax_mm:.1f} mm)")
    b_patch = mpatches.Patch(color="yellow", label="Endpoints A / B")
    c_patch = mpatches.Patch(color="cyan", label="Primary boundary")
    fig.legend(handles=[a_patch, b_patch, c_patch],
               loc="lower center", ncol=3, fontsize=7,
               facecolor="black", edgecolor="gray", labelcolor="white")

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    try:
        plt.savefig(op, dpi=dpi, bbox_inches="tight", facecolor="black")
        logger.info("Saved Dmax_QC → %s", op)
    except Exception as exc:
        logger.error("Failed to save Dmax_QC: %s", exc)
    plt.close(fig)


def _generate_nhoc_qc(
    pet: np.ndarray,
    ct: np.ndarray,
    tumor: np.ndarray,
    spacing: np.ndarray,
    qc: Dict,
    met: Dict,
    op: str,
    dpi: int,
    pvmax: float,
    primary: np.ndarray,
) -> None:
    """Generate NHOCmax QC figure.

    Magenta dashed line from hotspot to centroid; red star + cyan circle markers.
    """
    raw_hs = qc.get("nhoc_hotspot", (0, 0, 0))
    raw_ct = qc.get("nhoc_centroid", (0.0, 0.0, 0.0))
    hotspot = _validate_point(raw_hs, primary, "nhoc_hotspot")
    centroid = tuple(float(x) for x in raw_ct)

    nhoc = float(met.get("NHOCmax", 0.0))
    dmax = float(met.get("Dmax_mm", 1.0)) or 1.0

    views = ["axial", "coronal", "sagittal"]
    zoom_view = "axial"
    vs = np.array(pet.shape, dtype=float)

    fig, axes = _make_4panel(pet, ct, primary, spacing, pvmax,
                              hotspot, views, zoom_view, primary)

    for panel_idx, view in enumerate(views):
        ax = axes[panel_idx]
        xh, yh = _transform_point(hotspot, view, vs)
        xc, yc = _transform_point(centroid, view, vs)
        _draw_line(ax, (xh, yh), (xc, yc), "magenta", lw=1.8, ls="--")
        _draw_marker(ax, (xh, yh), "red", marker="*", size=100)
        _draw_marker(ax, (xc, yc), "cyan", marker="o", size=60)

    # Zoomed panel
    ax_zoom = axes[3]
    z_idx = _slice_for_point(hotspot, zoom_view)
    ms_z = _get_slice(primary.astype(float), zoom_view, z_idx)
    ms_z_disp = _orient_for_display(ms_z, zoom_view)
    r0, r1, c0, c1 = _tumor_bbox_2d(ms_z_disp > 0.5)
    crop = (r0, r1, c0, c1)
    xhc, yhc = _transform_cropped(hotspot, zoom_view, vs, crop)
    xcc, ycc = _transform_cropped(centroid, zoom_view, vs, crop)
    _draw_line(ax_zoom, (xhc, yhc), (xcc, ycc), "magenta", lw=2.0, ls="--")
    _draw_marker(ax_zoom, (xhc, yhc), "red", marker="*", size=120)
    _draw_marker(ax_zoom, (xcc, ycc), "cyan", marker="o", size=80)

    fig.suptitle(
        f"NHOCmax QC  |  NHOCmax = {nhoc:.3f}  (dist = {nhoc * dmax:.1f} mm / {dmax:.1f} mm)",
        color="white", fontsize=11,
    )

    hs_patch = mpatches.Patch(color="red", label="Hotspot (SUVmax)")
    ct_patch = mpatches.Patch(color="cyan", label="Centroid")
    line_patch = mpatches.Patch(color="magenta", label=f"NHOC line ({nhoc:.3f})")
    bnd_patch = mpatches.Patch(color="cyan", label="Primary boundary")
    fig.legend(handles=[hs_patch, ct_patch, line_patch, bnd_patch],
               loc="lower center", ncol=4, fontsize=7,
               facecolor="black", edgecolor="gray", labelcolor="white")

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    try:
        plt.savefig(op, dpi=dpi, bbox_inches="tight", facecolor="black")
        logger.info("Saved NHOCmax_QC → %s", op)
    except Exception as exc:
        logger.error("Failed to save NHOCmax_QC: %s", exc)
    plt.close(fig)


def _generate_nhop_qc(
    pet: np.ndarray,
    ct: np.ndarray,
    tumor: np.ndarray,
    spacing: np.ndarray,
    qc: Dict,
    met: Dict,
    op: str,
    dpi: int,
    pvmax: float,
    primary: np.ndarray,
) -> None:
    """Generate NHOPmax QC figure.

    Deepskyblue line from hotspot to boundary; red star + diamond markers.
    """
    raw_hs = qc.get("nhop_hotspot", (0, 0, 0))
    raw_bnd = qc.get("nhop_boundary", (0, 0, 0))
    hotspot = _validate_point(raw_hs, primary, "nhop_hotspot")
    boundary = _validate_point(raw_bnd, primary, "nhop_boundary")

    nhop = float(met.get("NHOPmax", 0.0))
    dmax = float(met.get("Dmax_mm", 1.0)) or 1.0

    views = ["axial", "coronal", "sagittal"]
    zoom_view = "axial"
    vs = np.array(pet.shape, dtype=float)

    fig, axes = _make_4panel(pet, ct, primary, spacing, pvmax,
                              hotspot, views, zoom_view, primary)

    for panel_idx, view in enumerate(views):
        ax = axes[panel_idx]
        xh, yh = _transform_point(hotspot, view, vs)
        xb, yb = _transform_point(boundary, view, vs)
        _draw_line(ax, (xh, yh), (xb, yb), "deepskyblue", lw=1.8)
        _draw_marker(ax, (xh, yh), "red", marker="*", size=100)
        _draw_marker(ax, (xb, yb), "deepskyblue", marker="D", size=60)

    # Zoomed panel
    ax_zoom = axes[3]
    z_idx = _slice_for_point(hotspot, zoom_view)
    ms_z = _get_slice(primary.astype(float), zoom_view, z_idx)
    ms_z_disp = _orient_for_display(ms_z, zoom_view)
    r0, r1, c0, c1 = _tumor_bbox_2d(ms_z_disp > 0.5)
    crop = (r0, r1, c0, c1)
    xhc, yhc = _transform_cropped(hotspot, zoom_view, vs, crop)
    xbc, ybc = _transform_cropped(boundary, zoom_view, vs, crop)
    _draw_line(ax_zoom, (xhc, yhc), (xbc, ybc), "deepskyblue", lw=2.0)
    _draw_marker(ax_zoom, (xhc, yhc), "red", marker="*", size=120)
    _draw_marker(ax_zoom, (xbc, ybc), "deepskyblue", marker="D", size=80)

    fig.suptitle(
        f"NHOPmax QC  |  NHOPmax = {nhop:.3f}  (dist = {nhop * dmax:.1f} mm / {dmax:.1f} mm)",
        color="white", fontsize=11,
    )

    hs_patch = mpatches.Patch(color="red", label="Hotspot (SUVmax)")
    bnd_patch_m = mpatches.Patch(color="deepskyblue", label="Boundary (farthest)")
    line_patch = mpatches.Patch(color="deepskyblue", label=f"NHOP line ({nhop:.3f})")
    cont_patch = mpatches.Patch(color="cyan", label="Primary boundary")
    fig.legend(handles=[hs_patch, bnd_patch_m, line_patch, cont_patch],
               loc="lower center", ncol=4, fontsize=7,
               facecolor="black", edgecolor="gray", labelcolor="white")

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    try:
        plt.savefig(op, dpi=dpi, bbox_inches="tight", facecolor="black")
        logger.info("Saved NHOPmax_QC → %s", op)
    except Exception as exc:
        logger.error("Failed to save NHOPmax_QC: %s", exc)
    plt.close(fig)


def _generate_getu_qc(
    pet: np.ndarray,
    ct: np.ndarray,
    tumor: np.ndarray,
    spacing: np.ndarray,
    qc: Dict,
    met: Dict,
    op: str,
    dpi: int,
    pvmax: float,
    primary: np.ndarray,
) -> None:
    """Generate gETU QC figure.

    Shows MTV threshold overlay (autumn colormap), yellow contours,
    cyan crosshair at centroid.
    """
    getu_val = float(met.get("gETU", 0.0))
    mtv_cm3 = float(met.get("gETU_MTV_cm3", float(met.get("MTV_cm3", 0.0))))
    suv_mean = float(met.get("gETU_SUVmean", 0.0))
    threshold = float(qc.get("getu_threshold", 0.41 * pvmax))
    suv_max = float(qc.get("getu_suv_max", pvmax))

    # Build MTV mask
    mtv_mask = primary & (pet >= threshold)

    raw_ctd = qc.get("getu_centroid", (0.0, 0.0, 0.0))
    coords_pm = np.argwhere(primary).astype(float)
    if coords_pm.size > 0:
        centroid_arr = coords_pm.mean(axis=0)
    else:
        centroid_arr = np.array([float(x) for x in raw_ctd])
    centroid = tuple(centroid_arr)

    views = ["axial", "coronal", "sagittal"]
    zoom_view = "axial"
    vs = np.array(pet.shape, dtype=float)

    # Use centroid as reference point for slice selection
    ref_pt = (int(round(centroid_arr[0])),
              int(round(centroid_arr[1])),
              int(round(centroid_arr[2])))

    fig, axes = _make_4panel(pet, ct, primary, spacing, pvmax,
                              ref_pt, views, zoom_view, primary)

    for panel_idx, view in enumerate(views):
        ax = axes[panel_idx]
        s_idx = _slice_for_point(ref_pt, view)
        aspect = _get_aspect_ratio(spacing, view)

        # MTV overlay (autumn colormap)
        mtv_slice = _get_slice(mtv_mask.astype(float), view, s_idx)
        pet_slice = _get_slice(pet, view, s_idx)
        mtv_pet = pet_slice * (mtv_slice > 0.5)
        mtv_disp = _orient_for_display(mtv_pet, view)

        ax.imshow(np.ma.masked_where(mtv_disp <= 0, mtv_disp),
                  cmap="autumn", alpha=0.6, vmin=threshold, vmax=suv_max,
                  aspect=aspect, origin="lower")

        # Yellow contour around MTV region
        mtv_disp_bin = _orient_for_display(mtv_slice, view)
        if mtv_disp_bin.any():
            ax.contour(mtv_disp_bin.astype(float), levels=[0.5],
                       colors=["yellow"], linewidths=[1.2])

        # Cyan crosshair at centroid
        xc, yc = _transform_point(ref_pt, view, vs)
        ch_size = 15.0
        _draw_line(ax, (xc - ch_size, yc), (xc + ch_size, yc),
                   "cyan", lw=1.2, zorder=11)
        _draw_line(ax, (xc, yc - ch_size), (xc, yc + ch_size),
                   "cyan", lw=1.2, zorder=11)

    # Zoomed panel with MTV overlay
    ax_zoom = axes[3]
    z_idx = _slice_for_point(ref_pt, zoom_view)
    aspect_z = _get_aspect_ratio(spacing, zoom_view)
    ms_z = _get_slice(primary.astype(float), zoom_view, z_idx)
    ms_z_disp = _orient_for_display(ms_z, zoom_view)
    r0, r1, c0, c1 = _tumor_bbox_2d(ms_z_disp > 0.5)
    crop = (r0, r1, c0, c1)

    mtv_slice_z = _get_slice(mtv_mask.astype(float), zoom_view, z_idx)
    pet_slice_z = _get_slice(pet, zoom_view, z_idx)
    mtv_pet_z = pet_slice_z * (mtv_slice_z > 0.5)

    def _crop2d(arr2d):
        return arr2d[r0:r1 + 1, c0:c1 + 1]

    mtv_pet_z_disp = _orient_for_display(_crop2d(mtv_pet_z), zoom_view)
    ax_zoom.imshow(np.ma.masked_where(mtv_pet_z_disp <= 0, mtv_pet_z_disp),
                   cmap="autumn", alpha=0.6, vmin=threshold, vmax=suv_max,
                   aspect=aspect_z, origin="lower")

    mtv_bin_z_disp = _orient_for_display(_crop2d(mtv_slice_z), zoom_view)
    if mtv_bin_z_disp.any():
        ax_zoom.contour(mtv_bin_z_disp.astype(float), levels=[0.5],
                        colors=["yellow"], linewidths=[1.5])

    xcc, ycc = _transform_cropped(ref_pt, zoom_view, vs, crop)
    ch_size_z = 12.0
    _draw_line(ax_zoom, (xcc - ch_size_z, ycc), (xcc + ch_size_z, ycc),
               "cyan", lw=1.5, zorder=11)
    _draw_line(ax_zoom, (xcc, ycc - ch_size_z), (xcc, ycc + ch_size_z),
               "cyan", lw=1.5, zorder=11)

    fig.suptitle(
        f"gETU QC  |  gETU = {getu_val:.2f}  |  MTV = {mtv_cm3:.2f} cm³  "
        f"|  SUVmean(MTV) = {suv_mean:.2f}  |  Threshold = {threshold:.2f}",
        color="white", fontsize=10,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.94])

    mtv_patch = mpatches.Patch(color="orange", label=f"MTV region (≥{threshold:.2f})")
    bnd_patch = mpatches.Patch(color="cyan", label="Primary boundary")
    ctd_patch = mpatches.Patch(color="cyan", label="Centroid crosshair")
    fig.legend(handles=[mtv_patch, bnd_patch, ctd_patch],
               loc="lower center", ncol=3, fontsize=7,
               facecolor="black", edgecolor="gray", labelcolor="white")

    try:
        plt.savefig(op, dpi=dpi, bbox_inches="tight", facecolor="black")
        logger.info("Saved gETU_QC → %s", op)
    except Exception as exc:
        logger.error("Failed to save gETU_QC: %s", exc)
    plt.close(fig)


# ---------------------------------------------------------------------------
# THE CRITICAL ENTRY POINT — called from Lung_ASP.py
# ---------------------------------------------------------------------------

def generate_metric_qc_overlays(
    pet_nifti_path: str,
    ct_nifti_path: str,
    tumor_mask_path: str,
    output_dir: str,
    metrics: Dict,
    qc_coords: Dict,
    dpi: int = 300,
) -> None:
    """Generate all four metric QC overlay images.

    Called from ``Lung_ASP.process_case`` after metrics computation.

    Produces:
      • ``<output_dir>/Dmax_QC.png``
      • ``<output_dir>/NHOCmax_QC.png``
      • ``<output_dir>/NHOPmax_QC.png``
      • ``<output_dir>/gETU_QC.png``

    Parameters
    ----------
    pet_nifti_path : str
        Path to PET NIfTI file (SUV).
    ct_nifti_path : str
        Path to CT NIfTI file (HU).
    tumor_mask_path : str
        Path to binary tumor mask NIfTI file.
    output_dir : str
        Directory for output images.
    metrics : dict
        Metrics dict returned by ``advanced_metrics.compute_all_metrics``.
    qc_coords : dict
        QC coordinates dict returned by ``advanced_metrics.compute_all_metrics``.
    dpi : int
        Output image resolution.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load NIfTI files
    try:
        pet_img = nib.load(pet_nifti_path)
        ct_img = nib.load(ct_nifti_path)
        mask_img = nib.load(tumor_mask_path)
    except Exception as exc:
        logger.error("generate_metric_qc_overlays: failed to load NIfTI — %s", exc)
        return

    pet = np.asarray(pet_img.get_fdata(), dtype=np.float32)
    ct = np.asarray(ct_img.get_fdata(), dtype=np.float32)
    tumor = np.asarray(mask_img.get_fdata(), dtype=bool)

    # 2. Resample CT and mask to PET space if shapes differ
    tshape = pet.shape

    def _resamp(vol: np.ndarray, order: int = 1) -> np.ndarray:
        if vol.shape == tshape:
            return vol
        factors = tuple(t / s for t, s in zip(tshape, vol.shape))
        return zoom(vol, factors, order=order)

    ct = _resamp(ct, order=1)
    tumor = _resamp(tumor.astype(np.float32), order=0) > 0.5

    # 3. Compute spacing and pet_vmax
    spacing = _voxel_spacing_from_affine(pet_img.affine)
    pet_vmax = float(np.percentile(pet[tumor], 99)) if tumor.any() else float(pet.max())
    if pet_vmax <= 0:
        pet_vmax = float(pet.max()) or 1.0

    # 4. Isolate primary mask ONCE
    primary = _isolate_primary_for_qc(tumor, pet, ct, spacing)

    qc_generators = [
        ("Dmax_QC.png",    _generate_dmax_qc),
        ("NHOCmax_QC.png", _generate_nhoc_qc),
        ("NHOPmax_QC.png", _generate_nhop_qc),
        ("gETU_QC.png",    _generate_getu_qc),
    ]

    # 5. Call each metric QC generator
    for fname, gen_fn in qc_generators:
        op = str(out_dir / fname)
        try:
            gen_fn(
                pet, ct, tumor, spacing,
                qc_coords, metrics,
                op, dpi, pet_vmax, primary,
            )
        except Exception as exc:
            logger.error("generate_metric_qc_overlays: %s failed — %s", fname, exc)

    # 6. Log completion
    logger.info(
        "generate_metric_qc_overlays: completed — Dmax, NHOCmax, NHOPmax, gETU QC saved to %s",
        output_dir,
    )


# ---------------------------------------------------------------------------
# ndimage import (needed inside the module)
# ---------------------------------------------------------------------------
from scipy import ndimage  # noqa: E402  (placed after function defs that use it)
