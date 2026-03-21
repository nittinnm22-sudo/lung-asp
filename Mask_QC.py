"""
Mask_QC.py  v5.9
QC overlay generation for Lung ASP pipeline.
Produces: Mask_QC.png, Dmax_QC.png, NHOPmax_QC.png, NHOCmax_QC.png, gETU_QC.png
"""

import logging
import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.ndimage import uniform_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Section 1: Basic helpers
# ---------------------------------------------------------------------------

def _voxel_spacing_from_affine(affine):
    """Extract voxel spacing (mm) from NIfTI affine matrix."""
    return tuple(float(np.abs(affine[i, i])) for i in range(3))


def _get_aspect_ratio(spacing, view):
    """Return matplotlib aspect ratio string for a given view and voxel spacing."""
    si, sj, sk = spacing
    if view == "axial":
        return sj / si
    elif view == "coronal":
        return sk / si
    elif view == "sagittal":
        return sk / sj
    return 1.0


def _orient_for_display(slice_2d, view):
    """Flip/rotate a 2D slice for standard radiological display."""
    arr = np.array(slice_2d)
    if view == "axial":
        return np.rot90(arr, k=1)
    elif view == "coronal":
        return np.rot90(arr, k=1)
    elif view == "sagittal":
        return np.rot90(arr, k=1)
    return arr


def _add_contour(ax, slice_2d, color, lw=1.5, ls="-"):
    """Overlay a binary mask contour on an axes."""
    arr = np.array(slice_2d, dtype=float)
    if arr.sum() > 0:
        ax.contour(arr, levels=[0.5], colors=[color], linewidths=[lw], linestyles=[ls])


# ---------------------------------------------------------------------------
# Section 2: 3x3 grid plot functions
# ---------------------------------------------------------------------------

def _plot_pet(ax, ps, ms, cs, title, vmax, view, aspect):
    """Plot PET slice with mask and constraint contours."""
    disp = _orient_for_display(ps, view)
    ax.imshow(disp, cmap="hot", vmin=0, vmax=vmax, aspect=aspect, origin="lower")
    if ms is not None:
        _add_contour(ax, _orient_for_display(ms, view), "cyan", lw=1.5)
    if cs is not None and cs.sum() > 0:
        _add_contour(ax, _orient_for_display(cs, view), "yellow", lw=1.0, ls="--")
    ax.set_title(title, fontsize=8, pad=2)
    ax.axis("off")


def _plot_ct(ax, cts, ms, cs, title, view, aspect):
    """Plot CT slice with mask and constraint contours."""
    disp = _orient_for_display(cts, view)
    ax.imshow(disp, cmap="gray", vmin=-1024, vmax=400, aspect=aspect, origin="lower")
    if ms is not None:
        _add_contour(ax, _orient_for_display(ms, view), "cyan", lw=1.5)
    if cs is not None and cs.sum() > 0:
        _add_contour(ax, _orient_for_display(cs, view), "yellow", lw=1.0, ls="--")
    ax.set_title(title, fontsize=8, pad=2)
    ax.axis("off")


def _plot_fusion(ax, cts, ps, ms, cs, title, vmax, view, aspect):
    """Plot CT/PET fusion with mask and constraint contours."""
    ct_disp = _orient_for_display(cts, view)
    pet_disp = _orient_for_display(ps, view)
    ax.imshow(ct_disp, cmap="gray", vmin=-1024, vmax=400, aspect=aspect, origin="lower")
    pet_alpha = np.clip(pet_disp / max(vmax, 1e-6), 0, 1) * 0.6
    ax.imshow(pet_disp, cmap="hot", vmin=0, vmax=vmax, alpha=pet_alpha,
              aspect=aspect, origin="lower")
    if ms is not None:
        _add_contour(ax, _orient_for_display(ms, view), "cyan", lw=1.5)
    if cs is not None and cs.sum() > 0:
        _add_contour(ax, _orient_for_display(cs, view), "yellow", lw=1.0, ls="--")
    ax.set_title(title, fontsize=8, pad=2)
    ax.axis("off")


# ---------------------------------------------------------------------------
# Section 3: Main QC overlay (3×3 grid) — produces Mask_QC.png
# ---------------------------------------------------------------------------

def generate_qc_overlays(
    pet_nifti_path,
    ct_nifti_path,
    tumor_mask_path,
    constraint_mask_path,
    output_path,
    dpi=150,
):
    """Generate 3x3 QC overlay figure showing axial/coronal/sagittal views in
    PET, CT, and fusion modes. Saves to output_path as Mask_QC.png.
    """
    logger.info(f"Generating Mask QC overlay -> {output_path}")
    # Load data
    pet_nii = nib.load(str(pet_nifti_path))
    ct_nii = nib.load(str(ct_nifti_path))
    mask_nii = nib.load(str(tumor_mask_path))

    pet_data = pet_nii.get_fdata().astype(np.float32)
    ct_data = ct_nii.get_fdata().astype(np.float32)
    tumor_mask = (mask_nii.get_fdata() > 0.5)
    spacing = _voxel_spacing_from_affine(ct_nii.affine)

    # Resample PET and mask to CT space if needed
    if pet_data.shape != ct_data.shape:
        from scipy.ndimage import zoom as spzoom
        pet_zoom = tuple(ct_data.shape[i] / pet_data.shape[i] for i in range(3))
        pet_data = spzoom(pet_data, pet_zoom, order=1)
    if tumor_mask.shape != ct_data.shape:
        from scipy.ndimage import zoom as spzoom
        mz = tuple(ct_data.shape[i] / tumor_mask.shape[i] for i in range(3))
        tumor_mask = spzoom(tumor_mask.astype(np.float32), mz, order=0) > 0.5

    # Load constraint mask if available
    constraint_mask = None
    try:
        if constraint_mask_path and Path(constraint_mask_path).exists():
            cm_nii = nib.load(str(constraint_mask_path))
            constraint_mask = (cm_nii.get_fdata() > 0.5)
            if constraint_mask.shape != ct_data.shape:
                from scipy.ndimage import zoom as spzoom
                cz = tuple(ct_data.shape[i] / constraint_mask.shape[i] for i in range(3))
                constraint_mask = spzoom(constraint_mask.astype(np.float32), cz, order=0) > 0.5
    except Exception as e:
        logger.warning(f"Could not load constraint mask: {e}")

    if tumor_mask.sum() == 0:
        logger.error("Tumor mask is empty, cannot generate QC")
        return

    # Find center of mass for slice selection
    com = ndimage.center_of_mass(tumor_mask)
    ci, cj, ck = [int(round(x)) for x in com]
    ci = np.clip(ci, 0, ct_data.shape[0] - 1)
    cj = np.clip(cj, 0, ct_data.shape[1] - 1)
    ck = np.clip(ck, 0, ct_data.shape[2] - 1)

    # PET vmax
    pvmax = float(pet_data[tumor_mask].max()) * 1.1
    pvmax = max(pvmax, 1.0)

    # Extract slices
    def get_slices(view, idx):
        if view == "axial":
            return (pet_data[:, :, idx], ct_data[:, :, idx],
                    tumor_mask[:, :, idx],
                    constraint_mask[:, :, idx] if constraint_mask is not None else None)
        elif view == "coronal":
            return (pet_data[:, idx, :], ct_data[:, idx, :],
                    tumor_mask[:, idx, :],
                    constraint_mask[:, idx, :] if constraint_mask is not None else None)
        elif view == "sagittal":
            return (pet_data[idx, :, :], ct_data[idx, :, :],
                    tumor_mask[idx, :, :],
                    constraint_mask[idx, :, :] if constraint_mask is not None else None)

    views = [("axial", ck), ("coronal", cj), ("sagittal", ci)]
    view_labels = ["Axial", "Coronal", "Sagittal"]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.patch.set_facecolor("black")

    for col, (view, idx) in enumerate(views):
        asp = _get_aspect_ratio(spacing, view)
        ps, cts_sl, ms, cs = get_slices(view, idx)
        label = view_labels[col]
        _plot_pet(axes[0, col], ps, ms, cs, f"PET — {label}", pvmax, view, asp)
        _plot_ct(axes[1, col], cts_sl, ms, cs, f"CT — {label}", view, asp)
        _plot_fusion(axes[2, col], cts_sl, ps, ms, cs, f"Fusion — {label}", pvmax, view, asp)
        for row in range(3):
            axes[row, col].set_facecolor("black")

    # Row labels
    for row, label in enumerate(["PET", "CT", "Fusion"]):
        axes[row, 0].set_ylabel(label, color="white", fontsize=10)

    # Build title
    n_vox = int(tumor_mask.sum())
    vox_vol = float(np.prod(spacing))
    mtv_cc = n_vox * vox_vol / 1000.0
    suv_max = float(pet_data[tumor_mask].max()) if tumor_mask.sum() > 0 else 0.0
    fig.suptitle(
        f"Mask QC — MTV={mtv_cc:.1f}cc  SUVmax={suv_max:.2f}  "
        f"Voxels={n_vox:,}",
        color="white", fontsize=11, y=0.99,
    )

    # Legend
    cyan_patch = mpatches.Patch(edgecolor="cyan", facecolor="none", label="Tumor mask")
    if constraint_mask is not None:
        yellow_patch = mpatches.Patch(edgecolor="yellow", facecolor="none",
                                       linestyle="--", label="Constraint mask")
        fig.legend(handles=[cyan_patch, yellow_patch], loc="lower center",
                   ncol=2, fontsize=8, facecolor="black", labelcolor="white",
                   framealpha=0.7)
    else:
        fig.legend(handles=[cyan_patch], loc="lower center",
                   fontsize=8, facecolor="black", labelcolor="white", framealpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    plt.close(fig)
    logger.info(f"Mask_QC.png saved: {output_path}")


# ---------------------------------------------------------------------------
# Section 4: Primary isolation for QC
# ---------------------------------------------------------------------------

def _bottleneck_qc(clean, pet_data, ct_data, spacing, smi, smv, os):
    """S0: Bottleneck isolation — try 1mm erosion to disconnect components."""
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
        logger.info(f"  QC S0 (bottleneck): kept={pr.sum():,}")
        return pr.astype(bool)
    return None


def _convex_hull_trim(clean, ct_data, pet_data, spacing, smi, smv, os):
    """S1: Convex hull isolation — use convex hull of high-SUV region."""
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
        hot_slice = hot[:, :, k]
        if hot_slice.sum() < 3:
            hull[:, :, k] = hot_slice
            continue
        try:
            hull[:, :, k] = convex_hull_image(hot_slice)
        except Exception:
            hull[:, :, k] = hot_slice
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
        logger.info(f"  QC S1 (convex hull): kept={pr.sum():,}")
        return pr.astype(bool)
    return None


def _lung_adjacency_filter(clean, ct_data, pet_data, spacing, smi, os, pm):
    """S2: Lung adjacency filter — keep regions adjacent to lung parenchyma."""
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
    logger.info(f"  QC S2 (lung adj): kept={pr.sum():,}, removed={rm:,}")
    return pr.astype(bool)


def _erosion_isolation(clean, spacing, smi, os):
    """S3: Erosion isolation — progressively erode until multiple components appear."""
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
            logger.info(f"  QC S3 (erosion={em}mm): kept={pr.sum():,}")
            return pr.astype(bool)
    return None


def _pet_distance_isolation(clean, pet_data, spacing, smi, smv, os):
    """S4: PET-distance isolation — weight by PET intensity and spatial distance."""
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
        logger.info(f"  QC S4 (distance): kept={pd.sum():,}")
        return pd.astype(bool)
    return None


def _ct_density_filter(clean, ct_data, spacing, smi, os):
    """S5: CT density filter — exclude non-lung-like density regions."""
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
    logger.info(f"  QC S5 (CT filter): kept={pk.sum():,}, removed={rm:,}")
    return pk.astype(bool)


def _isolate_primary_inner(tumor_mask, pet_data, ct_data, spacing):
    """Try all 6 isolation strategies in order. Return first successful result."""
    s = ndimage.generate_binary_structure(3, 1)
    # Get all connected components
    lc, nc = ndimage.label(tumor_mask, structure=s)
    if nc == 0:
        return tumor_mask.astype(bool)
    # Work on the full clean mask (all components)
    full_clean = tumor_mask.astype(bool)
    os_val = int(full_clean.sum())
    # Find SUVmax location within tumor
    masked_pet = pet_data.copy()
    masked_pet[~full_clean] = 0
    smi = tuple(np.unravel_index(np.argmax(masked_pet), masked_pet.shape))
    smv = float(masked_pet[smi])
    pm = full_clean.copy()
    logger.info(f"  QC isolation: SUVmax={smv:.2f} at {smi}, mask_size={os_val}")
    # If only one component, return as-is
    if nc == 1:
        logger.info("  QC isolation: single component, no isolation needed")
        return full_clean
    strategies = [
        ("S0", lambda: _bottleneck_qc(full_clean, pet_data, ct_data, spacing, smi, smv, os_val)),
        ("S1", lambda: _convex_hull_trim(full_clean, ct_data, pet_data, spacing, smi, smv, os_val)),
        ("S2", lambda: _lung_adjacency_filter(full_clean, ct_data, pet_data, spacing, smi, os_val, pm)),
        ("S3", lambda: _erosion_isolation(full_clean, spacing, smi, os_val)),
        ("S4", lambda: _pet_distance_isolation(full_clean, pet_data, spacing, smi, smv, os_val)),
        ("S5", lambda: _ct_density_filter(full_clean, ct_data, spacing, smi, os_val)),
    ]
    for name, fn in strategies:
        try:
            result = fn()
            if result is not None and result.sum() >= 20:
                logger.info(f"  QC primary isolation: {name} succeeded")
                return result
        except Exception as e:
            logger.warning(f"  QC {name} failed: {e}")
    logger.info("  QC isolation: all strategies failed, using largest component")
    sizes = ndimage.sum(tumor_mask, lc, range(1, nc + 1))
    best_label = int(np.argmax(sizes)) + 1
    return (lc == best_label).astype(bool)


def _isolate_primary_for_qc(tumor_mask, pet_data, ct_data, spacing):
    """Wrapper for primary isolation with fallback to full mask."""
    try:
        result = _isolate_primary_inner(tumor_mask, pet_data, ct_data, spacing)
        if result is not None and result.sum() >= 20:
            return result
    except Exception as e:
        logger.warning(f"Primary isolation for QC failed: {e}")
    return tumor_mask.astype(bool)


# ---------------------------------------------------------------------------
# Section 5: Coordinate transform helpers for metric QC
# ---------------------------------------------------------------------------

def _transform_point(pt, view, vs):
    """Transform voxel ijk point to display (x, y) coordinates for a given view.

    Convention:
      axial   (looking down k): x = j*vs[1], y = i*vs[0]
      coronal (looking down j): x = i*vs[0], y = k*vs[2]
      sagittal(looking down i): x = j*vs[1], y = k*vs[2]

    After _orient_for_display (rot90 k=1), the axes are rotated, so:
      axial:    x_display = i*vs[0],  y_display = j*vs[1]  (after rot90)
      coronal:  x_display = k*vs[2],  y_display = i*vs[0]
      sagittal: x_display = k*vs[2],  y_display = j*vs[1]
    """
    i, j, k = pt
    si, sj, sk = vs
    if view == "axial":
        return (float(i) * si, float(j) * sj)
    elif view == "coronal":
        return (float(k) * sk, float(i) * si)
    elif view == "sagittal":
        return (float(k) * sk, float(j) * sj)
    return (float(j) * sj, float(i) * si)


def _slice_for_point(pt, view):
    """Return the slice index for a point in a given view."""
    i, j, k = pt
    if view == "axial":
        return int(k)
    elif view == "coronal":
        return int(j)
    elif view == "sagittal":
        return int(i)
    return int(k)


def _clamp(val, lo, hi):
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, val))


def _get_slice(vol, view, idx):
    """Extract a 2D slice from a 3D volume for the given view."""
    ni, nj, nk = vol.shape
    if view == "axial":
        idx = _clamp(idx, 0, nk - 1)
        return vol[:, :, idx]
    elif view == "coronal":
        idx = _clamp(idx, 0, nj - 1)
        return vol[:, idx, :]
    elif view == "sagittal":
        idx = _clamp(idx, 0, ni - 1)
        return vol[idx, :, :]
    idx = _clamp(idx, 0, nk - 1)
    return vol[:, :, idx]


def _tumor_bbox_2d(ts, margin=10):
    """Find 2D bounding box (r0,r1,c0,c1) of tumor in a 2D slice, with margin."""
    rows = np.any(ts, axis=1)
    cols = np.any(ts, axis=0)
    if not rows.any():
        return None
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    h, w = ts.shape
    r0 = max(0, r0 - margin)
    r1 = min(h - 1, r1 + margin)
    c0 = max(0, c0 - margin)
    c1 = min(w - 1, c1 + margin)
    return (r0, r1, c0, c1)


def _transform_cropped(pt, view, vs, crop):
    """Transform ijk point to display coordinates after cropping.

    crop = (r0, r1, c0, c1) in the 2D display image (post-orient_for_display).
    Returns (x, y) in the cropped coordinate system.
    """
    x, y = _transform_point(pt, view, vs)
    r0, r1, c0, c1 = crop
    # x corresponds to column axis, y corresponds to row axis in display
    # After orient_for_display (rot90 k=1), axes are:
    #   rows (vertical axis, y in imshow) — depends on view
    #   cols (horizontal axis, x in imshow)
    # Subtract crop offsets
    x_cropped = x - c0
    y_cropped = y - r0
    return (x_cropped, y_cropped)


def _plot_base(ax, cs, ps, ms, view, aspect, pvmax, crop):
    """Plot base CT/PET fusion with tumor contour on an axes, with cropping.

    crop = (r0, r1, c0, c1) or None for full image.
    """
    ct_disp = _orient_for_display(cs, view)
    pet_disp = _orient_for_display(ps, view)
    mask_disp = _orient_for_display(ms, view)
    if crop is not None:
        r0, r1, c0, c1 = crop
        ct_disp = ct_disp[r0:r1+1, c0:c1+1]
        pet_disp = pet_disp[r0:r1+1, c0:c1+1]
        mask_disp = mask_disp[r0:r1+1, c0:c1+1]
    ax.imshow(ct_disp, cmap="gray", vmin=-1024, vmax=400,
              aspect=aspect, origin="lower")
    pet_alpha = np.clip(pet_disp / max(pvmax, 1e-6), 0, 1) * 0.6
    ax.imshow(pet_disp, cmap="hot", vmin=0, vmax=pvmax,
              alpha=pet_alpha, aspect=aspect, origin="lower")
    if mask_disp.sum() > 0:
        ax.contour(mask_disp, levels=[0.5], colors=["cyan"], linewidths=[1.5])
    ax.axis("off")


def _draw_marker(ax, xy, color, marker="*", size=120, edge="white", ew=0.5, zorder=10):
    """Draw a scatter marker on ax at position (x, y)."""
    ax.scatter([xy[0]], [xy[1]], c=color, marker=marker,
               s=size, edgecolors=edge, linewidths=ew, zorder=zorder)


def _draw_line(ax, p1, p2, color, lw=1.5, ls="-", zorder=8):
    """Draw a line between two (x, y) display points."""
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
            color=color, linewidth=lw, linestyle=ls, zorder=zorder)


def _make_4panel(pet, ct, dm, spacing, pvmax, pp, views=("axial", "coronal", "sagittal"),
                 zv="axial", zm=30):
    """Create a 4-panel figure: 3 full-view panels + 1 zoomed panel.

    Parameters
    ----------
    pet, ct, dm : np.ndarray
        PET, CT, and mask (primary) arrays
    spacing : tuple
        Voxel spacing (si, sj, sk)
    pvmax : float
        PET display max
    pp : tuple
        Primary point (i, j, k) around which to center zoom
    views : tuple
        Three views for panels 1-3
    zv : str
        View for zoomed panel (panel 4)
    zm : int
        Zoom margin in pixels

    Returns
    -------
    fig : matplotlib Figure
    axes : list of 4 Axes
    slices : list of 3 slice indices (for views)
    zslice : int
        Slice index for zoomed panel
    crops : list of 4 crop tuples (r0, r1, c0, c1) or None
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor("black")
    for ax in axes:
        ax.set_facecolor("black")

    slice_indices = [_slice_for_point(pp, v) for v in views]
    zslice = _slice_for_point(pp, zv)

    crops = []
    for idx_panel, (ax, view, sidx) in enumerate(zip(axes[:3], views, slice_indices)):
        asp = _get_aspect_ratio(spacing, view)
        cs = _get_slice(ct, view, sidx)
        ps = _get_slice(pet, view, sidx)
        ms = _get_slice(dm, view, sidx)
        _plot_base(ax, cs, ps, ms, view, asp, pvmax, None)
        crops.append(None)
        ax.set_title(view.capitalize(), color="white", fontsize=9)

    # Zoomed panel (panel 4)
    ax_z = axes[3]
    asp_z = _get_aspect_ratio(spacing, zv)
    cs_z = _get_slice(ct, zv, zslice)
    ps_z = _get_slice(pet, zv, zslice)
    ms_z = _get_slice(dm, zv, zslice)
    ms_disp = _orient_for_display(ms_z, zv)
    crop = _tumor_bbox_2d(ms_disp, margin=zm)
    if crop is None:
        # Use full image
        crop = (0, ms_disp.shape[0] - 1, 0, ms_disp.shape[1] - 1)
    _plot_base(ax_z, cs_z, ps_z, ms_z, zv, asp_z, pvmax, crop)
    crops.append(crop)
    ax_z.set_title("Zoomed", color="white", fontsize=9)

    return fig, list(axes), slice_indices, zslice, crops


def _validate_point(pt, pm, label="point"):
    """Validate that pt is inside primary mask pm. Snap to nearest if not."""
    i, j, k = [int(round(x)) for x in pt]
    i = _clamp(i, 0, pm.shape[0] - 1)
    j = _clamp(j, 0, pm.shape[1] - 1)
    k = _clamp(k, 0, pm.shape[2] - 1)
    if pm[i, j, k]:
        return (i, j, k)
    # Snap to nearest voxel in mask
    coords = np.argwhere(pm)
    if len(coords) == 0:
        logger.warning(f"  {label}: primary mask is empty")
        return (i, j, k)
    dists = np.sum((coords - np.array([i, j, k])) ** 2, axis=1)
    nearest = coords[np.argmin(dists)]
    logger.info(f"  {label}: snapped ({i},{j},{k}) -> ({nearest[0]},{nearest[1]},{nearest[2]})")
    return tuple(nearest)


def _find_boundary_point_in_primary(pm, hi, spacing):
    """Find the boundary point of primary mask closest to the hotspot hi.

    Parameters
    ----------
    pm : np.ndarray (bool)
        Primary mask
    hi : tuple (i, j, k)
        Hotspot voxel index
    spacing : tuple
        Voxel spacing

    Returns
    -------
    tuple (i, j, k)
        Nearest boundary voxel
    """
    s = ndimage.generate_binary_structure(3, 1)
    eroded = ndimage.binary_erosion(pm, structure=s)
    boundary = pm & ~eroded
    bc = np.argwhere(boundary)
    if len(bc) == 0:
        bc = np.argwhere(pm)
    if len(bc) == 0:
        return hi
    hi_arr = np.array(hi, dtype=float)
    dists = np.sqrt(np.sum(((bc - hi_arr) * np.array(spacing)) ** 2, axis=1))
    return tuple(bc[np.argmin(dists)])


def _find_dmax_endpoints_in_primary(pm, spacing):
    """Find Dmax endpoints by pairwise distances on boundary voxels.

    Returns (pt_a, pt_b, dmax_mm).
    """
    s = ndimage.generate_binary_structure(3, 1)
    eroded = ndimage.binary_erosion(pm, structure=s)
    boundary = pm & ~eroded
    bc = np.argwhere(boundary)
    if len(bc) < 2:
        bc = np.argwhere(pm)
    if len(bc) < 2:
        coords = np.argwhere(pm)
        if len(coords) < 2:
            return (tuple(coords[0]) if len(coords) > 0 else (0, 0, 0),
                    tuple(coords[0]) if len(coords) > 0 else (0, 0, 0), 0.0)
        return (tuple(coords[0]), tuple(coords[-1]), 0.0)
    # Subsample for efficiency
    max_bc = 500
    if len(bc) > max_bc:
        stride = max(len(bc) // max_bc, 1)
        bc = bc[::stride]
    bc_phys = bc * np.array(spacing)
    diffs = bc_phys[:, None, :] - bc_phys[None, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    idx = np.unravel_index(np.argmax(dists), dists.shape)
    dmax_mm = float(dists[idx])
    return (tuple(bc[idx[0]]), tuple(bc[idx[1]]), dmax_mm)


# ---------------------------------------------------------------------------
# Section 6: Metric QC generators
# ---------------------------------------------------------------------------

def _generate_dmax_qc(pet, ct, tumor, spacing, qc, met, op, dpi, pvmax, primary):
    """Generate Dmax_QC.png — 4-panel figure with Dmax line and endpoints."""
    logger.info("Generating Dmax_QC.png...")
    vs = spacing

    # Get Dmax endpoints — recompute on primary mask for accuracy
    pt_a, pt_b, dmax_mm = _find_dmax_endpoints_in_primary(primary, spacing)

    # Use midpoint as center for panels
    mid = tuple(int(round((pt_a[ax_i] + pt_b[ax_i]) / 2)) for ax_i in range(3))
    mid = _validate_point(mid, primary, "Dmax midpoint")

    views = ("axial", "coronal", "sagittal")
    fig, axes, slice_indices, zslice, crops = _make_4panel(
        pet, ct, primary, spacing, pvmax, mid,
        views=views, zv="axial", zm=35
    )

    # Draw Dmax line and endpoints on each panel
    for ax_idx, (ax, view, sidx, crop) in enumerate(zip(axes[:3], views, slice_indices, crops[:3])):
        xa, ya = _transform_point(pt_a, view, vs)
        xb, yb = _transform_point(pt_b, view, vs)
        _draw_line(ax, (xa, ya), (xb, yb), "orange", lw=2.0, ls="-")
        _draw_marker(ax, (xa, ya), "yellow", marker="o", size=80)
        _draw_marker(ax, (xb, yb), "orange", marker="o", size=80)

    # Zoomed panel
    ax_z = axes[3]
    crop_z = crops[3]
    xa_z, ya_z = _transform_cropped(pt_a, "axial", vs, crop_z) if crop_z else _transform_point(pt_a, "axial", vs)
    xb_z, yb_z = _transform_cropped(pt_b, "axial", vs, crop_z) if crop_z else _transform_point(pt_b, "axial", vs)
    _draw_line(ax_z, (xa_z, ya_z), (xb_z, yb_z), "orange", lw=2.5, ls="-")
    _draw_marker(ax_z, (xa_z, ya_z), "yellow", marker="o", size=100, zorder=11)
    _draw_marker(ax_z, (xb_z, yb_z), "orange", marker="o", size=100, zorder=11)
    ax_z.annotate(f"Dmax={dmax_mm:.1f}mm",
                  xy=((xa_z + xb_z) / 2, (ya_z + yb_z) / 2),
                  color="white", fontsize=8, ha="center",
                  bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))

    # Legend
    handles = [
        mpatches.Patch(edgecolor="cyan", facecolor="none", label="Primary boundary"),
        mpatches.Patch(color="orange", label=f"Dmax line ({dmax_mm:.1f}mm)"),
        plt.scatter([], [], c="yellow", marker="o", s=60, label="Endpoint A"),
        plt.scatter([], [], c="orange", marker="o", s=60, label="Endpoint B"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7,
               facecolor="black", labelcolor="white", framealpha=0.7)

    suv_max = float(met.get("SUVmax", 0.0))
    fig.suptitle(
        f"Dmax QC — Dmax={dmax_mm:.1f}mm  "
        f"A={pt_a}  B={pt_b}  SUVmax={suv_max:.2f}",
        color="white", fontsize=9, y=1.01
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1.0])
    out_path = Path(op) / "Dmax_QC.png"
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    plt.close(fig)
    logger.info(f"Dmax_QC.png saved: {out_path}")


def _generate_nhop_qc(pet, ct, tumor, spacing, qc, met, op, dpi, pvmax, primary):
    """Generate NHOPmax_QC.png — hotspot to nearest primary boundary."""
    logger.info("Generating NHOPmax_QC.png...")
    vs = spacing

    # Get hotspot and boundary from qc_coords or recompute
    if "hotspot_ijk" in qc:
        hotspot = tuple(int(x) for x in qc["hotspot_ijk"])
    else:
        masked_pet = pet.copy()
        masked_pet[~primary] = 0
        hotspot = tuple(np.unravel_index(np.argmax(masked_pet), masked_pet.shape))

    hotspot = _validate_point(hotspot, primary, "NHOPmax hotspot")

    # Find nearest boundary point
    boundary_pt = _find_boundary_point_in_primary(primary, hotspot, spacing)

    nhopmax_mm = float(met.get("NHOPmax_mm", 0.0))
    suv_max = float(met.get("SUVmax", 0.0))

    views = ("axial", "coronal", "sagittal")
    fig, axes, slice_indices, zslice, crops = _make_4panel(
        pet, ct, primary, spacing, pvmax, hotspot,
        views=views, zv="axial", zm=35
    )

    # Draw on each view panel
    for ax, view, sidx, crop in zip(axes[:3], views, slice_indices, crops[:3]):
        xh, yh = _transform_point(hotspot, view, vs)
        xb, yb = _transform_point(boundary_pt, view, vs)
        _draw_line(ax, (xh, yh), (xb, yb), "lime", lw=1.5, ls="-")
        _draw_marker(ax, (xh, yh), "red", marker="*", size=150, zorder=12)
        _draw_marker(ax, (xb, yb), "lime", marker="D", size=80, zorder=11)

    # Zoomed panel
    ax_z = axes[3]
    crop_z = crops[3]
    if crop_z is not None:
        xh_z, yh_z = _transform_cropped(hotspot, "axial", vs, crop_z)
        xb_z, yb_z = _transform_cropped(boundary_pt, "axial", vs, crop_z)
    else:
        xh_z, yh_z = _transform_point(hotspot, "axial", vs)
        xb_z, yb_z = _transform_point(boundary_pt, "axial", vs)
    _draw_line(ax_z, (xh_z, yh_z), (xb_z, yb_z), "lime", lw=2.0, ls="-")
    _draw_marker(ax_z, (xh_z, yh_z), "red", marker="*", size=180, zorder=12)
    _draw_marker(ax_z, (xb_z, yb_z), "lime", marker="D", size=100, zorder=11)
    ax_z.annotate(f"NHOPmax={nhopmax_mm:.1f}mm",
                  xy=((xh_z + xb_z) / 2, (yh_z + yb_z) / 2),
                  color="white", fontsize=8, ha="center",
                  bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))

    # Legend
    handles = [
        mpatches.Patch(edgecolor="cyan", facecolor="none", label="Primary boundary"),
        plt.scatter([], [], c="red", marker="*", s=100, label=f"SUVmax hotspot ({suv_max:.2f})"),
        plt.scatter([], [], c="lime", marker="D", s=60, label="Nearest boundary"),
        mpatches.Patch(color="lime", label=f"Hotspot→Boundary ({nhopmax_mm:.1f}mm)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7,
               facecolor="black", labelcolor="white", framealpha=0.7)

    fig.suptitle(
        f"NHOPmax QC — NHOPmax={nhopmax_mm:.1f}mm  "
        f"SUVmax={suv_max:.2f}  hotspot={hotspot}  boundary={boundary_pt}",
        color="white", fontsize=9, y=1.01
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1.0])
    out_path = Path(op) / "NHOPmax_QC.png"
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    plt.close(fig)
    logger.info(f"NHOPmax_QC.png saved: {out_path}")


def _generate_nhoc_qc(pet, ct, tumor, spacing, qc, met, op, dpi, pvmax, primary):
    """Generate NHOCmax_QC.png — hotspot to centroid distance."""
    logger.info("Generating NHOCmax_QC.png...")
    vs = spacing

    # Get hotspot from qc_coords or recompute
    if "hotspot_ijk" in qc:
        hotspot = tuple(int(x) for x in qc["hotspot_ijk"])
    else:
        masked_pet = pet.copy()
        masked_pet[~primary] = 0
        hotspot = tuple(np.unravel_index(np.argmax(masked_pet), masked_pet.shape))
    hotspot = _validate_point(hotspot, primary, "NHOCmax hotspot")

    # Get centroid from qc_coords or compute
    if "centroid_ijk" in qc:
        centroid = tuple(int(x) for x in qc["centroid_ijk"])
    else:
        com = ndimage.center_of_mass(primary)
        centroid = tuple(int(round(x)) for x in com)
    centroid = _validate_point(centroid, primary, "NHOCmax centroid")

    nhocmax_mm = float(met.get("NHOCmax_mm", 0.0))
    suv_max = float(met.get("SUVmax", 0.0))

    views = ("axial", "coronal", "sagittal")
    fig, axes, slice_indices, zslice, crops = _make_4panel(
        pet, ct, primary, spacing, pvmax, hotspot,
        views=views, zv="axial", zm=35
    )

    # Draw on each view panel
    for ax, view, sidx, crop in zip(axes[:3], views, slice_indices, crops[:3]):
        xh, yh = _transform_point(hotspot, view, vs)
        xc, yc = _transform_point(centroid, view, vs)
        _draw_line(ax, (xh, yh), (xc, yc), "magenta", lw=1.5, ls="--")
        _draw_marker(ax, (xh, yh), "red", marker="*", size=150, zorder=12)
        _draw_marker(ax, (xc, yc), "magenta", marker="o", size=80, zorder=11)

    # Zoomed panel
    ax_z = axes[3]
    crop_z = crops[3]
    if crop_z is not None:
        xh_z, yh_z = _transform_cropped(hotspot, "axial", vs, crop_z)
        xc_z, yc_z = _transform_cropped(centroid, "axial", vs, crop_z)
    else:
        xh_z, yh_z = _transform_point(hotspot, "axial", vs)
        xc_z, yc_z = _transform_point(centroid, "axial", vs)
    _draw_line(ax_z, (xh_z, yh_z), (xc_z, yc_z), "magenta", lw=2.0, ls="--")
    _draw_marker(ax_z, (xh_z, yh_z), "red", marker="*", size=180, zorder=12)
    _draw_marker(ax_z, (xc_z, yc_z), "magenta", marker="o", size=100, zorder=11)
    ax_z.annotate(f"NHOCmax={nhocmax_mm:.1f}mm",
                  xy=((xh_z + xc_z) / 2, (yh_z + yc_z) / 2),
                  color="white", fontsize=8, ha="center",
                  bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))

    # Legend
    handles = [
        mpatches.Patch(edgecolor="cyan", facecolor="none", label="Primary boundary"),
        plt.scatter([], [], c="red", marker="*", s=100, label=f"SUVmax hotspot ({suv_max:.2f})"),
        plt.scatter([], [], c="magenta", marker="o", s=60, label="Centroid"),
        mpatches.Patch(color="magenta", linestyle="--",
                       label=f"Hotspot→Centroid ({nhocmax_mm:.1f}mm)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7,
               facecolor="black", labelcolor="white", framealpha=0.7)

    fig.suptitle(
        f"NHOCmax QC — NHOCmax={nhocmax_mm:.1f}mm  "
        f"SUVmax={suv_max:.2f}  hotspot={hotspot}  centroid={centroid}",
        color="white", fontsize=9, y=1.01
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1.0])
    out_path = Path(op) / "NHOCmax_QC.png"
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    plt.close(fig)
    logger.info(f"NHOCmax_QC.png saved: {out_path}")


def _generate_getu_qc(pet, ct, tumor, spacing, qc, met, op, dpi, pvmax, primary):
    """Generate gETU_QC.png — MTV threshold contour overlay with SUVmax marker."""
    logger.info("Generating gETU_QC.png...")
    vs = spacing

    # Get gETU metrics
    getu_index = float(met.get("gETU_index", 0.0))
    getu_raw = float(met.get("gETU_raw", 0.0))
    mtv_threshold = float(met.get("mtv_threshold", 0.0))
    suv_max = float(met.get("SUVmax", 0.0))
    mtv_cc = float(met.get("MTV", 0.0))

    # Get hotspot (SUVmax location)
    if "hotspot_ijk" in qc:
        hotspot = tuple(int(x) for x in qc["hotspot_ijk"])
    else:
        masked_pet = pet.copy()
        masked_pet[~tumor] = 0
        hotspot = tuple(np.unravel_index(np.argmax(masked_pet), masked_pet.shape))
    hotspot = _validate_point(hotspot, primary, "gETU hotspot")

    # Create MTV mask: PET >= threshold within tumor
    if mtv_threshold > 0:
        mtv_mask = (pet >= mtv_threshold) & tumor
    else:
        mtv_mask = tumor.copy()

    views = ("axial", "coronal", "sagittal")
    fig, axes, slice_indices, zslice, crops = _make_4panel(
        pet, ct, primary, spacing, pvmax, hotspot,
        views=views, zv="axial", zm=35
    )

    # Draw MTV threshold contour and SUVmax marker on each panel
    for ax, view, sidx, crop in zip(axes[:3], views, slice_indices, crops[:3]):
        # MTV threshold contour
        mtv_sl = _get_slice(mtv_mask, view, sidx)
        mtv_disp = _orient_for_display(mtv_sl, view)
        if mtv_disp.sum() > 0:
            ax.contour(mtv_disp, levels=[0.5], colors=["yellow"],
                       linewidths=[1.2], linestyles=["--"])
        # SUVmax marker
        xh, yh = _transform_point(hotspot, view, vs)
        _draw_marker(ax, (xh, yh), "red", marker="*", size=150, zorder=12)

    # Zoomed panel
    ax_z = axes[3]
    crop_z = crops[3]
    # MTV contour on zoomed panel
    mtv_z = _get_slice(mtv_mask, "axial", zslice)
    mtv_z_disp = _orient_for_display(mtv_z, "axial")
    if crop_z is not None:
        r0, r1, c0, c1 = crop_z
        mtv_z_disp_crop = mtv_z_disp[r0:r1+1, c0:c1+1]
    else:
        mtv_z_disp_crop = mtv_z_disp
    if mtv_z_disp_crop.sum() > 0:
        ax_z.contour(mtv_z_disp_crop, levels=[0.5], colors=["yellow"],
                     linewidths=[1.5], linestyles=["--"])
    # SUVmax marker on zoomed
    if crop_z is not None:
        xh_z, yh_z = _transform_cropped(hotspot, "axial", vs, crop_z)
    else:
        xh_z, yh_z = _transform_point(hotspot, "axial", vs)
    _draw_marker(ax_z, (xh_z, yh_z), "red", marker="*", size=180, zorder=12)
    ax_z.annotate(f"SUVmax={suv_max:.2f}",
                  xy=(xh_z, yh_z), xytext=(xh_z + 5, yh_z + 5),
                  color="white", fontsize=8,
                  bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))

    # Legend
    handles = [
        mpatches.Patch(edgecolor="cyan", facecolor="none", label="Primary boundary"),
        mpatches.Patch(edgecolor="yellow", facecolor="none", linestyle="--",
                       label=f"MTV threshold ({mtv_threshold:.2f} SUV)"),
        plt.scatter([], [], c="red", marker="*", s=100,
                    label=f"SUVmax ({suv_max:.2f})"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7,
               facecolor="black", labelcolor="white", framealpha=0.7)

    fig.suptitle(
        f"gETU QC — MTV={mtv_cc:.1f}cc  threshold={mtv_threshold:.2f}  "
        f"gETU_index={getu_index:.3f}  gETU_raw={getu_raw:.3f}",
        color="white", fontsize=9, y=1.01
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1.0])
    out_path = Path(op) / "gETU_QC.png"
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    plt.close(fig)
    logger.info(f"gETU_QC.png saved: {out_path}")


# ---------------------------------------------------------------------------
# Section 7: Main metric QC entry point
# ---------------------------------------------------------------------------

def generate_metric_qc_overlays(
    pet_nifti_path,
    ct_nifti_path,
    tumor_mask_path,
    output_dir,
    metrics,
    dpi=150,
):
    """Generate all metric QC overlay images.

    Produces: Dmax_QC.png, NHOPmax_QC.png, NHOCmax_QC.png, gETU_QC.png

    Parameters
    ----------
    pet_nifti_path : str
        Path to PET NIfTI file
    ct_nifti_path : str
        Path to CT NIfTI file
    tumor_mask_path : str
        Path to tumor mask NIfTI file
    output_dir : str
        Directory to save QC images
    metrics : dict
        Metrics dict from advanced_metrics.compute_all_metrics()
        Must contain 'qc_coords' key.
    dpi : int
        DPI for output images
    """
    logger.info("Generating metric QC overlays...")

    # Check qc_coords exists
    if "qc_coords" not in metrics or not metrics["qc_coords"]:
        logger.error("metrics['qc_coords'] missing — cannot generate metric QC overlays")
        return

    qc = metrics["qc_coords"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load NIfTI files
    logger.info("Loading NIfTI files for metric QC...")
    ct_nii = nib.load(str(ct_nifti_path))
    pet_nii = nib.load(str(pet_nifti_path))
    mask_nii = nib.load(str(tumor_mask_path))

    ct_data = ct_nii.get_fdata().astype(np.float32)
    pet_data = pet_nii.get_fdata().astype(np.float32)
    tumor_mask = (mask_nii.get_fdata() > 0.5)
    spacing = _voxel_spacing_from_affine(ct_nii.affine)

    # Resample PET and mask to CT space if shapes differ
    if pet_data.shape != ct_data.shape:
        from scipy.ndimage import zoom as spzoom
        pet_zoom = tuple(ct_data.shape[i] / pet_data.shape[i] for i in range(3))
        pet_data = spzoom(pet_data, pet_zoom, order=1)
    if tumor_mask.shape != ct_data.shape:
        from scipy.ndimage import zoom as spzoom
        mz = tuple(ct_data.shape[i] / tumor_mask.shape[i] for i in range(3))
        tumor_mask = spzoom(tumor_mask.astype(np.float32), mz, order=0) > 0.5

    if tumor_mask.sum() == 0:
        logger.error("Empty tumor mask — cannot generate metric QC overlays")
        return

    # PET display max
    pvmax = float(pet_data[tumor_mask].max()) * 1.1
    pvmax = max(pvmax, 1.0)

    # Isolate primary mask ONCE
    logger.info("Isolating primary tumor for metric QC...")
    primary = _isolate_primary_for_qc(tumor_mask, pet_data, ct_data, spacing)
    logger.info(f"Primary mask: {primary.sum():,} voxels")

    # Dispatch to each metric QC generator
    generators = [
        (_generate_getu_qc,  "gETU_QC.png"),
        (_generate_nhop_qc,  "NHOPmax_QC.png"),
        (_generate_nhoc_qc,  "NHOCmax_QC.png"),
        (_generate_dmax_qc,  "Dmax_QC.png"),
    ]

    for gen_fn, fname in generators:
        try:
            gen_fn(
                pet=pet_data,
                ct=ct_data,
                tumor=tumor_mask,
                spacing=spacing,
                qc=qc,
                met=metrics,
                op=str(output_dir),
                dpi=dpi,
                pvmax=pvmax,
                primary=primary,
            )
        except Exception as e:
            logger.error(f"Failed to generate {fname}: {e}", exc_info=True)

    logger.info("Metric QC overlays complete.")
