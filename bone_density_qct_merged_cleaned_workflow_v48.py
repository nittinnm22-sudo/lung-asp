"""
bone_density_qct_merged_cleaned_workflow_v48.py
Bone Density QCT Analysis Workflow v48

Computes:
  - Volumetric BMD (vBMD) from axial CT ROIs using QCT calibration
  - Pseudo-DXA areal BMD (aBMD) from AP projection of CT volume
  - T-scores and Z-scores for both modalities
"""

import os
import sys
import logging
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── QCT Calibration ────────────────────────────────────────────────────────
QCT_SPINE_SLOPE = 0.9        # mg/cm³ per HU
QCT_SPINE_INTERCEPT = -10.0  # mg/cm³
QCT_FN_SLOPE = 0.9
QCT_FN_INTERCEPT = -10.0

# ── Axial CT QCT Norms (vBMD mg/cm³) ──────────────────────────────────────
QCT_SPINE_NORM_MEAN = 180.0  # young adult spine
QCT_SPINE_NORM_SD = 30.0
QCT_FN_NORM_MEAN = 220.0    # young adult femoral neck
QCT_FN_NORM_SD = 35.0

# ── DXA Norms (aBMD mg/cm²) ────────────────────────────────────────────────
DXA_SPINE_NORM_MEAN = 1000.0  # young adult L1-L4
DXA_SPINE_NORM_SD = 120.0
DXA_FN_NORM_MEAN = 850.0     # young adult femoral neck
DXA_FN_NORM_SD = 120.0

# ── DXA Projection Calibration ─────────────────────────────────────────────
# For mean bone HU ~300: aBMD = 300 * 3.0 + 100 = 1000 mg/cm² (normal spine)
PROJ_SPINE_SLOPE = 3.0
PROJ_SPINE_INTERCEPT = 100.0
PROJ_FN_SLOPE = 3.0
PROJ_FN_INTERCEPT = 50.0

# Bone threshold for projection (HU)
BONE_THRESHOLD = 150.0

SPINE_TAGS = {"L1", "L2", "L3", "L4", "L5"}
FN_TAGS = {"FN_R", "FN_L", "FEMORAL_NECK_R", "FEMORAL_NECK_L"}


# ─────────────────────────────────────────────────────────────────────────────
# ROI class
# ─────────────────────────────────────────────────────────────────────────────

class ROI:
    """Represents a 3D spherical ROI placed in the CT volume."""

    def __init__(
        self,
        tag: str,
        center_ijk,
        radius_mm: float,
        voxel_spacing=(1.0, 1.0, 1.0),
    ):
        """
        Parameters
        ----------
        tag          : str   – label, e.g. 'L1', 'FN_R'
        center_ijk   : (x, y, z) in *voxel* coordinates
        radius_mm    : float – sphere radius in mm
        voxel_spacing: (dx, dy, dz) in mm
        """
        self.tag = tag
        self.center_ijk = np.array(center_ijk, dtype=float)  # (x, y, z)
        self.radius_mm = float(radius_mm)
        self.voxel_spacing = tuple(voxel_spacing)
        self.canvas = None  # optional GUI canvas handle

    # ------------------------------------------------------------------
    @property
    def radius_voxels(self) -> float:
        return self.radius_mm / float(np.mean(self.voxel_spacing))

    # ------------------------------------------------------------------
    def get_mask(self, volume_shape) -> np.ndarray:
        """Return boolean 3-D mask (nz, ny, nx) for this ROI sphere."""
        dx, dy, dz = self.voxel_spacing
        cx, cy, cz = self.center_ijk
        nz, ny, nx = volume_shape

        zz, yy, xx = np.mgrid[0:nz, 0:ny, 0:nx]
        dist_sq = (
            ((xx - cx) * dx) ** 2
            + ((yy - cy) * dy) ** 2
            + ((zz - cz) * dz) ** 2
        )
        return dist_sq <= self.radius_mm ** 2

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"ROI(tag={self.tag!r}, center={self.center_ijk.tolist()}, "
            f"radius_mm={self.radius_mm})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# BoneDensityQCT class
# ─────────────────────────────────────────────────────────────────────────────

class BoneDensityQCT:
    """
    Bone Density QCT Analysis Workflow.

    Workflow
    --------
    1. Load CT volume (DICOM directory or NIfTI file).
    2. Add ROIs (add_roi) or load from JSON.
    3. Call run() to compute axial vBMD and pseudo-DXA aBMD.
    4. Retrieve results via get_results_summary().
    """

    def __init__(self):
        self.ct_volume: np.ndarray | None = None   # (nz, ny, nx) HU
        self.voxel_spacing = (1.0, 1.0, 1.0)       # (dx, dy, dz) mm
        self.rois: dict[str, ROI] = {}

        # ── QCT calibration (axial) ──
        self.qct_spine_slope = QCT_SPINE_SLOPE
        self.qct_spine_intercept = QCT_SPINE_INTERCEPT
        self.qct_fn_slope = QCT_FN_SLOPE
        self.qct_fn_intercept = QCT_FN_INTERCEPT

        # ── Axial QCT norms ──
        self.qct_spine_norm_mean = QCT_SPINE_NORM_MEAN
        self.qct_spine_norm_sd = QCT_SPINE_NORM_SD
        self.qct_fn_norm_mean = QCT_FN_NORM_MEAN
        self.qct_fn_norm_sd = QCT_FN_NORM_SD

        # ── DXA projection calibration ──
        self.proj_spine_slope = PROJ_SPINE_SLOPE
        self.proj_spine_intercept = PROJ_SPINE_INTERCEPT
        self.proj_fn_slope = PROJ_FN_SLOPE
        self.proj_fn_intercept = PROJ_FN_INTERCEPT

        # ── DXA norms ──
        self.dxa_spine_norm_mean = DXA_SPINE_NORM_MEAN
        self.dxa_spine_norm_sd = DXA_SPINE_NORM_SD
        self.dxa_fn_norm_mean = DXA_FN_NORM_MEAN
        self.dxa_fn_norm_sd = DXA_FN_NORM_SD

        # Bone threshold for projection (HU)
        self.bone_threshold = BONE_THRESHOLD

        # Internal caches / result stores
        self._projection_cache: dict = {}
        self.axial_results: dict = {}
        self.dxa_results: dict = {}

        logger.info("BoneDensityQCT initialised")

    # ──────────────────────────────────────────────────────────────────
    # Data loading
    # ──────────────────────────────────────────────────────────────────

    def load_dicom(self, dicom_dir: str) -> None:
        """Load a DICOM series from *dicom_dir* and build the 3-D CT volume."""
        try:
            import pydicom
        except ImportError as exc:
            raise ImportError("pydicom is required: pip install pydicom") from exc

        dicom_dir = Path(dicom_dir)
        if not dicom_dir.is_dir():
            raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")

        dcm_files = sorted(dicom_dir.glob("*.dcm"))
        if not dcm_files:
            # Try without extension
            dcm_files = sorted(
                p for p in dicom_dir.iterdir() if p.is_file()
            )

        if not dcm_files:
            raise FileNotFoundError(f"No DICOM files found in {dicom_dir}")

        datasets = []
        for f in dcm_files:
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=False)
                if hasattr(ds, "pixel_array"):
                    datasets.append(ds)
            except Exception:
                pass

        if not datasets:
            raise ValueError("No readable DICOM slices found.")

        # Sort by ImagePositionPatient z
        def _z_pos(ds):
            try:
                return float(ds.ImagePositionPatient[2])
            except Exception:
                return float(getattr(ds, "InstanceNumber", 0))

        datasets.sort(key=_z_pos)

        # Extract pixel data and apply rescale
        slices = []
        for ds in datasets:
            arr = ds.pixel_array.astype(np.float32)
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            slices.append(arr * slope + intercept)

        self.ct_volume = np.stack(slices, axis=0)  # (nz, ny, nx)

        # Voxel spacing
        try:
            px_spacing = datasets[0].PixelSpacing
            dx = float(px_spacing[1])
            dy = float(px_spacing[0])
        except Exception:
            dx = dy = 1.0

        try:
            positions = [_z_pos(ds) for ds in datasets]
            if len(positions) > 1:
                dz = abs(positions[1] - positions[0])
            else:
                dz = float(getattr(datasets[0], "SliceThickness", 1.0))
        except Exception:
            dz = 1.0

        self.voxel_spacing = (dx, dy, dz)
        self._projection_cache.clear()
        logger.info(
            "Loaded %d DICOM slices; volume shape=%s; spacing=%s",
            len(slices),
            self.ct_volume.shape,
            self.voxel_spacing,
        )

    def load_nifti(self, nifti_path: str) -> None:
        """Load a NIfTI file as the CT volume."""
        try:
            import nibabel as nib
        except ImportError as exc:
            raise ImportError("nibabel is required: pip install nibabel") from exc

        img = nib.load(str(nifti_path))
        data = np.asarray(img.dataobj, dtype=np.float32)

        # NIfTI stores (x, y, z); we need (z, y, x)
        if data.ndim == 3:
            data = np.transpose(data, (2, 1, 0))
        elif data.ndim == 4:
            data = np.transpose(data[..., 0], (2, 1, 0))
        else:
            raise ValueError(f"Unexpected NIfTI ndim={data.ndim}")

        self.ct_volume = data
        zooms = img.header.get_zooms()
        self.voxel_spacing = (
            float(zooms[0]),
            float(zooms[1]),
            float(zooms[2]) if len(zooms) > 2 else 1.0,
        )
        self._projection_cache.clear()
        logger.info(
            "Loaded NIfTI volume shape=%s; spacing=%s",
            self.ct_volume.shape,
            self.voxel_spacing,
        )

    # ──────────────────────────────────────────────────────────────────
    # ROI management
    # ──────────────────────────────────────────────────────────────────

    def add_roi(self, tag: str, center_ijk, radius_mm: float) -> ROI:
        """Add a spherical ROI to the analysis."""
        roi = ROI(tag, center_ijk, radius_mm, self.voxel_spacing)
        self.rois[tag] = roi
        logger.debug("Added ROI: %s", roi)
        return roi

    def remove_roi(self, tag: str) -> None:
        """Remove an ROI by tag."""
        self.rois.pop(tag, None)

    def _find_roi_by_tag(self, tag: str):
        """Return (canvas, roi) tuple, or (None, None) if not found."""
        if tag in self.rois:
            roi = self.rois[tag]
            canvas = roi.canvas
            return canvas, roi
        return None, None

    def load_rois_from_json(self, json_path: str) -> None:
        """Load ROI definitions from a JSON file.

        Expected format::

            [
              {"tag": "L1", "center": [x, y, z], "radius_mm": 15.0},
              ...
            ]
        """
        with open(json_path) as fh:
            data = json.load(fh)
        for entry in data:
            self.add_roi(
                tag=entry["tag"],
                center_ijk=entry["center"],
                radius_mm=entry["radius_mm"],
            )
        logger.info("Loaded %d ROIs from %s", len(data), json_path)

    # ──────────────────────────────────────────────────────────────────
    # Axial CT / QCT processing
    # ──────────────────────────────────────────────────────────────────

    def _compute_qct_roi_stats(self, tag: str) -> dict:
        """Compute vBMD and scores for the axial CT ROI identified by *tag*.

        Returns
        -------
        dict with keys: tag, mean_hu, std_hu, voxel_count, vbmd, t_score,
        z_score, z_score_note.

        .. warning::
            ``z_score`` is a *placeholder* equal to ``t_score``.
            Proper age-matched Z-scores require population reference data
            that has not yet been integrated.
        """
        _, roi = self._find_roi_by_tag(tag)
        if roi is None:
            raise KeyError(f"ROI tag '{tag}' not found")
        if self.ct_volume is None:
            raise RuntimeError("CT volume not loaded")

        mask = roi.get_mask(self.ct_volume.shape)
        voxels = self.ct_volume[mask]

        if voxels.size == 0:
            logger.warning("ROI '%s' mask is empty", tag)
            return {
                "tag": tag,
                "mean_hu": np.nan,
                "std_hu": np.nan,
                "voxel_count": 0,
                "vbmd": np.nan,
                "t_score": np.nan,
                "z_score": np.nan,
            }

        mean_hu = float(np.mean(voxels))
        std_hu = float(np.std(voxels))
        voxel_count = int(voxels.size)

        is_spine = tag.upper() in SPINE_TAGS
        if is_spine:
            slope = self.qct_spine_slope
            intercept = self.qct_spine_intercept
            norm_mean = self.qct_spine_norm_mean
            norm_sd = self.qct_spine_norm_sd
        else:
            slope = self.qct_fn_slope
            intercept = self.qct_fn_intercept
            norm_mean = self.qct_fn_norm_mean
            norm_sd = self.qct_fn_norm_sd

        raw_vbmd = mean_hu * slope + intercept
        if raw_vbmd < 0.0:
            logger.warning(
                "ROI '%s': raw vBMD=%.1f mg/cm³ is negative (possible calibration/data issue); clamping to 0.",
                tag, raw_vbmd,
            )
        vbmd = max(0.0, raw_vbmd)
        t_score = (vbmd - norm_mean) / norm_sd
        # NOTE: Z-score requires age-matched reference data which is not yet
        # implemented.  The value below equals the T-score and is a placeholder.
        z_score = t_score

        return {
            "tag": tag,
            "mean_hu": mean_hu,
            "std_hu": std_hu,
            "voxel_count": voxel_count,
            "vbmd": vbmd,
            "t_score": t_score,
            "z_score": z_score,
            "z_score_note": "placeholder – equals T-score (age-matched norms not implemented)",
        }

    def _compute_axial_results(self) -> dict:
        """Process all registered ROIs for axial QCT measurements."""
        results = {}
        for tag in list(self.rois.keys()):
            try:
                results[tag] = self._compute_qct_roi_stats(tag)
            except Exception as exc:
                logger.error("Axial QCT failed for ROI '%s': %s", tag, exc)
                results[tag] = {"tag": tag, "error": str(exc)}
        self.axial_results = results
        return results

    # ──────────────────────────────────────────────────────────────────
    # DXA projection processing
    # ──────────────────────────────────────────────────────────────────

    def _make_ap_projection(self, z_start: int, z_end: int) -> np.ndarray:
        """Create an anterior-posterior projection of the CT sub-volume.

        Projects along the y-axis (anterior-posterior direction).
        Only voxels above *bone_threshold* contribute to the mean.

        Parameters
        ----------
        z_start, z_end : int – slice indices (z_end is exclusive)

        Returns
        -------
        2-D float32 array of shape (nz, nx) containing mean bone HU
        along each (z, x) ray, or 0 where no bone voxels exist.
        """
        if self.ct_volume is None:
            raise RuntimeError("CT volume not loaded")

        z_start = max(0, z_start)
        z_end = min(self.ct_volume.shape[0], z_end)

        sub_vol = self.ct_volume[z_start:z_end, :, :]  # (nz, ny, nx)

        bone_mask = sub_vol > self.bone_threshold           # (nz, ny, nx)
        bone_count = bone_mask.sum(axis=1).astype(np.float32)  # (nz, nx)
        bone_hu_sum = (sub_vol * bone_mask).sum(axis=1).astype(np.float32)

        valid = bone_count > 0
        proj = np.zeros_like(bone_count, dtype=np.float32)
        proj[valid] = bone_hu_sum[valid] / bone_count[valid]
        return proj

    def _get_cached_projection(self, z_start: int, z_end: int) -> np.ndarray:
        """Return a cached AP projection or compute and cache it."""
        key = (z_start, z_end)
        if key not in self._projection_cache:
            self._projection_cache[key] = self._make_ap_projection(z_start, z_end)
        return self._projection_cache[key]

    def _ensure_dxa_projection_results(self) -> dict:
        """Compute pseudo-DXA aBMD for every registered ROI.

        For each ROI:
          1. Determine z-range from the ROI sphere bounds.
          2. Build an AP projection of that z-range.
          3. Sample the projection within a circle at (z_center, x_center).
          4. Convert mean projected HU → pseudo-aBMD using calibration.
          5. Compute T-score / Z-score against DXA norms.

        Returns
        -------
        dict: tag -> result dict

        .. warning::
            ``z_score`` is a *placeholder* equal to ``t_score``.
            Proper age-matched Z-scores require population reference data
            that has not yet been integrated.
        """
        if self.ct_volume is None:
            raise RuntimeError("CT volume not loaded")

        results = {}
        nz, ny, nx = self.ct_volume.shape
        dx, dy, dz = self.voxel_spacing

        for tag, roi in self.rois.items():
            try:
                cx, cy, cz = roi.center_ijk  # voxel coordinates

                # z-range for this ROI
                r_z_vox = roi.radius_mm / dz
                z_start = int(max(0, np.floor(cz - r_z_vox)))
                z_end = int(min(nz, np.ceil(cz + r_z_vox) + 1))

                if z_end <= z_start:
                    raise ValueError(
                        f"Empty z-range for ROI '{tag}': z_start={z_start}, z_end={z_end}"
                    )

                proj = self._get_cached_projection(z_start, z_end)
                # proj shape: (z_end - z_start, nx)

                # ROI center in projection coordinates
                cz_proj = cz - z_start      # row in projection image
                cx_proj = cx                # column in projection image

                # Sample radius in projection voxels
                r_z_proj = roi.radius_mm / dz
                r_x_proj = roi.radius_mm / dx

                # Build coordinate grids for projection sub-image
                proj_nz, proj_nx = proj.shape
                zp, xp = np.mgrid[0:proj_nz, 0:proj_nx]

                # Elliptical mask in projection space
                dist_sq = (
                    ((zp - cz_proj) / max(r_z_proj, 1e-6)) ** 2
                    + ((xp - cx_proj) / max(r_x_proj, 1e-6)) ** 2
                )
                sample_mask = dist_sq <= 1.0

                sampled = proj[sample_mask]
                nonzero = sampled[sampled > 0]

                if nonzero.size == 0:
                    mean_proj_hu = 0.0
                else:
                    mean_proj_hu = float(np.mean(nonzero))

                is_spine = tag.upper() in SPINE_TAGS
                if is_spine:
                    slope = self.proj_spine_slope
                    intercept = self.proj_spine_intercept
                    dxa_mean = self.dxa_spine_norm_mean
                    dxa_sd = self.dxa_spine_norm_sd
                else:
                    slope = self.proj_fn_slope
                    intercept = self.proj_fn_intercept
                    dxa_mean = self.dxa_fn_norm_mean
                    dxa_sd = self.dxa_fn_norm_sd

                pseudo_abmd = max(0.0, mean_proj_hu * slope + intercept)
                t_score = (pseudo_abmd - dxa_mean) / dxa_sd
                # NOTE: Z-score is a placeholder equal to T-score.
                # Age-matched DXA norms have not been integrated.
                z_score = t_score

                results[tag] = {
                    "tag": tag,
                    "mean_proj_hu": mean_proj_hu,
                    "pseudo_abmd": pseudo_abmd,
                    "t_score": t_score,
                    "z_score": z_score,
                    "z_score_note": "placeholder – equals T-score (age-matched norms not implemented)",
                    "sampled_voxels": int(nonzero.size),
                }

            except Exception as exc:
                logger.error("DXA projection failed for ROI '%s': %s", tag, exc)
                results[tag] = {"tag": tag, "error": str(exc)}

        self.dxa_results = results
        return results

    # ──────────────────────────────────────────────────────────────────
    # Full workflow
    # ──────────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """Run the full bone density analysis pipeline.

        Returns
        -------
        dict with keys 'axial' (vBMD) and 'dxa' (pseudo-aBMD) results.
        """
        if self.ct_volume is None:
            raise RuntimeError("CT volume not loaded. Call load_dicom() or load_nifti() first.")
        if not self.rois:
            raise RuntimeError("No ROIs registered. Call add_roi() or load_rois_from_json() first.")

        logger.info("Running axial QCT analysis on %d ROIs …", len(self.rois))
        axial = self._compute_axial_results()

        logger.info("Running pseudo-DXA projection analysis …")
        dxa = self._ensure_dxa_projection_results()

        return {"axial": axial, "dxa": dxa}

    def get_results_summary(self) -> str:
        """Return a human-readable text summary of all results."""
        lines = ["=" * 60, "  Bone Density QCT Results Summary", "=" * 60]

        if self.axial_results:
            lines.append("\nAxial CT / QCT (vBMD mg/cm³)")
            lines.append("-" * 40)
            for tag, r in self.axial_results.items():
                if "error" in r:
                    lines.append(f"  {tag:12s}: ERROR – {r['error']}")
                else:
                    lines.append(
                        f"  {tag:12s}: mean HU={r['mean_hu']:7.1f}  "
                        f"vBMD={r['vbmd']:7.1f} mg/cm³  "
                        f"T={r['t_score']:+.2f}"
                    )

        if self.dxa_results:
            lines.append("\nPseudo-DXA Projection (aBMD mg/cm²)")
            lines.append("-" * 40)
            for tag, r in self.dxa_results.items():
                if "error" in r:
                    lines.append(f"  {tag:12s}: ERROR – {r['error']}")
                else:
                    lines.append(
                        f"  {tag:12s}: proj HU={r['mean_proj_hu']:7.1f}  "
                        f"aBMD={r['pseudo_abmd']:7.1f} mg/cm²  "
                        f"T={r['t_score']:+.2f}"
                    )

        lines.append("=" * 60)
        return "\n".join(lines)

    def export_results_json(self, output_path: str) -> None:
        """Export axial + DXA results to a JSON file."""
        payload = {
            "axial": self.axial_results,
            "dxa": self.dxa_results,
        }
        with open(output_path, "w") as fh:
            json.dump(payload, fh, indent=2, default=_json_default)
        logger.info("Results exported to %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _json_default(obj):
    """JSON serialiser for numpy scalars."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


def _build_synthetic_demo() -> BoneDensityQCT:
    """Build a small synthetic CT volume for demo/testing purposes."""
    qct = BoneDensityQCT()
    vol = np.full((40, 60, 40), -300.0, dtype=np.float32)  # soft tissue background

    # Simulate two lumbar vertebral bodies
    vol[5:18, 20:40, 10:30] = 180.0   # L1 trabecular
    vol[22:35, 20:40, 10:30] = 160.0  # L2 trabecular

    # Cortical shell (brighter)
    vol[5:18, 20:22, 10:30] = 800.0
    vol[5:18, 38:40, 10:30] = 800.0
    vol[22:35, 20:22, 10:30] = 800.0
    vol[22:35, 38:40, 10:30] = 800.0

    qct.ct_volume = vol
    qct.voxel_spacing = (0.9375, 0.9375, 3.0)

    qct.add_roi("L1", center_ijk=(20.0, 30.0, 11.0), radius_mm=12.0)
    qct.add_roi("L2", center_ijk=(20.0, 30.0, 28.0), radius_mm=12.0)
    return qct


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Bone Density QCT Analysis v48",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # -- analyse subcommand --
    analyse = sub.add_parser("analyse", help="Analyse a CT volume")
    analyse.add_argument(
        "--dicom-dir", metavar="DIR", help="Path to DICOM series directory"
    )
    analyse.add_argument(
        "--nifti", metavar="FILE", help="Path to NIfTI file (.nii / .nii.gz)"
    )
    analyse.add_argument(
        "--rois", metavar="JSON", help="Path to ROI definitions JSON file"
    )
    analyse.add_argument(
        "--output", metavar="FILE", default="results.json",
        help="Output JSON file for results",
    )

    # -- demo subcommand --
    sub.add_parser("demo", help="Run on a synthetic demo volume")

    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    if args.command == "demo" or args.command is None:
        logger.info("Running synthetic demo …")
        qct = _build_synthetic_demo()
        results = qct.run()
        print(qct.get_results_summary())
        return 0

    if args.command == "analyse":
        qct = BoneDensityQCT()

        if args.dicom_dir:
            qct.load_dicom(args.dicom_dir)
        elif args.nifti:
            qct.load_nifti(args.nifti)
        else:
            logger.error("Specify either --dicom-dir or --nifti")
            return 1

        if args.rois:
            qct.load_rois_from_json(args.rois)
        else:
            logger.warning(
                "No ROI file provided; using demo ROIs centred on volume middle"
            )
            nz, ny, nx = qct.ct_volume.shape
            qct.add_roi("L1", center_ijk=(nx / 2, ny / 2, nz * 0.3), radius_mm=15.0)
            qct.add_roi("L2", center_ijk=(nx / 2, ny / 2, nz * 0.5), radius_mm=15.0)

        results = qct.run()
        print(qct.get_results_summary())
        qct.export_results_json(args.output)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
