#!/usr/bin/env python3
# file: bone_density_qct_merged.py
# Final integrated build: publication UI + editable DXA/AP + TotalSegmentator auto-placement
# Merged from bone_density_qct_autoSEG.py + bone_density_qct_autoSEG1.py
# Build: AutoSEG with TotalSegmentator for L1–L4 + bilateral femoral necks
# Corrected: all module-scope methods moved into class, extra_masks init, debug_roi_hu fix,
#            auto-measure after placement, closeEvent properly bound.
# FIXES APPLIED:
#   1. Removed duplicate _find_roi_by_tag; unified to tuple-returning version; all callers updated.
#   2. _analyze_roi now uses ROI's stored slice_index instead of canvas.current_slice.
#   3. _make_ap_projection uses bone HU thresholds and fixes image orientation.
#   4. _compute_qc_for_site and overlay code properly unpack (canvas, roi) tuples.

import os, sys, csv, math, tempfile, subprocess, zipfile, shutil, re
from typing import Optional, Tuple, Dict, List
from datetime import datetime
from glob import glob
import numpy as np

# Optional deps
try:
    from scipy.ndimage import (
        rotate as ndi_rotate,
        zoom as ndi_zoom,
        binary_erosion,
        distance_transform_edt,
        label as ndi_label,
    )
except Exception:
    ndi_rotate = None
    ndi_zoom = None
    binary_erosion = None
    ndi_label = None

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except Exception:
    sitk = None
    SITK_AVAILABLE = False

try:
    import nibabel as nib
    HAVE_NIB = True
except Exception:
    nib = None
    HAVE_NIB = False

# TotalSegmentator detection
TOTALSEG_AVAILABLE = False
TOTALSEG_API = False
try:
    from totalsegmentator.python_api import totalseg
    TOTALSEG_AVAILABLE = True
    TOTALSEG_API = True
except Exception:
    try:
        import totalsegmentator  # noqa: F401
        TOTALSEG_AVAILABLE = True
    except Exception:
        TOTALSEG_AVAILABLE = False

# Optional DICOM header reader
try:
    import pydicom
    PYDICOM_AVAILABLE = True
except Exception:
    pydicom = None
    PYDICOM_AVAILABLE = False

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSlider, QFileDialog, QMessageBox, QRadioButton, QTextEdit,
    QSpinBox, QCheckBox, QDoubleSpinBox, QGroupBox, QFormLayout, QComboBox,
    QDialog, QTableWidget, QTableWidgetItem, QHeaderView, QToolButton,
    QSizePolicy, QProgressBar, QTabWidget, QScrollArea, QGridLayout,
)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt6.QtCore import (
    Qt, QPoint, QRect, QRectF, pyqtSignal, QPointF, QThread, QTimer, QSize,
)

# --- Debug logging ---
import logging

_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qct_app.log")
logging.basicConfig(
    filename=_LOG_PATH,
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logging.debug("Module import start.")


# ================================================================
# Norms (for T/Z-score computation)
# ================================================================
def young_adult_norms(sex: str) -> Tuple[float, float]:
    if (sex or "F").upper().startswith("F"):
        return 160.0, 28.0
    return 170.0, 32.0


def age_matched_norms(age: int, sex: str) -> Tuple[float, float]:
    age = int(max(20, min(90, age)))
    if (sex or "F").upper().startswith("F"):
        mean0, sd0, slope = 160.0, 28.0, 0.9
    else:
        mean0, sd0, slope = 170.0, 32.0, 0.8
    mean = max(80.0, mean0 - slope * max(0, age - 30))
    return float(mean), float(sd0)


# ================================================================
# ROI
# ================================================================
class SphericalROI:
    """Simple circular ROI in slice pixel coordinates."""

    def __init__(self, cx, cy, r=20, tag: str = "ROI", slice_index: int = None):
        self.center = QPointF(float(cx), float(cy))
        self.radius = int(r)
        self.tag = str(tag) if tag is not None else "ROI"
        self.slice_index = None if slice_index is None else int(slice_index)
        self.locked_to_slice = True
        self.follow_on_scroll = False

    def contains(self, pt):
        return np.hypot(pt[0] - self.center.x(), pt[1] - self.center.y()) <= self.radius

    def move_to(self, pt):
        self.center = pt

    def resize(self, d):
        self.radius = max(5, int(self.radius + d))

    def copy(self):
        roi = SphericalROI(
            self.center.x(),
            self.center.y(),
            self.radius,
            tag=self.tag,
            slice_index=self.slice_index,
        )
        roi.locked_to_slice = bool(getattr(self, "locked_to_slice", True))
        roi.follow_on_scroll = bool(getattr(self, "follow_on_scroll", False))
        return roi


# ================================================================
# Canvas
# ================================================================
class ImageCanvas(QLabel):
    roi_changed = pyqtSignal()
    crosshair_moved = pyqtSignal(int, int, int)
    slice_scrolled = pyqtSignal(int)

    def __init__(self, view="axial", parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.view = view
        self.main = parent
        self.current_slice = 0
        self.crosshair_pos = QPointF(0, 0)
        self.zoom_factor = 1.0
        self.pan_offset = QPointF(0, 0)
        self.rotation_angle = 0
        self.flip_h = False
        self.flip_v = False
        self.is_panning = False
        self.dragging_roi_move = False
        self.dragging_roi_resize = False
        self.roi_resize_start_dist = 0.0
        self.last_pan_point = QPointF(0, 0)
        self.spherical_rois: List[SphericalROI] = []
        self.selected_roi: Optional[SphericalROI] = None
        self.hover_roi: Optional[SphericalROI] = None
        self.follow_selected_across_slices = True
        self.base_pixmap = QPixmap()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(260, 220)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("border: 2px solid #333; background-color: black;")
        self.setMouseTracking(True)
        self.hot_colormap = self._hot_colormap()

    # ---- colour helpers ----
    def _hot_colormap(self) -> np.ndarray:
        x = np.linspace(0.0, 1.0, 256)
        r = np.clip(3 * x, 0, 1)
        g = np.clip(3 * x - 1, 0, 1)
        b = np.clip(3 * x - 2, 0, 1)
        lut = np.stack([r, g, b], axis=1)
        return (lut * 255.0 + 0.5).astype(np.uint8)

    def _norm(self, arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        if arr is None:
            return None
        a = np.nan_to_num(arr, nan=vmin)
        lo, hi = (float(vmin), float(vmax)) if vmax > vmin else (float(vmin), float(vmin) + 1.0)
        a = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
        return (a * 255.0 + 0.5).astype(np.uint8)

    def _fuse(self, ct_slice: np.ndarray, pet_slice: np.ndarray) -> np.ndarray:
        c, w = self.main.ct_win_center, max(self.main.ct_win_width, 1.0)
        vmin, vmax = float(c - w / 2.0), float(c + w / 2.0)
        ct8 = self._norm(ct_slice, vmin, vmax)
        ct_rgb = np.repeat(ct8[..., None], 3, axis=2)
        pet = np.nan_to_num(pet_slice.astype(np.float32))
        if not np.isfinite(pet).any() or pet.size == 0:
            return ct_rgb
        low = np.percentile(pet, float(self.main.pet_low_pct))
        hi = np.percentile(pet, 99.5)
        if not np.isfinite(hi) or hi <= low:
            hi = low + 1.0
        petn = np.clip((pet - low) / (hi - low), 0.0, 1.0)
        idx = (petn * 255.0).astype(np.uint8)
        pet_rgb = self.hot_colormap[idx]
        a = np.clip(float(getattr(self.main, "pet_alpha", 0.4)) * petn, 0.0, 1.0)[..., None]
        fused = (1.0 - a) * ct_rgb.astype(np.float32) + a * pet_rgb.astype(np.float32)
        return np.clip(fused, 0, 255).astype(np.uint8)

    # ---- coordinate transforms ----
    def _img_to_widget(self, pt: QPointF, pan_zoom: bool = False) -> QPointF:
        if self.base_pixmap.isNull():
            return QPointF(self.width() / 2, self.height() / 2)
        W, H = self._slice_dims()
        rect = self._centered_rect(self.base_pixmap.size())
        uv = self._forward_map(pt.x(), pt.y())
        sx = rect.width() / max(W, 1)
        sy = rect.height() / max(H, 1)
        xw = rect.left() + uv.x() * sx
        yw = rect.top() + uv.y() * sy
        if not pan_zoom:
            return QPointF(xw, yw)
        z = float(self.zoom_factor)
        return QPointF(z * xw + self.pan_offset.x(), z * yw + self.pan_offset.y())

    def _widget_to_img(self, pos: QPointF) -> Optional[QPointF]:
        if self.base_pixmap.isNull():
            return None
        rect = self._centered_rect(self.base_pixmap.size())
        z = max(float(self.zoom_factor), 1e-6)
        x_local = (pos.x() - self.pan_offset.x()) / z
        y_local = (pos.y() - self.pan_offset.y()) / z
        if not rect.contains(int(x_local), int(y_local)):
            return None
        W, H = self._slice_dims()
        u_disp = (x_local - rect.left()) * max(W, 1) / max(rect.width(), 1)
        v_disp = (y_local - rect.top()) * max(H, 1) / max(rect.height(), 1)
        uv = self._inverse_map(u_disp, v_disp)
        return QPointF(uv.x(), uv.y())

    def _roi_at_widget_pos(self, wpos: QPointF):
        if not self.spherical_rois or self.base_pixmap.isNull():
            return None
        W, H = self._slice_dims()
        rect = self._centered_rect(self.base_pixmap.size())
        scale_x = rect.width() / max(W, 1)
        candidate_rois = []
        for r in self.spherical_rois:
            same_slice = getattr(r, "slice_index", None) in (None, self.current_slice)
            if same_slice or (r is self.selected_roi):
                candidate_rois.append(r)
        for roi in reversed(candidate_rois):
            c_w = self._img_to_widget(roi.center, pan_zoom=True)
            r_w = float(roi.radius) * scale_x * float(self.zoom_factor)
            dx = float(wpos.x() - c_w.x())
            dy = float(wpos.y() - c_w.y())
            if (dx * dx + dy * dy) <= (r_w * r_w):
                return roi
        return None

    # ---- slice / view ----
    def set_slice(self, idx: int):
        self.current_slice = int(idx)
        try:
            if self.selected_roi is not None:
                sl = getattr(self.selected_roi, "slice_index", None)
                if (sl is not None) and (int(sl) != int(self.current_slice)):
                    self.selected_roi = None
                    self.hover_roi = None
        except Exception:
            pass
        self.update_image()

    def flip(self, horizontal: bool):
        if horizontal:
            self.flip_h = not self.flip_h
        else:
            self.flip_v = not self.flip_v
        self.update_image()
        self.main._sync_crosshairs(*self.main.crosshair)

    def rotate(self, angle: int):
        self.rotation_angle = (self.rotation_angle + angle) % 360
        self.update_image()
        self.main._sync_crosshairs(*self.main.crosshair)

    def reset_view(self):
        self.zoom_factor = 1.0
        self.pan_offset = QPointF(0, 0)
        self.rotation_angle = 0
        self.flip_h = False
        self.flip_v = False
        self.update_image()
        self.main._sync_crosshairs(*self.main.crosshair)

    def get_slice_data(self, pet=False):
        data = (
            self.main.registered_pet
            if (pet and self.main.registered_pet is not None)
            else self.main.image_data
        )
        if data is None:
            return None
        if self.view == "axial":
            idx = int(np.clip(self.current_slice, 0, data.shape[2] - 1))
            s = data[:, :, idx].T
        elif self.view == "coronal":
            idx = int(np.clip(self.current_slice, 0, data.shape[1] - 1))
            s = data[:, idx, :].T
        else:  # sagittal
            idx = int(np.clip(self.current_slice, 0, data.shape[0] - 1))
            s = data[idx, :, :].T

        if self.view == "axial" and hasattr(self.main, "axial_flip_ud") and bool(self.main.axial_flip_ud):
            s = np.flipud(s)
        if self.view == "coronal" and hasattr(self.main, "coronal_flip_ud") and bool(self.main.coronal_flip_ud):
            s = np.flipud(s)
        if self.view == "sagittal" and hasattr(self.main, "sagittal_flip_ud") and bool(self.main.sagittal_flip_ud):
            s = np.flipud(s)
        if self.flip_h:
            s = np.fliplr(s)
        if self.flip_v:
            s = np.flipud(s)
        if self.rotation_angle != 0 and ndi_rotate is not None:
            s = ndi_rotate(
                s,
                self.rotation_angle,
                reshape=False,
                order=1,
                mode="constant",
                cval=np.min(s) if s.size > 0 else 0,
            )
        return s

    def update_image(self):
        slice_data = self.get_slice_data()
        if slice_data is None:
            self.base_pixmap = QPixmap()
            self.update()
            return
        is_rgb = False
        if self.main.fusion_mode == "Fusion" and self.main.registered_pet is not None:
            pet_slice = self.main._get_slice_for_canvas(self, use_pet=True)
            if pet_slice is not None:
                slice_data = self._fuse(slice_data, pet_slice)
                is_rgb = True

        if is_rgb:
            h, w, _ = slice_data.shape
            qimg = QImage(
                np.ascontiguousarray(slice_data).tobytes(),
                w, h, 3 * w,
                QImage.Format.Format_RGB888,
            )
        else:
            c, w_val = self.main.ct_win_center, max(self.main.ct_win_width, 1.0)
            vmin, vmax = float(c - w_val / 2.0), float(c + w_val / 2.0)
            norm = self._norm(slice_data, vmin, vmax)
            h, w = norm.shape
            qimg = QImage(norm.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)

        self.base_pixmap = QPixmap.fromImage(qimg)
        self.update()

    # ---- paint ----
    def paintEvent(self, _):
        if self.base_pixmap.isNull():
            return super().paintEvent(_)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self._centered_rect(self.base_pixmap.size())

        z = float(self.zoom_factor)
        target = QRectF(
            z * rect.left() + self.pan_offset.x(),
            z * rect.top() + self.pan_offset.y(),
            z * rect.width(),
            z * rect.height(),
        )
        src_rect = QRectF(self.base_pixmap.rect())
        p.drawPixmap(target, self.base_pixmap, src_rect)

        # QC overlay
        try:
            if getattr(self.main, "qc_overlay_enabled", False) and hasattr(
                self.main, "_get_qc_overlay_for_canvas"
            ):
                ov = self.main._get_qc_overlay_for_canvas(self)
                if isinstance(ov, dict) and ov.get("qimg") is not None:
                    p.drawImage(target, ov["qimg"], src_rect)
                    info = ov.get("info")
                    if info:
                        p.setFont(QFont("Consolas", 9))
                        metrics = p.fontMetrics()
                        tw = metrics.horizontalAdvance(info) + 12
                        th = metrics.height() + 8
                        p.fillRect(QRect(8, 8 + 28, tw, th), QColor(0, 0, 0, 150))
                        p.setPen(QPen(QColor(255, 255, 255), 1))
                        p.drawText(14, 8 + 28 + metrics.ascent() + 4, info)
        except Exception:
            pass

        W, H = self._slice_dims()
        scale_x = rect.width() / max(W, 1)
        scale_y = rect.height() / max(H, 1)
        vis_rois = [
            r for r in self.spherical_rois
            if getattr(r, "slice_index", None) in (None, self.current_slice)
        ]

        for roi in vis_rois:
            c_w = self._img_to_widget(roi.center, pan_zoom=True)
            avg_scale = (scale_x + scale_y) / 2.0
            r_w = float(roi.radius) * avg_scale * float(self.zoom_factor)

            if roi == self.selected_roi:
                pen = QPen(QColor(255, 255, 0), 2)
            elif roi == self.hover_roi:
                pen = QPen(QColor(0, 255, 255), 2)
            else:
                pen = QPen(QColor(255, 0, 0), 2)
            p.setPen(pen)
            p.drawEllipse(c_w, r_w, r_w)
            try:
                tag = getattr(roi, "tag", "")
                if tag:
                    p.setFont(QFont("Consolas", 9))
                    p.drawText(int(c_w.x() + 6), int(c_w.y() - 6), str(tag))
            except Exception:
                pass

        ch = self._img_to_widget(self.crosshair_pos, pan_zoom=True)
        p.setPen(QPen(QColor(0, 255, 0), 1, Qt.PenStyle.DotLine))
        p.drawLine(int(ch.x()), 0, int(ch.x()), self.height())
        p.drawLine(0, int(ch.y()), self.width(), int(ch.y()))

        # HUD
        try:
            p.setFont(QFont("Consolas", 9))
            hud = f"VIEW {self.view.upper()} | ROI {getattr(self.main, 'roi_mode', '-') or '-'} | SL {self.current_slice}"
            metrics = p.fontMetrics()
            tw = metrics.horizontalAdvance(hud) + 12
            th = metrics.height() + 8
            x0 = self.width() - tw - 8
            y0 = 8
            p.fillRect(QRect(x0, y0, tw, th), QColor(0, 0, 0, 150))
            p.setPen(QPen(QColor(255, 255, 255), 1))
            p.drawText(x0 + 6, y0 + metrics.ascent() + 4, hud)
        except Exception:
            pass

        if getattr(self.main, "show_overlay", True):
            x, y, zidx = self.main.crosshair
            hu = float("nan")
            if self.main.ct_data is not None:
                try:
                    hu = float(self.main.ct_data[x, y, zidx])
                except Exception:
                    pass
            sx, sy, sz = (
                self.main.ct_spacing
                if self.main.ct_spacing is not None
                else (1.0, 1.0, 1.0)
            )
            wl = f"W/L={self.main.ct_win_width:.0f}/{self.main.ct_win_center:.0f}"
            txt = f"{self.view.upper()} idx=[{x},{y},{zidx}] mm≈[{x*sx:.1f},{y*sy:.1f},{zidx*sz:.1f}] HU≈{hu:.1f} {wl}"
            p.setFont(QFont("Consolas", 9))
            metrics = p.fontMetrics()
            tw = metrics.horizontalAdvance(txt) + 12
            th = metrics.height() + 8
            p.fillRect(QRect(8, self.height() - th - 8, tw, th), QColor(0, 0, 0, 150))
            p.setPen(QPen(QColor(255, 255, 255), 1))
            p.drawText(14, self.height() - 8 - (th - metrics.ascent() - 4), txt)

        p.end()

    # ---- mouse events ----
    def mousePressEvent(self, e):
        self.setFocus()
        if hasattr(self.main, "_set_active_canvas"):
            self.main._set_active_canvas(self)
        img_pos = self._widget_to_img(e.position())
        inside = self._roi_at_widget_pos(e.position())

        if e.button() == Qt.MouseButton.LeftButton:
            if inside is not None:
                self.selected_roi = inside
                self.dragging_roi_move = True
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                return

            if (getattr(self.main, "roi_mode", None) == "Cross") or (
                e.modifiers() & Qt.KeyboardModifier.ControlModifier
            ):
                if img_pos is not None:
                    self._emit_crosshair(img_pos)
                    return

            require_alt = bool(getattr(self.main, "require_alt_for_new_roi", True))
            want_create = bool(
                e.modifiers()
                & (
                    Qt.KeyboardModifier.AltModifier
                    if require_alt
                    else Qt.KeyboardModifier.ShiftModifier
                )
            )
            if want_create and (img_pos is not None):
                self.selected_roi = SphericalROI(
                    img_pos.x(),
                    img_pos.y(),
                    20,
                    tag="Manual",
                    slice_index=int(self.current_slice),
                )
                self.spherical_rois.append(self.selected_roi)
                if hasattr(self.main, "set_roi_mode"):
                    self.main.set_roi_mode("Spherical")
                self.dragging_roi_move = True
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                self.update()
                return

            self.is_panning = True
            self.last_pan_point = e.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if (e.button() == Qt.MouseButton.RightButton) and (inside is not None):
            self.selected_roi = inside
            self.dragging_roi_resize = True
            if img_pos is not None:
                cx, cy = self.selected_roi.center.x(), self.selected_roi.center.y()
                self.roi_resize_start_dist = float(
                    np.hypot(img_pos.x() - cx, img_pos.y() - cy)
                )
            else:
                self.roi_resize_start_dist = 0.0
            self.setCursor(Qt.CursorShape.SizeAllCursor)

    def mouseMoveEvent(self, e):
        img_pos = self._widget_to_img(e.position())
        if not (self.is_panning or self.dragging_roi_move or self.dragging_roi_resize):
            prev = self.hover_roi
            self.hover_roi = self._roi_at_widget_pos(e.position())
            if prev is not self.hover_roi:
                self.update()
        if self.dragging_roi_move and self.selected_roi and (img_pos is not None):
            self.selected_roi.move_to(img_pos)
            self.update()
            return
        if self.dragging_roi_resize and self.selected_roi and (img_pos is not None):
            cx, cy = self.selected_roi.center.x(), self.selected_roi.center.y()
            dist = float(np.hypot(img_pos.x() - cx, img_pos.y() - cy))
            delta = int(round(dist - self.roi_resize_start_dist))
            if delta != 0:
                self.selected_roi.resize(delta)
                self.roi_resize_start_dist = dist
                self.update()
            return
        if self.is_panning:
            delta = e.position() - self.last_pan_point
            self.pan_offset += delta
            self.last_pan_point = e.position()
            self.update()
            return

    def mouseDoubleClickEvent(self, e):
        img_pos = self._widget_to_img(e.position())
        if img_pos is not None:
            self._emit_crosshair(img_pos)

    def mouseReleaseEvent(self, _e):
        self.is_panning = False
        self.dragging_roi_move = False
        self.dragging_roi_resize = False
        self.setCursor(
            Qt.CursorShape.OpenHandCursor
            if getattr(self.main, "roi_mode", None) == "Pan"
            else Qt.CursorShape.ArrowCursor
        )
        self.roi_changed.emit()

    def wheelEvent(self, e):
        d = e.angleDelta().y()
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom_at(1.1 if d > 0 else 1 / 1.1, e.position())
        elif self.selected_roi is not None:
            self.selected_roi.resize(2 if d > 0 else -2)
            self.update()
            self.roi_changed.emit()
        else:
            self.slice_scrolled.emit(1 if d > 0 else -1)

    def zoom_in(self):
        self.zoom_at(1.1, QPointF(self.width() / 2, self.height() / 2))

    def zoom_out(self):
        self.zoom_at(1 / 1.1, QPointF(self.width() / 2, self.height() / 2))

    def zoom_at(self, factor, posF: QPointF):
        before_img = self._widget_to_img(posF)
        self.zoom_factor = float(self.zoom_factor) * float(factor)
        if before_img is not None:
            unscaled = self._img_to_widget(before_img, pan_zoom=False)
            self.pan_offset = posF - QPointF(
                self.zoom_factor * unscaled.x(), self.zoom_factor * unscaled.y()
            )
        self.update()

    # ---- geometry ----
    def _centered_rect(self, size):
        if size.height() == 0:
            return QRect()
        ar = size.width() / size.height()
        wr = self.width() / max(self.height(), 1)
        if wr > ar:
            h = self.height()
            w = h * ar
        else:
            w = self.width()
            h = w / ar
        r = QRect(0, 0, int(w), int(h))
        r.moveCenter(self.rect().center())
        return r

    def _slice_dims(self):
        if self.main.image_data is None:
            return (1, 1)
        s = self.main.image_data.shape  # x,y,z
        if self.view == "axial":
            return (s[0], s[1])
        elif self.view == "coronal":
            return (s[0], s[2])
        else:
            return (s[1], s[2])

    def _forward_map(self, u, v):
        """Map canonical slice coordinates to displayed slice coordinates."""
        W, H = self._slice_dims()
        uu, vv = float(u), float(v)

        # View-specific vertical display flips used in get_slice_data()
        if self.view == "axial" and getattr(self.main, "axial_flip_ud", False):
            vv = (H - 1) - vv
        if self.view == "coronal" and getattr(self.main, "coronal_flip_ud", False):
            vv = (H - 1) - vv
        if self.view == "sagittal" and getattr(self.main, "sagittal_flip_ud", False):
            vv = (H - 1) - vv

        # User display transforms
        if self.flip_h:
            uu = (W - 1) - uu
        if self.flip_v:
            vv = (H - 1) - vv

        ang = self.rotation_angle % 360
        if ang != 0:
            cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
            uu_c, vv_c = uu - cx, vv - cy
            rad = np.deg2rad(ang)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            uu_rot = uu_c * cos_a - vv_c * sin_a
            vv_rot = uu_c * sin_a + vv_c * cos_a
            uu, vv = uu_rot + cx, vv_rot + cy
        return QPointF(uu, vv)

    def _inverse_map(self, u_disp, v_disp):
        """Map displayed slice coordinates back to canonical slice coordinates."""
        W, H = self._slice_dims()
        uu, vv = float(u_disp), float(v_disp)

        ang = self.rotation_angle % 360
        if ang != 0:
            cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
            uu_c, vv_c = uu - cx, vv - cy
            rad = np.deg2rad(-ang)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            uu_rot = uu_c * cos_a - vv_c * sin_a
            vv_rot = uu_c * sin_a + vv_c * cos_a
            uu, vv = uu_rot + cx, vv_rot + cy

        # Undo user display transforms
        if self.flip_h:
            uu = (W - 1) - uu
        if self.flip_v:
            vv = (H - 1) - vv

        # Undo view-specific vertical display flips used in get_slice_data()
        if self.view == "axial" and getattr(self.main, "axial_flip_ud", False):
            vv = (H - 1) - vv
        if self.view == "coronal" and getattr(self.main, "coronal_flip_ud", False):
            vv = (H - 1) - vv
        if self.view == "sagittal" and getattr(self.main, "sagittal_flip_ud", False):
            vv = (H - 1) - vv
        return QPointF(uu, vv)

    def _emit_crosshair(self, img_pos: QPointF):
        u, v = int(round(img_pos.x())), int(round(img_pos.y()))
        if self.main.ct_data is None:
            return
        sx, sy, sz = self.main.ct_data.shape
        if self.view == "axial":
            x = int(np.clip(u, 0, sx - 1))
            y = int(np.clip(v, 0, sy - 1))
            z = int(np.clip(self.current_slice, 0, sz - 1))
        elif self.view == "coronal":
            x = int(np.clip(u, 0, sx - 1))
            y = int(np.clip(self.current_slice, 0, sy - 1))
            z = int(np.clip(v, 0, sz - 1))
        else:
            x = int(np.clip(self.current_slice, 0, sx - 1))
            y = int(np.clip(u, 0, sy - 1))
            z = int(np.clip(v, 0, sz - 1))
        self.crosshair_moved.emit(x, y, z)

    def keyPressEvent(self, e):
        try:
            if e.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
                if self.selected_roi is not None and self.selected_roi in self.spherical_rois:
                    self.spherical_rois.remove(self.selected_roi)
                    self.selected_roi = None
                    self.hover_roi = None
                    self.update()
                    self.roi_changed.emit()
                    return
        except Exception:
            pass
        return super().keyPressEvent(e)


# ================================================================
# Workers
# ================================================================
class SegWorker(QThread):
    ok = pyqtSignal(dict)
    fail = pyqtSignal(str)
    stage = pyqtSignal(str)

    def __init__(self, ct_path: str, out_dir: str, fast: bool = True):
        super().__init__()
        self.ct_path = ct_path
        self.out_dir = out_dir
        self.fast = fast

    def run(self):
        try:
            self.stage.emit("Running TotalSegmentator (spine + psoas + femur)…")
            os.makedirs(self.out_dir, exist_ok=True)

            roi_subset = [
                "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5",
                "iliopsoas_left", "iliopsoas_right",
                "femur_left", "femur_right",
            ]

            if TOTALSEG_API:
                try:
                    totalseg(
                        self.ct_path, self.out_dir, task="total",
                        roi_subset=roi_subset, fast=self.fast,
                    )
                except TypeError:
                    totalseg(self.ct_path, self.out_dir, task="total", fast=self.fast)
            elif TOTALSEG_AVAILABLE:
                tried = []
                ran = False
                for exe in ["TotalSegmentator", "totalsegmentator"]:
                    exe_path = shutil.which(exe) or exe
                    cmd = [exe_path, "-i", self.ct_path, "-o", self.out_dir, "--task", "total"]
                    if self.fast:
                        cmd.append("--fast")
                    cmd.extend(["--roi_subset"] + roi_subset)
                    self.stage.emit(f"Running CLI: {os.path.basename(str(exe_path))} …")
                    try:
                        proc = subprocess.run(
                            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, shell=False,
                        )
                        tried.append((
                            exe_path,
                            proc.returncode,
                            "CMD: " + subprocess.list2cmdline([str(x) for x in cmd])
                            + "\nSTDOUT:\n" + (proc.stdout or "<empty>")
                            + "\nSTDERR:\n" + (proc.stderr or "<empty>"),
                        ))
                        if proc.returncode == 0:
                            ran = True
                            break
                    except Exception as e:
                        tried.append((exe_path, str(e)))

                if not ran:
                    for exe in ["TotalSegmentator", "totalsegmentator"]:
                        exe_path = shutil.which(exe) or exe
                        cmd = [exe_path, "-i", self.ct_path, "-o", self.out_dir, "--task", "total"]
                        if self.fast:
                            cmd.append("--fast")
                        self.stage.emit(f"Retrying CLI without roi_subset: {os.path.basename(str(exe_path))} …")
                        try:
                            proc = subprocess.run(
                                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, shell=False,
                            )
                            tried.append((
                                str(exe_path) + "(no_subset)",
                                proc.returncode,
                                "CMD: " + subprocess.list2cmdline([str(x) for x in cmd])
                                + "\nSTDOUT:\n" + (proc.stdout or "<empty>")
                                + "\nSTDERR:\n" + (proc.stderr or "<empty>"),
                            ))
                            if proc.returncode == 0:
                                ran = True
                                break
                        except Exception as e:
                            tried.append((str(exe_path) + "(no_subset)", str(e)))

                if not ran:
                    details = [
                        f"Input CT: {self.ct_path}",
                        f"Input exists: {os.path.exists(self.ct_path)}",
                        f"Input size bytes: {os.path.getsize(self.ct_path) if os.path.exists(self.ct_path) else 'n/a'}",
                        f"Output dir: {self.out_dir}",
                        "",
                        "Attempts:",
                    ]
                    for item in tried:
                        if len(item) >= 3:
                            exe_name, rc, extra = item[0], item[1], item[2]
                            details.append(f"- {exe_name} -> returncode={rc}")
                            if extra:
                                details.append(str(extra))
                        else:
                            details.append(f"- {item}")
                        details.append("")
                    raise RuntimeError("Could not run TotalSegmentator CLI.\n\n" + "\n".join(details))
            else:
                raise RuntimeError("TotalSegmentator not installed.")

            self.stage.emit("Collecting masks…")
            files = {}
            all_masks = []
            for ext in ("*.nii.gz", "*.nii"):
                all_masks.extend(glob(os.path.join(self.out_dir, "**", ext), recursive=True))
            lookup = {os.path.basename(p).lower(): p for p in all_masks}

            def find_mask(candidates):
                for cand in candidates:
                    c = cand.lower()
                    if c in lookup:
                        return lookup[c]
                    for base, full in lookup.items():
                        if c in base:
                            return full
                return None

            for name in ["L1", "L2", "L3", "L4", "L5"]:
                hit = find_mask([
                    f"vertebrae_{name}.nii.gz", f"vertebrae_{name}.nii",
                    f"vertebra_{name}.nii.gz", f"vertebra_{name}.nii",
                    f"{name}.nii.gz", f"{name}.nii",
                ])
                if hit:
                    files[name] = hit

            for key, label in [("PSOAS_L", "iliopsoas_left"), ("PSOAS_R", "iliopsoas_right")]:
                hit = find_mask([f"{label}.nii.gz", f"{label}.nii", label])
                if hit:
                    files[key] = hit

            for key, label in [("FEMUR_L", "femur_left"), ("FEMUR_R", "femur_right")]:
                hit = find_mask([f"{label}.nii.gz", f"{label}.nii", label])
                if hit:
                    files[key] = hit

            if not any(k in files for k in ["L1", "L2", "L3", "L4", "L5"]):
                raise RuntimeError("No vertebra masks found in output.")
            self.ok.emit(files)
        except Exception as e:
            self.fail.emit(str(e))


class RegistrationWorker(QThread):
    done = pyqtSignal(object)
    error = pyqtSignal(str)
    stage = pyqtSignal(str)

    def __init__(self, ct_img, pet_img, ct_np, pet_np):
        super().__init__()
        self.ct_img = ct_img
        self.pet_img = pet_img
        self.ct_np = ct_np
        self.pet_np = pet_np

    @staticmethod
    def _resize_like(mov, ref):
        if ndi_zoom is None:
            scale = np.array(ref.shape) / np.array(mov.shape)
            scale_i = np.maximum(1, np.round(scale).astype(int))
            up = mov.repeat(scale_i[0], axis=0).repeat(scale_i[1], axis=1).repeat(scale_i[2], axis=2)
            out = np.zeros(ref.shape, dtype=mov.dtype)
            nx, ny, nz = np.minimum(up.shape, ref.shape)
            out[:nx, :ny, :nz] = up[:nx, :ny, :nz]
            return out
        factors = np.array(ref.shape) / np.array(mov.shape)
        return ndi_zoom(mov, factors, order=1)

    @staticmethod
    def _nmi(a, b, bins=64):
        a = np.asarray(a, np.float32)
        b = np.asarray(b, np.float32)
        if a.size < 100 or b.size < 100:
            return float("nan")
        a = np.clip(a, np.percentile(a, 1), np.percentile(a, 99))
        b = np.clip(b, np.percentile(b, 1), np.percentile(b, 99))
        H, _, _ = np.histogram2d(a.ravel(), b.ravel(), bins=bins)
        H = H + 1e-6
        Ha = np.sum(H, axis=1)
        Hb = np.sum(H, axis=0)
        pxy = H / np.sum(H)
        px = Ha / np.sum(H)
        py = Hb / np.sum(H)

        def ent(p_arr):
            p_arr = p_arr[p_arr > 0]
            return -np.sum(p_arr * np.log(p_arr))

        Hx = ent(px)
        Hy = ent(py)
        Hxy = ent(pxy.ravel())
        return (Hx + Hy) / Hxy

    def run(self):
        try:
            if SITK_AVAILABLE and (self.ct_img is not None) and (self.pet_img is not None):
                self.stage.emit("Affine registration (Mattes MI)…")
                fixed = sitk.Cast(self.ct_img, sitk.sitkFloat32)
                moving = sitk.Cast(self.pet_img, sitk.sitkFloat32)
                initial = sitk.CenteredTransformInitializer(
                    fixed, moving, sitk.AffineTransform(3),
                    sitk.CenteredTransformInitializerFilter.GEOMETRY,
                )
                reg = sitk.ImageRegistrationMethod()
                reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
                reg.SetMetricSamplingStrategy(reg.RANDOM)
                reg.SetMetricSamplingPercentage(0.2, 1234)
                reg.SetInterpolator(sitk.sitkLinear)
                reg.SetOptimizerAsRegularStepGradientDescent(
                    learningRate=2.0, minStep=1e-3, numberOfIterations=200, relaxationFactor=0.5,
                )
                reg.SetShrinkFactorsPerLevel([4, 2, 1])
                reg.SetSmoothingSigmasPerLevel([2, 1, 0])
                reg.SetInitialTransform(initial, inPlace=False)
                final_tx = reg.Execute(fixed, moving)
                self.stage.emit("Resampling PET to CT grid…")
                pet_reg = sitk.Resample(
                    moving, fixed, final_tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32,
                )
                arr = sitk.GetArrayFromImage(pet_reg).transpose(2, 1, 0)
                mi_val = reg.GetMetricValue()
                mid = self.ct_np.shape[2] // 2
                nmi = self._nmi(self.ct_np[:, :, mid], arr[:, :, mid])
                self.done.emit(
                    {"arr": arr, "info": f"Affine MI={mi_val:.4f}  NMI≈{nmi:.3f}", "tx": final_tx}
                )
                return
            self.stage.emit("Preview: resize PET to CT grid")
            pet_resized = self._resize_like(self.pet_np, self.ct_np)
            mid = self.ct_np.shape[2] // 2
            nmi = self._nmi(self.ct_np[:, :, mid], pet_resized[:, :, mid])
            self.done.emit(
                {"arr": pet_resized, "info": f"Preview only (no registration). NMI≈{nmi:.3f}", "tx": None}
            )
        except Exception as e:
            self.error.emit(f"Registration failed: {e}")


# ================================================================
# DICOM Series Selection Dialog
# ================================================================
class SeriesSelectionDialog(QDialog):
    def __init__(
        self,
        series_list: List[Dict],
        preselect_ct_uid: Optional[str] = None,
        preselect_pet_uid: Optional[str] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Select DICOM Series")
        self.resize(980, 520)
        self.series_list = series_list
        self.selected_ct_uid = preselect_ct_uid
        self.selected_pet_uid = preselect_pet_uid

        v = QVBoxLayout(self)
        info = QLabel(
            "Multiple DICOM series detected. Choose the CT (highest resolution) "
            "and the PET (attenuation-corrected) series to load."
        )
        info.setWordWrap(True)
        v.addWidget(info)

        top = QHBoxLayout()
        self.cb_ct = QComboBox()
        self.cb_pet = QComboBox()
        top.addWidget(QLabel("CT series:"))
        top.addWidget(self.cb_ct, 3)
        top.addSpacing(10)
        top.addWidget(QLabel("PET series:"))
        top.addWidget(self.cb_pet, 3)
        v.addLayout(top)

        self.tbl = QTableWidget(0, 10)
        self.tbl.setHorizontalHeaderLabels(
            ["Mod", "SeriesDescription", "Protocol", "Kernel/Recon", "Px(mm)",
             "Thk(mm)", "Matrix", "Files", "CT score", "PET score"]
        )
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        v.addWidget(self.tbl, 1)

        btns = QHBoxLayout()
        btns.addStretch(1)
        self.btn_ok = QPushButton("Load selected")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_ok.setMinimumHeight(34)
        self.btn_cancel.setMinimumHeight(34)
        btns.addWidget(self.btn_cancel)
        btns.addWidget(self.btn_ok)
        v.addLayout(btns)

        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

        self._populate()
        self.cb_ct.currentIndexChanged.connect(self._sync_table_highlight)
        self.cb_pet.currentIndexChanged.connect(self._sync_table_highlight)
        self.tbl.itemSelectionChanged.connect(self._sync_combos_from_table)
        self._sync_table_highlight()

    def _populate(self):
        def fmt_float(x):
            try:
                return f"{float(x):.2f}" if x is not None else ""
            except Exception:
                return ""

        def fmt_px(px):
            if not px or len(px) < 2:
                return ""
            try:
                return f"{float(px[0]):.2f}×{float(px[1]):.2f}"
            except Exception:
                return ""

        def fmt_mat(r, c):
            if not r or not c:
                return ""
            return f"{int(r)}×{int(c)}"

        self.tbl.setRowCount(0)
        self.cb_ct.clear()
        self.cb_pet.clear()

        ct_items = []
        pet_items = []

        for s in self.series_list:
            mod = (s.get("modality") or "UNK").upper()
            label = s.get("series_desc") or s.get("protocol") or "Unnamed"
            pretty = f"{mod} | {label} | {s.get('n_files', 0)} files"
            uid = s.get("series_id")

            if mod == "CT":
                ct_items.append((pretty, uid))
            if mod in ("PT", "NM", "PET"):
                pet_items.append((pretty, uid))

            r = self.tbl.rowCount()
            self.tbl.insertRow(r)
            vals = [
                mod,
                (s.get("series_desc") or "")[:80],
                (s.get("protocol") or "")[:60],
                (s.get("kernel") or s.get("recon") or "")[:30],
                fmt_px(s.get("pixel_spacing")),
                fmt_float(s.get("slice_thickness") or s.get("spacing_between_slices")),
                fmt_mat(s.get("rows"), s.get("cols")),
                str(s.get("n_files", 0)),
                f"{s.get('ct_score', 0):.1f}",
                f"{s.get('pet_score', 0):.1f}",
            ]
            for c_idx, val in enumerate(vals):
                it = QTableWidgetItem(val)
                it.setData(Qt.ItemDataRole.UserRole, uid)
                self.tbl.setItem(r, c_idx, it)

        for pretty, uid in ct_items:
            self.cb_ct.addItem(pretty, uid)
        for pretty, uid in pet_items:
            self.cb_pet.addItem(pretty, uid)

        if self.selected_ct_uid:
            idx = self.cb_ct.findData(self.selected_ct_uid)
            if idx >= 0:
                self.cb_ct.setCurrentIndex(idx)
        if self.selected_pet_uid:
            idx = self.cb_pet.findData(self.selected_pet_uid)
            if idx >= 0:
                self.cb_pet.setCurrentIndex(idx)

    def _sync_table_highlight(self):
        ct_uid = self.cb_ct.currentData()
        pet_uid = self.cb_pet.currentData()
        for r in range(self.tbl.rowCount()):
            uid = self.tbl.item(r, 0).data(Qt.ItemDataRole.UserRole)
            is_ct = uid == ct_uid
            is_pet = uid == pet_uid
            for c_idx in range(self.tbl.columnCount()):
                item = self.tbl.item(r, c_idx)
                if not item:
                    continue
                if is_ct and is_pet:
                    item.setBackground(QColor(220, 255, 220))
                elif is_ct:
                    item.setBackground(QColor(230, 245, 255))
                elif is_pet:
                    item.setBackground(QColor(255, 235, 235))
                else:
                    item.setBackground(QColor(255, 255, 255))

    def _sync_combos_from_table(self):
        rows = self.tbl.selectionModel().selectedRows()
        if not rows:
            return
        uid = self.tbl.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole)
        idx = self.cb_ct.findData(uid)
        if idx >= 0:
            self.cb_ct.setCurrentIndex(idx)
        idx2 = self.cb_pet.findData(uid)
        if idx2 >= 0:
            self.cb_pet.setCurrentIndex(idx2)

    def get_selection(self) -> Tuple[Optional[str], Optional[str]]:
        return self.cb_ct.currentData(), self.cb_pet.currentData()


# ================================================================
# Main Window
# ================================================================
class BoneDensityAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bone Density Analyzer – AutoSEG (TotalSegmentator ROI Workflow)")
        self.setStyleSheet("""
            QWidget { font-size: 9.8pt; }
            QGroupBox {
                font-weight: 600;
                border: 1px solid #454545;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
            }
            QPushButton {
                padding: 8px 12px;
                min-height: 34px;
            }
            QComboBox, QSpinBox, QDoubleSpinBox { min-height: 27px; }
            QTextEdit, QTableWidget { font-size: 9.8pt; }
        """)

        # ---------- Data / state ----------
        self.image_data = self.ct_data = self.pet_data = self.registered_pet = None
        self.ct_img = None
        self.pet_img = None
        self.reg_tx = None
        self.ct_path = None
        self.pet_path = None
        self.ct_spacing: Optional[Tuple[float, float, float]] = None
        self.fusion_mode = "CT"
        self.roi_mode = None
        self.patient_age = 60
        self.patient_sex = "F"
        self.contrast_mode = True
        self.vbmd_young_mean = 150.0
        self.vbmd_young_sd = 30.0
        self.vbmd_age_mean = 120.0
        self.vbmd_age_sd = 25.0
        self.dxa_spine_young_mean = 980.0
        self.dxa_spine_young_sd = 110.0
        self.dxa_fn_young_mean = 860.0
        self.dxa_fn_young_sd = 100.0
        # Projection-specific pseudo-norms used only for AP/areal results.
        # These are kept separate from CT vBMD norms and scanner DXA norms.
        self.proj_spine_young_mean = 210.0
        self.proj_spine_young_sd = 35.0
        self.proj_spine_age_mean = 180.0
        self.proj_spine_age_sd = 30.0
        self.proj_fn_young_mean = 235.0
        self.proj_fn_young_sd = 40.0
        self.proj_fn_age_mean = 205.0
        self.proj_fn_age_sd = 35.0
        self.proj_spine_slope = 1.00
        self.proj_spine_intercept = 0.0
        self.proj_fn_slope = 1.00
        self.proj_fn_intercept = 0.0
        self.ap_bone_only_zero_clip = True
        self.ap_auto_hu_min = True
        self.ap_show_thickness_overlay = True
        self.ap_hu_min = 50.0
        self.ap_hu_max = 2000.0
        self.ap_lock_to_vertebral_body = True
        self.site_names: List[str] = ["L1", "L2", "L3", "L4", "FN_L", "FN_R"]
        self.vertebral: Dict[str, Optional[dict]] = {v: None for v in self.site_names}
        self.vertebra_masks: Dict[str, Optional[np.ndarray]] = {
            v: None for v in ["L1", "L2", "L3", "L4", "L5"]
        }
        # >>> FIX: initialise extra_masks (was missing → crash on femoral neck placement) <<<
        self.extra_masks: Dict[str, Optional[np.ndarray]] = {}
        self.qc_results: Dict[str, dict] = {}

        # QC thresholds
        self.qc_overlap_min = 0.25
        self.qc_metal_hu_thr = 2500.0
        self.qc_metal_frac_thr = 0.005
        self.qc_cortex_hi_frac_thr = 0.10
        self.qc_femur_min_z_mm = 120.0

        # QC overlay preview
        self.qc_overlay_enabled = True
        self.qc_overlay_site = None
        self._qc_overlay_cache = None

        self.roi_clipboard = None
        self._site_rows: Optional[List[Tuple[float, float]]] = None
        self._last_active_canvas = None
        self._femur_cropped: Dict[str, bool] = {}
        self._temp_nifti_dir: Optional[str] = None
        self._dicom_extract_dir: Optional[str] = None
        self.target_roi_specs: Dict[str, dict] = {}

        # Display / fusion
        self.pet_alpha = 0.4
        self.pet_low_pct = 5.0
        self.axial_flip_ud = False
        self.axial_invert_z = False
        self.coronal_flip_ud = True
        self.sagittal_flip_ud = True
        self.sync_views = True
        self.show_overlay = True
        self.ct_win_center = 500.0
        self.ct_win_width = 1400.0

        # Calibration
        self.cal_roi_slope = 0.80
        self.cal_roi_intercept = 10.0
        self.cal_roi_r2 = 1.0
        self.cal_fat_bmd_anchor = -70.0
        self.cal_psoas_bmd_anchor = 40.0
        self.dxa_proj_last = None
        self.cal_preset = "contrast"
        self.preset_params = {"contrast": (0.71, 13.8), "native": (0.81, 0.84)}
        self.cal_blend_lambda = 0.40
        self.cal_slope_eff = 0.76
        self.cal_intercept_eff = 12.3
        self.cal_r2_eff = 1.0

        # Mask
        self.mask_enable = True
        self.mask_hu_min = -100.0
        self.mask_hu_max = 400.0
        self.mask_erode_px = 1
        self.auto_cortex = True
        self.cortex_hu_thr = 350.0
        self.cortex_min_px = 1
        self.cortex_max_px = 8
        self.exclusion_rule = "None"

        # Auto-ROI
        self.seg_fast = False
        self.seg_shrink = 0.60
        self.seg_auto_clear = True
        self.seg_place_psoas = True
        self.seg_place_fat = True
        self.seg_use_cache = True
        self.seg_cache_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "totalseg_cache")

        # Crosshair
        self.crosshair = [0, 0, 0]
        self._updating_gui = False

        # ROI creation preference
        self.require_alt_for_new_roi = True
        self._open_dxa_after_autoplace = False
        self._last_totalseg_ct_path = None
        self._last_totalseg_out_dir = None

        self._build_ui()
        self._update_roi_target_status()
        self._recompute_effective_calibration()
        self._fit_window_to_screen()
        QTimer.singleShot(0, self._fit_window_to_screen)

    # ----------------------------------------------------------------
    # UI build
    # ----------------------------------------------------------------
    def _fit_window_to_screen(self):
        scr = self.screen() or QApplication.primaryScreen()
        ag = scr.availableGeometry() if scr else None
        aw, ah = (ag.width(), ag.height()) if ag else (1600, 900)
        self.setMinimumSize(1100, 720)
        w = min(int(aw * 0.92), 1600)
        h = min(int(ah * 0.92), 950)
        self.resize(max(1100, w), max(720, h))
        if hasattr(self, "results"):
            self.results.setMinimumWidth(420)

    def _build_ui(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        main = QHBoxLayout(cw)
        main.setSpacing(12)
        main.setContentsMargins(10, 10, 10, 10)

        left_wrap = QWidget()
        left_wrap.setLayout(self._left_panel())
        left_wrap.setMinimumWidth(470)

        middle_wrap = QWidget()
        middle_wrap.setLayout(self._middle_panel())

        main.addWidget(left_wrap, 0)
        main.addWidget(middle_wrap, 1)
        main.addWidget(self._right_tabs_scrolled(), 0)
        self._right_panel_widget.setMinimumWidth(720)

    def _left_panel(self):
        panel = QVBoxLayout()
        panel.setSpacing(10)
        panel.setContentsMargins(4, 4, 4, 4)
        panel.addWidget(self._patient_left_box())
        panel.addWidget(self._files_box())
        panel.addWidget(self._display_box())
        panel.addWidget(self._norms_left_box())
        panel.addWidget(self._image_controls())
        panel.addWidget(self._roi_tools())
        panel.addStretch()
        return panel

    def _patient_left_box(self):
        g = QGroupBox("Patient")
        f = QFormLayout()
        f.setContentsMargins(10, 8, 10, 8)
        f.setHorizontalSpacing(12)
        f.setVerticalSpacing(6)
        self.age_left = QSpinBox()
        self.age_left.setRange(1, 120)
        self.age_left.setValue(int(self.patient_age))
        self.age_left.valueChanged.connect(self._set_age_all)
        sex_row = QWidget()
        h = QHBoxLayout(sex_row); h.setContentsMargins(0,0,0,0); h.setSpacing(8)
        self.sexF_left = QRadioButton("Female")
        self.sexM_left = QRadioButton("Male")
        self.sexF_left.setChecked(True)
        self.sexF_left.toggled.connect(lambda: self._set_sex_all('F' if self.sexF_left.isChecked() else 'M'))
        h.addWidget(self.sexF_left); h.addWidget(self.sexM_left); h.addStretch(1)
        self.chk_contrast_mode = QCheckBox("Contrast PET/CT mode")
        self.chk_contrast_mode.setChecked(self.contrast_mode)
        self.chk_contrast_mode.toggled.connect(self._set_contrast_mode)
        btn_row = QWidget()
        hb = QHBoxLayout(btn_row); hb.setContentsMargins(0,0,0,0); hb.setSpacing(8)
        btn_export = QPushButton("Export CSV")
        btn_export.clicked.connect(self.export_csv)
        btn_results = QPushButton("Open Results Window")
        btn_results.clicked.connect(self.open_results_window)
        hb.addWidget(btn_export); hb.addWidget(btn_results)
        f.addRow("Age (yrs)", self.age_left)
        f.addRow("Sex", sex_row)
        f.addRow(self.chk_contrast_mode)
        f.addRow(btn_row)
        g.setLayout(f)
        return g

    def _norms_left_box(self):
        g = QGroupBox("Norms / DXA")
        f = QFormLayout()
        f.setContentsMargins(10, 8, 10, 8)
        f.setHorizontalSpacing(12)
        f.setVerticalSpacing(6)
        self.sp_vbmd_young_mean = QDoubleSpinBox(); self.sp_vbmd_young_mean.setRange(0,5000); self.sp_vbmd_young_mean.setValue(self.vbmd_young_mean)
        self.sp_vbmd_young_sd = QDoubleSpinBox(); self.sp_vbmd_young_sd.setRange(0.1,1000); self.sp_vbmd_young_sd.setValue(self.vbmd_young_sd)
        self.sp_vbmd_age_mean = QDoubleSpinBox(); self.sp_vbmd_age_mean.setRange(0,5000); self.sp_vbmd_age_mean.setValue(self.vbmd_age_mean)
        self.sp_vbmd_age_sd = QDoubleSpinBox(); self.sp_vbmd_age_sd.setRange(0.1,1000); self.sp_vbmd_age_sd.setValue(self.vbmd_age_sd)
        self.sp_dxa_spine_mean = QDoubleSpinBox(); self.sp_dxa_spine_mean.setRange(0,5000); self.sp_dxa_spine_mean.setValue(self.dxa_spine_young_mean)
        self.sp_dxa_spine_sd = QDoubleSpinBox(); self.sp_dxa_spine_sd.setRange(0.1,1000); self.sp_dxa_spine_sd.setValue(self.dxa_spine_young_sd)
        self.sp_dxa_fn_mean = QDoubleSpinBox(); self.sp_dxa_fn_mean.setRange(0,5000); self.sp_dxa_fn_mean.setValue(self.dxa_fn_young_mean)
        self.sp_dxa_fn_sd = QDoubleSpinBox(); self.sp_dxa_fn_sd.setRange(0.1,1000); self.sp_dxa_fn_sd.setValue(self.dxa_fn_young_sd)
        for w, attr in [(self.sp_vbmd_young_mean,'vbmd_young_mean'),(self.sp_vbmd_young_sd,'vbmd_young_sd'),(self.sp_vbmd_age_mean,'vbmd_age_mean'),(self.sp_vbmd_age_sd,'vbmd_age_sd'),(self.sp_dxa_spine_mean,'dxa_spine_young_mean'),(self.sp_dxa_spine_sd,'dxa_spine_young_sd'),(self.sp_dxa_fn_mean,'dxa_fn_young_mean'),(self.sp_dxa_fn_sd,'dxa_fn_young_sd')]:
            w.valueChanged.connect(lambda v, a=attr: setattr(self, a, float(v)))
        f.addRow(QLabel("vBMD Norms (mg/cm³)"))
        f.addRow("Young mean", self.sp_vbmd_young_mean)
        f.addRow("Young SD", self.sp_vbmd_young_sd)
        f.addRow("Age mean", self.sp_vbmd_age_mean)
        f.addRow("Age SD", self.sp_vbmd_age_sd)
        f.addRow(QLabel("DXA Norms aBMD (mg/cm²)"))
        f.addRow("Spine young mean", self.sp_dxa_spine_mean)
        f.addRow("Spine young SD", self.sp_dxa_spine_sd)
        f.addRow("FN young mean", self.sp_dxa_fn_mean)
        f.addRow("FN young SD", self.sp_dxa_fn_sd)
        g.setLayout(f)
        return g

    def _middle_panel(self):
        panel = QVBoxLayout()
        panel.setSpacing(8)
        top = QHBoxLayout()
        top.setSpacing(8)
        self.canvas_axial = ImageCanvas("axial", self)
        top.addWidget(self._view_box("Axial", self.canvas_axial))
        self.canvas_coronal = ImageCanvas("coronal", self)
        top.addWidget(self._view_box("Coronal", self.canvas_coronal))
        panel.addLayout(top)
        self.canvas_sagittal = ImageCanvas("sagittal", self)
        panel.addWidget(self._view_box("Sagittal", self.canvas_sagittal))
        for c in self._canvases():
            c.crosshair_moved.connect(self._sync_crosshairs)
        return panel

    def _right_panel_scrolled(self):
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        panel = self._right_panel_compact()
        self._right_panel_widget = sa
        sa.setWidget(panel)
        sa.setMinimumWidth(560)
        return sa

    def _right_panel_compact(self):
        wrap = QWidget()
        v = QVBoxLayout(wrap)
        v.setSpacing(8)
        v.setContentsMargins(0, 0, 0, 0)

        # Main clinical workflow: ROI/calibration first, then AP projection, then optional auto-seg.
        v.addWidget(self._tab_calibration())
        v.addWidget(self._tab_composite())
        v.addWidget(self._tab_auto())

        title = QLabel("RESULTS")
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #2196F3;")
        btn_dxa_inline = QPushButton("Open Editable Areal 2D DXA/AP")
        btn_dxa_inline.setMinimumHeight(34)
        btn_dxa_inline.clicked.connect(self.open_dxa_projection_dialog)
        self.results = QTextEdit()
        self.results.setReadOnly(True)
        self.results.setMinimumHeight(260)
        self.results.setFont(QFont("Courier New", 9))
        v.addWidget(title)
        v.addWidget(btn_dxa_inline)
        v.addWidget(self.results)
        v.addStretch(1)
        return wrap

    def _right_tabs_scrolled(self):
        inner = self._right_tabs()
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sa.setWidget(inner)
        sa.setMinimumWidth(640)
        self._right_panel_widget = sa
        return sa

    def _right_tabs(self):
        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.TabPosition.North)
        tabs.setDocumentMode(True)
        tabs.setUsesScrollButtons(True)
        tabs.setElideMode(Qt.TextElideMode.ElideNone)
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #454545;
                border-radius: 8px;
                top: -1px;
            }
            QTabBar::tab {
                min-width: 112px;
                min-height: 27px;
                padding: 5px 9px;
                margin-right: 4px;
                font-size: 9.8pt;
                font-weight: 600;
            }
        """)

        tabs.addTab(self._tab_calibration(), "Workflow")
        tabs.addTab(self._tab_composite(), "Areal 2D DXA/AP")
        tabs.addTab(self._tab_vertebrae(), "Sites")
        tabs.addTab(self._tab_mask(), "Mask / Cortex")
        tabs.addTab(self._tab_qc(), "QC")
        tabs.addTab(self._tab_auto(), "Auto-Seg")
        tabs.addTab(self._tab_patient(), "Patient / Report")
        tabs.addTab(self._results_tab(), "Results")

        wrap = QWidget()
        wrap.setContentsMargins(0, 0, 0, 0)
        v = QVBoxLayout()
        v.setSpacing(8)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(tabs)
        wrap.setLayout(v)
        return wrap


    def _ensure_auto_seg_buttons(self):
        if not hasattr(self, "btn_totalseg_check"):
            self.btn_totalseg_check = QPushButton("Check TotalSegmentator Setup")
            self.btn_totalseg_check.setMinimumHeight(34)
            self.btn_totalseg_check.clicked.connect(self._check_totalseg_environment)
        if not hasattr(self, "btn_totalseg"):
            self.btn_totalseg = QPushButton("Run TotalSegmentator + Auto-place Editable ROIs")
            self.btn_totalseg.setMinimumHeight(34)
            self.btn_totalseg.clicked.connect(self.auto_place_with_totalseg)
        if not hasattr(self, "btn_auto_workflow"):
            self.btn_auto_workflow = QPushButton("Auto-place → Measure → Open Areal 2D DXA/AP")
            self.btn_auto_workflow.setMinimumHeight(36)
            self.btn_auto_workflow.clicked.connect(self.run_full_auto_workflow)
        self.btn_totalseg.setEnabled(TOTALSEG_AVAILABLE)
        self.btn_auto_workflow.setEnabled(TOTALSEG_AVAILABLE)

    def _results_tab(self):
        w = QWidget()
        v = QVBoxLayout()
        v.setSpacing(10)
        v.setContentsMargins(10, 10, 10, 10)

        title = QLabel("Measurement Results")
        title.setStyleSheet("font-weight: 700; font-size: 14pt; color: #E8F1FF;")
        v.addWidget(title)

        subtitle = QLabel("Clean report-style output for screenshots and publication figures.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #B8C7D9;")
        v.addWidget(subtitle)

        self.results = QTextEdit()
        self.results.setReadOnly(True)
        self.results.setMinimumHeight(560)
        self.results.setFont(QFont("Consolas", 10))
        v.addWidget(self.results)
        w.setLayout(v)
        return w

    # ---- Tab builders ----
    
    def _tab_calibration(self):
        w = QWidget()
        outer = QVBoxLayout(w)
        outer.setSpacing(10)
        outer.setContentsMargins(8, 8, 8, 8)

        def _mkbtn(text, slot, h=42):
            b = QPushButton(text)
            b.setMinimumHeight(h)
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            b.clicked.connect(slot)
            return b

        g_roi = QGroupBox("ROI Tools")
        roi_v = QVBoxLayout(g_roi)
        roi_v.setContentsMargins(8, 6, 8, 6)
        roi_v.setSpacing(4)
        self.roi_label_combo = QComboBox()
        self.roi_label_combo.addItems(["L1", "L2", "L3", "L4", "FN_L", "FN_R", "FAT", "MUSCLE", "PSOAS_L", "PSOAS_R"])
        roi_v.addWidget(QLabel("Label"))
        roi_v.addWidget(self.roi_label_combo)

        roi_grid = QGridLayout()
        roi_grid.setHorizontalSpacing(8)
        roi_grid.setVerticalSpacing(4)
        roi_grid.addWidget(_mkbtn("New ROI @ Crosshair", self.new_update_roi_at_crosshair), 0, 0, 1, 2)
        roi_grid.addWidget(_mkbtn("Lock Selected ROI Here", self.lock_selected_roi_here), 1, 0, 1, 2)
        roi_grid.addWidget(_mkbtn("Measure All", self.measure_all_sites), 2, 0, 1, 2)
        roi_v.addLayout(roi_grid)

        self.lbl_roi_targets = QLabel("")
        self.lbl_roi_targets.setWordWrap(True)
        self.lbl_roi_targets.setStyleSheet("font-family: Consolas, 'Courier New', monospace; font-size: 9.8pt;")
        roi_v.addWidget(QLabel("Target ROI status"))
        roi_v.addWidget(self.lbl_roi_targets)

        self._ensure_auto_seg_buttons()
        g_auto = QGroupBox("Automatic ROI Placement (TotalSegmentator)")
        auto_v = QVBoxLayout(g_auto)
        auto_v.setContentsMargins(8, 6, 8, 6)
        auto_v.setSpacing(4)

        auto_btns = QGridLayout()
        auto_btns.setHorizontalSpacing(8)
        auto_btns.setVerticalSpacing(4)
        auto_btns.addWidget(self.btn_totalseg_check, 0, 0)
        auto_btns.addWidget(self.btn_totalseg, 0, 1)
        auto_btns.addWidget(self.btn_auto_workflow, 1, 0, 1, 2)
        auto_v.addLayout(auto_btns)

        g_eq = QGroupBox("Calibration Settings / Equation")
        f = QFormLayout()
        f.setContentsMargins(10, 8, 10, 8)
        f.setHorizontalSpacing(12)
        f.setVerticalSpacing(6)
        self.rb_p_contrast = QRadioButton("Preset: Contrast")
        self.rb_p_native = QRadioButton("Preset: Native")
        if self.cal_preset == "contrast":
            self.rb_p_contrast.setChecked(True)
        else:
            self.rb_p_native.setChecked(True)
        self.rb_p_contrast.toggled.connect(lambda: self._set_preset("contrast"))
        self.rb_p_native.toggled.connect(lambda: self._set_preset("native"))
        self.lbl_eq_eff = QLabel(self._eq_eff_text())
        self.lbl_eq_eff.setWordWrap(True)
        self.lbl_eq_eff.setStyleSheet("font-weight:600;")
        self.blend = QSlider(Qt.Orientation.Horizontal)
        self.blend.setRange(0, 100)
        self.blend.setValue(int(self.cal_blend_lambda * 100))
        self.lbl_blend = QLabel(self._blend_text())
        self.lbl_blend.setWordWrap(True)
        self.blend.valueChanged.connect(lambda v: self._on_blend(float(v) / 100.0))
        btn_recalc = _mkbtn("Recalculate All", self.recompute_all_with_calibration, h=40)
        f.addRow(self.rb_p_contrast)
        f.addRow(self.rb_p_native)
        f.addRow("Blend λ (0=preset,1=phantomless)", self.blend)
        f.addRow(self.lbl_blend)
        f.addRow("Effective equation", self.lbl_eq_eff)
        f.addRow(btn_recalc)
        g_eq.setLayout(f)

        g_cal = QGroupBox("Calibration ROIs (FAT / MUSCLE)")
        cal_v = QVBoxLayout(g_cal)
        cal_v.setContentsMargins(8, 6, 8, 6)
        cal_v.setSpacing(4)

        action_grid = QGridLayout()
        action_grid.setHorizontalSpacing(8)
        action_grid.setVerticalSpacing(4)
        action_grid.addWidget(_mkbtn("Place FAT @ center", lambda: self.place_calibration_roi_center('FAT')), 0, 0)
        action_grid.addWidget(_mkbtn("Place MUSCLE @ center", lambda: self.place_calibration_roi_center('MUSCLE')), 0, 1)
        action_grid.addWidget(_mkbtn("Apply → FAT", lambda: self._retag_selected_roi('FAT')), 1, 0)
        action_grid.addWidget(_mkbtn("Apply → MUSCLE", lambda: self._retag_selected_roi('MUSCLE')), 1, 1)
        action_grid.addWidget(_mkbtn("Compute 2-point from FAT + MUSCLE", self._compute_manual_calibration, h=42), 2, 0, 1, 2)
        action_grid.addWidget(_mkbtn("Validate Calibration", self.validate_calibration, h=42), 3, 0, 1, 2)
        cal_v.addLayout(action_grid)

        self.sp_nudge = QSpinBox()
        self.sp_nudge.setRange(1, 50)
        self.sp_nudge.setValue(2)
        self.sp_cal_radius = QSpinBox()
        self.sp_cal_radius.setRange(3, 100)
        self.sp_cal_radius.setValue(18)
        self.sp_thickness = QDoubleSpinBox()
        self.sp_thickness.setRange(1, 50)
        self.sp_thickness.setValue(10.0)

        adv_btn = QToolButton()
        adv_btn.setText("Show ROI size / offset settings")
        adv_btn.setCheckable(True)
        adv_btn.setChecked(False)
        adv_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        adv_btn.setArrowType(Qt.ArrowType.RightArrow)

        adv_wrap = QWidget()
        adv_form = QFormLayout(adv_wrap)
        adv_form.setContentsMargins(10, 0, 10, 0)
        adv_form.setHorizontalSpacing(12)
        adv_form.setVerticalSpacing(4)
        adv_form.addRow("Nudge px", self.sp_nudge)
        adv_form.addRow("Radius (px)", self.sp_cal_radius)
        adv_form.addRow("T (mm)", self.sp_thickness)
        adv_wrap.setVisible(False)

        def _toggle_adv(checked):
            adv_wrap.setVisible(bool(checked))
            adv_btn.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
            adv_btn.setText("Hide ROI size / offset settings" if checked else "Show ROI size / offset settings")

        adv_btn.toggled.connect(_toggle_adv)
        cal_v.addWidget(adv_btn)
        cal_v.addWidget(adv_wrap)

        outer.addWidget(g_roi)
        outer.addWidget(g_auto)
        outer.addWidget(g_eq)
        outer.addWidget(g_cal)
        outer.addStretch(1)
        return w

    def _tab_mask(self):

        w = QWidget()
        outer = QVBoxLayout(w)
        outer.setSpacing(12)
        outer.setContentsMargins(8, 8, 8, 8)
        g = QGroupBox("Cancellous Mask / Cortex Auto", w)
        f = QFormLayout()
        f.setContentsMargins(10, 8, 10, 8)
        f.setHorizontalSpacing(12)
        f.setVerticalSpacing(8)
        self.chk_mask = QCheckBox("Enable cancellous-only stats")
        self.chk_mask.setChecked(self.mask_enable)
        self.chk_mask.toggled.connect(lambda v: setattr(self, "mask_enable", bool(v)))
        self.sp_min = QDoubleSpinBox()
        self.sp_min.setRange(-2000.0, 2000.0)
        self.sp_min.setDecimals(1)
        self.sp_min.setValue(self.mask_hu_min)
        self.sp_max = QDoubleSpinBox()
        self.sp_max.setRange(-2000.0, 2000.0)
        self.sp_max.setDecimals(1)
        self.sp_max.setValue(self.mask_hu_max)
        self.sp_ero = QSpinBox()
        self.sp_ero.setRange(0, 20)
        self.sp_ero.setValue(self.mask_erode_px)
        self.chk_auto = QCheckBox("Auto cortex shrink")
        self.chk_auto.setChecked(self.auto_cortex)
        self.sp_thr = QDoubleSpinBox()
        self.sp_thr.setRange(-1000.0, 3000.0)
        self.sp_thr.setDecimals(1)
        self.sp_thr.setValue(self.cortex_hu_thr)
        self.sp_cmin = QSpinBox()
        self.sp_cmin.setRange(0, 20)
        self.sp_cmin.setValue(self.cortex_min_px)
        self.sp_cmax = QSpinBox()
        self.sp_cmax.setRange(0, 30)
        self.sp_cmax.setValue(self.cortex_max_px)
        self.sp_min.valueChanged.connect(lambda v: setattr(self, "mask_hu_min", float(v)))
        self.sp_max.valueChanged.connect(lambda v: setattr(self, "mask_hu_max", float(v)))
        self.sp_ero.valueChanged.connect(lambda v: setattr(self, "mask_erode_px", int(v)))
        self.chk_auto.toggled.connect(lambda v: setattr(self, "auto_cortex", bool(v)))
        self.sp_thr.valueChanged.connect(lambda v: setattr(self, "cortex_hu_thr", float(v)))
        self.sp_cmin.valueChanged.connect(lambda v: setattr(self, "cortex_min_px", int(v)))
        self.sp_cmax.valueChanged.connect(lambda v: setattr(self, "cortex_max_px", int(v)))
        f.addRow(self.chk_mask)
        f.addRow("HU min:", self.sp_min)
        f.addRow("HU max:", self.sp_max)
        f.addRow("Extra erode (px):", self.sp_ero)
        f.addRow(self.chk_auto)
        f.addRow("Cortex HU thr:", self.sp_thr)
        row = QWidget()
        h = QHBoxLayout(row)
        h.setSpacing(8)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(QLabel("Shrink min:"))
        h.addWidget(self.sp_cmin)
        h.addSpacing(12)
        h.addWidget(QLabel("max:"))
        h.addWidget(self.sp_cmax)
        h.addStretch(1)
        f.addRow(row)
        g.setLayout(f)
        outer.addWidget(g)
        outer.addStretch(1)
        return w


    def _tab_auto(self):
        self._ensure_auto_seg_buttons()
        w = QWidget()
        v = QVBoxLayout(w)
        v.setSpacing(12)
        v.setContentsMargins(8, 8, 8, 8)

        header = QLabel(
            "Fully automated TotalSegmentator workflow: segment CT, auto-place editable ROIs for L1–L4 and bilateral femoral necks, then optionally add psoas and retroperitoneal fat calibration ROIs."
        )
        header.setWordWrap(True)
        header.setStyleSheet("font-weight:bold;")
        v.addWidget(header)

        status_lines = [
            f"TotalSegmentator backend: {'available' if TOTALSEG_AVAILABLE else 'NOT FOUND'}",
            "Input accepted: CT NIfTI, DICOM folder, or zipped DICOM. The app writes a temporary CT NIfTI automatically when needed.",
            "After auto-placement, every ROI remains movable and resizable for manual correction.",
        ]
        lbl_status = QLabel("\n".join(status_lines))
        lbl_status.setWordWrap(True)
        lbl_status.setStyleSheet(
            "background:#E8F5E9; border:1px solid #C8E6C9; padding:8px; border-radius:6px; color:#2E7D32;"
            if TOTALSEG_AVAILABLE
            else "background:#FFF3CD; border:1px solid #FFEEBA; padding:8px; border-radius:6px; color:#856404;"
        )
        v.addWidget(lbl_status)

        g = QGroupBox("Auto-Segmentation & ROI Placement")
        form = QFormLayout()
        form.setContentsMargins(10, 8, 10, 8)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

        self.chk_fast = QCheckBox("Fast mode (lower resolution, faster)")
        self.chk_fast.setChecked(self.seg_fast)
        self.chk_fast.toggled.connect(lambda b: setattr(self, "seg_fast", bool(b)))

        self.chk_seg_clear = QCheckBox("Clear existing auto-ROI tags before new auto-placement")
        self.chk_seg_clear.setChecked(self.seg_auto_clear)
        self.chk_seg_clear.toggled.connect(lambda b: setattr(self, "seg_auto_clear", bool(b)))

        self.chk_seg_psoas = QCheckBox("Auto-place psoas ROIs when masks are available")
        self.chk_seg_psoas.setChecked(self.seg_place_psoas)
        self.chk_seg_psoas.toggled.connect(lambda b: setattr(self, "seg_place_psoas", bool(b)))

        self.chk_seg_fat = QCheckBox("Auto-place retroperitoneal fat ROI near L3")
        self.chk_seg_fat.setChecked(self.seg_place_fat)
        self.chk_seg_fat.toggled.connect(lambda b: setattr(self, "seg_place_fat", bool(b)))

        self.chk_seg_cache = QCheckBox("Reuse cached segmentations for the same CT")
        self.chk_seg_cache.setChecked(self.seg_use_cache)
        self.chk_seg_cache.toggled.connect(lambda b: setattr(self, "seg_use_cache", bool(b)))

        self.sp_shrink = QDoubleSpinBox()
        self.sp_shrink.setRange(0.30, 0.95)
        self.sp_shrink.setSingleStep(0.05)
        self.sp_shrink.setDecimals(2)
        self.sp_shrink.setValue(self.seg_shrink)
        self.sp_shrink.valueChanged.connect(lambda x: setattr(self, "seg_shrink", float(x)))

        self.btn_totalseg_check = QPushButton("Check TotalSegmentator Setup")
        self.btn_totalseg_check.setMinimumHeight(34)
        self.btn_totalseg_check.clicked.connect(self._check_totalseg_environment)

        self.btn_totalseg = QPushButton("Run TotalSegmentator + Auto-place Editable ROIs")
        self.btn_totalseg.setMinimumHeight(34)
        self.btn_totalseg.setEnabled(TOTALSEG_AVAILABLE)
        self.btn_totalseg.clicked.connect(self.auto_place_with_totalseg)

        self.btn_auto_workflow = QPushButton("Auto-place → Measure → Open Areal 2D DXA/AP")
        self.btn_auto_workflow.setMinimumHeight(36)
        self.btn_auto_workflow.setEnabled(TOTALSEG_AVAILABLE)
        self.btn_auto_workflow.clicked.connect(self.run_full_auto_workflow)

        self.btn_measure_all = QPushButton("Measure ALL Placed Sites (L1–L4 + FN)")
        self.btn_measure_all.setMinimumHeight(34)
        self.btn_measure_all.clicked.connect(self.measure_all_sites)

        form.addRow(self.chk_fast)
        form.addRow(self.chk_seg_clear)
        form.addRow(self.chk_seg_psoas)
        form.addRow(self.chk_seg_fat)
        form.addRow(self.chk_seg_cache)
        form.addRow("ROI shrink inside mask:", self.sp_shrink)
        form.addRow(self.btn_totalseg_check)
        form.addRow(self.btn_totalseg)
        form.addRow(self.btn_auto_workflow)
        form.addRow(self.btn_measure_all)
        form.addRow("Auto-Seg status:", self.lbl_stage)
        form.addRow(self.progress)
        g.setLayout(form)
        v.addWidget(g)

        note = QLabel(
            "Windows note: install PyTorch first, then install TotalSegmentator in the same Python environment. Initial model download can take several minutes and several GB of disk space."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#B0BEC5;")
        v.addWidget(note)
        v.addStretch(1)
        return w

    def _tab_qc(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setSpacing(12)
        v.setContentsMargins(8, 8, 8, 8)

        header = QLabel("Quality Control (QC)")
        header.setWordWrap(True)
        header.setStyleSheet("font-weight:bold;")

        self.qc_table = QTableWidget(0, 7)
        self.qc_table.setHorizontalHeaderLabels(
            ["Site", "Status", "Reasons", "Overlap", "MetalFrac", "HasMask", "HasROI"]
        )
        self.qc_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.qc_table.verticalHeader().setVisible(False)

        row = QWidget()
        hb = QHBoxLayout(row)
        hb.setContentsMargins(0, 0, 0, 0)
        hb.setSpacing(6)

        btn_run = QPushButton("Run QC Now")
        btn_run.setMinimumHeight(44)
        btn_run.clicked.connect(self.run_qc)
        btn_export = QPushButton("Export QC CSV")
        btn_export.setMinimumHeight(44)
        btn_export.clicked.connect(self.export_qc_csv)
        hb.addWidget(btn_run)
        hb.addWidget(btn_export)
        hb.addStretch(1)

        # Overlay preview controls
        ov = QGroupBox("Overlay QC View")
        ovl = QVBoxLayout(ov)
        ovl.setContentsMargins(8, 8, 8, 8)
        ovl.setSpacing(6)

        self.cb_qc_overlay = QCheckBox("Show overlay on images")
        self.cb_qc_overlay.setChecked(True)
        self.cb_qc_overlay.toggled.connect(self._on_qc_overlay_toggle)

        row2 = QWidget()
        hb2 = QHBoxLayout(row2)
        hb2.setContentsMargins(0, 0, 0, 0)
        hb2.setSpacing(6)

        self.cmb_qc_overlay_site = QComboBox()
        self.cmb_qc_overlay_site.addItem("Selected ROI")
        for s in self.site_names:
            self.cmb_qc_overlay_site.addItem(str(s))
        self.cmb_qc_overlay_site.currentTextChanged.connect(self._on_qc_overlay_site_changed)

        btn_focus = QPushButton("Focus Site ROI")
        btn_focus.clicked.connect(self._focus_overlay_site_roi)

        self.lbl_qc_overlay_info = QLabel("Overlay: -")
        self.lbl_qc_overlay_info.setWordWrap(True)

        hb2.addWidget(QLabel("Site:"))
        hb2.addWidget(self.cmb_qc_overlay_site, 1)
        hb2.addWidget(btn_focus)

        ovl.addWidget(self.cb_qc_overlay)
        ovl.addWidget(row2)
        ovl.addWidget(self.lbl_qc_overlay_info)

        v.addWidget(header)
        v.addWidget(row)
        v.addWidget(ov)
        v.addWidget(self.qc_table)
        v.addStretch(1)
        return w

    def _tab_composite(self):
        w = QWidget()
        outer = QVBoxLayout(w)
        outer.setSpacing(12)
        outer.setContentsMargins(8, 8, 8, 8)

        preview_box = QGroupBox("Areal 2D DXA/AP Preview")
        pv = QVBoxLayout(preview_box)
        pv.setContentsMargins(10, 10, 10, 10)
        pv.setSpacing(8)

        self.lbl_dxa_preview_info = QLabel(
            "This panel shows the labelled 2D areal DXA/AP image generated from the current CT ROI workflow. "
            "Use the editor button to move or resize projected ROIs."
        )
        self.lbl_dxa_preview_info.setWordWrap(True)
        pv.addWidget(self.lbl_dxa_preview_info)

        self.lbl_dxa_preview = QLabel("Load CT and place L1-L4 / FN_L / FN_R ROIs to generate the areal DXA/AP preview.")
        self.lbl_dxa_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_dxa_preview.setMinimumHeight(420)
        self.lbl_dxa_preview.setStyleSheet("border: 1px solid #444; background-color: black;")
        pv.addWidget(self.lbl_dxa_preview, 1)

        row_preview = QWidget()
        hb_preview = QHBoxLayout(row_preview)
        hb_preview.setContentsMargins(0, 0, 0, 0)
        hb_preview.setSpacing(8)
        btn_refresh_preview = QPushButton("Refresh Labelled DXA/AP Preview")
        btn_refresh_preview.setMinimumHeight(34)
        btn_refresh_preview.clicked.connect(self.refresh_dxa_tab_preview)
        btn_dxa_proj = QPushButton("Open Editable Areal 2D DXA/AP")
        btn_dxa_proj.setMinimumHeight(34)
        btn_dxa_proj.clicked.connect(self.open_dxa_projection_dialog)
        hb_preview.addWidget(btn_refresh_preview)
        hb_preview.addWidget(btn_dxa_proj)
        pv.addWidget(row_preview)

        g = QGroupBox("Areal 2D DXA/AP Controls")
        f = QFormLayout()
        f.setContentsMargins(10, 8, 10, 8)
        self.combo_rule = QComboBox(); self.combo_rule.addItems(["None", "BMD>1SD", "T>1.0"])
        self.combo_rule.currentTextChanged.connect(lambda t: setattr(self, "exclusion_rule", t))
        self.chk_ap_bone_only = QCheckBox("Bone-only + zero-clip"); self.chk_ap_bone_only.setChecked(self.ap_bone_only_zero_clip); self.chk_ap_bone_only.toggled.connect(lambda b: setattr(self,'ap_bone_only_zero_clip',bool(b)))
        self.chk_ap_auto = QCheckBox("Auto-pick HU min"); self.chk_ap_auto.setChecked(self.ap_auto_hu_min); self.chk_ap_auto.toggled.connect(lambda b: setattr(self,'ap_auto_hu_min',bool(b)))
        self.chk_ap_overlay = QCheckBox("Show AP thickness overlay"); self.chk_ap_overlay.setChecked(self.ap_show_thickness_overlay); self.chk_ap_overlay.toggled.connect(lambda b: setattr(self,'ap_show_thickness_overlay',bool(b)))
        self.chk_ap_lock = QCheckBox("Lock to vertebral body (snap center)"); self.chk_ap_lock.setChecked(self.ap_lock_to_vertebral_body); self.chk_ap_lock.toggled.connect(lambda b: setattr(self,'ap_lock_to_vertebral_body',bool(b)))
        self.sp_ap_min = QDoubleSpinBox(); self.sp_ap_min.setRange(-1000,3000); self.sp_ap_min.setValue(self.ap_hu_min); self.sp_ap_min.valueChanged.connect(lambda v: setattr(self,'ap_hu_min',float(v)))
        self.sp_ap_max = QDoubleSpinBox(); self.sp_ap_max.setRange(-1000,4000); self.sp_ap_max.setValue(self.ap_hu_max); self.sp_ap_max.valueChanged.connect(lambda v: setattr(self,'ap_hu_max',float(v)))
        btn_comp = QPushButton("Compute L1–L4 Composite"); btn_comp.setMinimumHeight(34); btn_comp.clicked.connect(self.compute_composite)
        btn_export = QPushButton("Export CSV"); btn_export.setMinimumHeight(34); btn_export.clicked.connect(self.export_csv)
        f.addRow("Exclusion rule", self.combo_rule)
        f.addRow(btn_comp)
        f.addRow(self.chk_ap_bone_only)
        f.addRow(self.chk_ap_auto)
        f.addRow(self.chk_ap_overlay)
        f.addRow("AP HU min", self.sp_ap_min)
        f.addRow("AP HU max", self.sp_ap_max)
        f.addRow(self.chk_ap_lock)
        f.addRow(btn_export)
        g.setLayout(f)

        outer.addWidget(preview_box, 1)
        outer.addWidget(g)
        outer.addStretch(1)
        return w

    def _tab_patient(self):
        w = QWidget()
        outer = QVBoxLayout(w)
        outer.setSpacing(12)
        outer.setContentsMargins(8, 8, 8, 8)
        g = QGroupBox("Patient / Report", w)
        f = QFormLayout()
        f.setContentsMargins(10, 8, 10, 8)
        self.age = QSpinBox(); self.age.setRange(1,120); self.age.setValue(int(self.patient_age)); self.age.valueChanged.connect(self._set_age_all)
        row = QWidget(); h = QHBoxLayout(row); h.setContentsMargins(0,0,0,0); h.setSpacing(8)
        self.sexF = QRadioButton("Female"); self.sexM = QRadioButton("Male"); self.sexF.setChecked(True)
        self.sexF.toggled.connect(lambda: self._set_sex_all('F' if self.sexF.isChecked() else 'M'))
        h.addWidget(self.sexF); h.addWidget(self.sexM); h.addStretch(1)
        self.chk_contrast_mode_tab = QCheckBox("Contrast PET/CT mode"); self.chk_contrast_mode_tab.setChecked(self.contrast_mode); self.chk_contrast_mode_tab.toggled.connect(self._set_contrast_mode)
        btn_report = QPushButton("Generate Report"); btn_report.clicked.connect(self.report)
        btn_results = QPushButton("Open Results Window"); btn_results.clicked.connect(self.open_results_window)
        btn_dxa_direct = QPushButton("Open Areal 2D DXA/AP"); btn_dxa_direct.clicked.connect(self.open_dxa_projection_dialog)
        f.addRow("Age", self.age)
        f.addRow("Sex", row)
        f.addRow(self.chk_contrast_mode_tab)
        f.addRow(btn_report)
        f.addRow(btn_results)
        f.addRow(btn_dxa_direct)
        g.setLayout(f)
        outer.addWidget(g)
        outer.addStretch(1)
        return w

    def _set_age_all(self, v):
        self.patient_age = int(v)
        for nm in ("age_left","age"):
            w=getattr(self,nm,None)
            if w is not None and w.value()!=int(v):
                w.blockSignals(True); w.setValue(int(v)); w.blockSignals(False)

    def _set_sex_all(self, s):
        self.patient_sex = 'F' if str(s).upper().startswith('F') else 'M'
        for nm,val in (("sexF_left", self.patient_sex=='F'),("sexM_left", self.patient_sex=='M'),("sexF", self.patient_sex=='F'),("sexM", self.patient_sex=='M')):
            w=getattr(self,nm,None)
            if w is not None and w.isChecked()!=val:
                w.blockSignals(True); w.setChecked(val); w.blockSignals(False)

    def _set_contrast_mode(self, b):
        self.contrast_mode = bool(b)
        self._set_preset('contrast' if self.contrast_mode else 'native')
        for nm in ('chk_contrast_mode','chk_contrast_mode_tab'):
            w=getattr(self,nm,None)
            if w is not None and w.isChecked()!=bool(b):
                w.blockSignals(True); w.setChecked(bool(b)); w.blockSignals(False)

    def _current_crosshair_imgpos(self, view='axial'):
        x,y,z = self.crosshair
        if self.ct_data is None: return None, None
        if view=='axial': return QPointF(float(x), float(y)), z if not self.axial_invert_z else (self.ct_data.shape[2]-1-z)
        if view=='coronal': return QPointF(float(x), float(z)), y
        return QPointF(float(y), float(z)), x

    def _find_roi_by_exact_tag(self, tag: str):
        _c, roi = self._find_latest_roi_by_exact_tag_anywhere(tag)
        return roi

    def _required_target_labels(self):
        return ["L1", "L2", "L3", "L4", "FN_L", "FN_R"]


    def _get_target_spec(self, tag: str):
        return dict(self.target_roi_specs.get(str(tag), {})) if hasattr(self, "target_roi_specs") else {}

    def _set_target_spec(self, tag: str, x: int, y: int, z: int, radius: int, status: str = "locked"):
        if not hasattr(self, "target_roi_specs"):
            self.target_roi_specs = {}
        self.target_roi_specs[str(tag)] = {
            "x": int(x), "y": int(y), "z": int(z),
            "radius": int(max(5, radius)),
            "status": str(status),
        }

    def _clear_target_spec(self, tag: str):
        if hasattr(self, "target_roi_specs"):
            self.target_roi_specs.pop(str(tag), None)

    def _sync_target_roi_pair_from_spec(self, tag: str, select_view: str = "coronal"):
        spec = self._get_target_spec(tag)
        if not spec or self.ct_data is None:
            return None, None
        x = int(spec.get("x", 0)); y = int(spec.get("y", 0)); z = int(spec.get("z", 0))
        radius = int(max(5, spec.get("radius", 18)))
        status = str(spec.get("status", "locked"))
        self._remove_rois_by_tag(tag, target_views=["axial", "coronal"])
        ax_pt, ax_sl = self._volume_to_canvas_center("axial", x, y, z)
        cor_pt, cor_sl = self._volume_to_canvas_center("coronal", x, y, z)

        ax_roi = SphericalROI(ax_pt.x(), ax_pt.y(), radius, tag=tag, slice_index=int(ax_sl))
        ax_roi.locked_to_slice = True
        ax_roi.follow_on_scroll = False
        ax_roi.volume_xyz = (x, y, z)
        ax_roi.source_view = "axial"
        ax_roi.auto_status = status
        self.canvas_axial.spherical_rois.append(ax_roi)

        cor_roi = SphericalROI(cor_pt.x(), cor_pt.y(), radius, tag=tag, slice_index=int(cor_sl))
        cor_roi.locked_to_slice = True
        cor_roi.follow_on_scroll = False
        cor_roi.volume_xyz = (x, y, z)
        cor_roi.source_view = "coronal"
        cor_roi.auto_status = status
        self.canvas_coronal.spherical_rois.append(cor_roi)

        if select_view == "axial":
            self.canvas_axial.selected_roi = ax_roi
            self.canvas_coronal.selected_roi = None
            self._set_active_canvas(self.canvas_axial)
        else:
            self.canvas_coronal.selected_roi = cor_roi
            self.canvas_axial.selected_roi = None
            self._set_active_canvas(self.canvas_coronal)
        self._sync_crosshairs(x, y, z)
        return ax_roi, cor_roi

    def _iter_all_roi_pairs(self):
        for c in self._canvases():
            if c is None:
                continue
            for roi in getattr(c, "spherical_rois", []):
                yield c, roi

    def _find_latest_roi_by_exact_tag_anywhere(self, tag: str):
        last = (None, None)
        for c, roi in self._iter_all_roi_pairs():
            if getattr(roi, "tag", "") == str(tag):
                last = (c, roi)
        return last



    def _mark_target_status(self, tag: str, status: str):
        spec = self._get_target_spec(tag)
        if spec:
            self._set_target_spec(tag, int(spec.get("x",0)), int(spec.get("y",0)), int(spec.get("z",0)), int(spec.get("radius",18)), status=str(status))
        for _c, roi in self._iter_all_roi_pairs():
            if getattr(roi, "tag", "") == str(tag):
                roi.auto_status = str(status)
    def _estimate_target_volume_center(self, tag: str):
        if self.ct_data is None:
            return None
        sx, sy, sz = self.ct_data.shape
        mask = None
        if tag in ("L1", "L2", "L3", "L4"):
            mask = self.vertebra_masks.get(tag)
        elif tag == "FN_L":
            mask = self.extra_masks.get("FEMUR_L")
        elif tag == "FN_R":
            mask = self.extra_masks.get("FEMUR_R")
        if mask is not None and getattr(mask, "shape", None) == self.ct_data.shape:
            idx = np.argwhere(mask > 0)
            if idx.size > 0:
                if tag in ("FN_L", "FN_R"):
                    # Bias neck placeholders toward the proximal femur rather than the shaft.
                    x_min = int(np.min(idx[:, 0]))
                    x_max = int(np.max(idx[:, 0]))
                    cut = x_min + max(3, int(round((x_max - x_min) * 0.30)))
                    idx2 = idx[idx[:, 0] <= cut]
                    if idx2.size > 0:
                        idx = idx2
                return int(np.median(idx[:, 0])), int(np.median(idx[:, 1])), int(np.median(idx[:, 2]))
        ch = getattr(self, "crosshair", None)
        if ch and len(ch) == 3:
            return int(ch[0]), int(ch[1]), int(ch[2])
        return sx // 2, sy // 2, sz // 2

    def _ensure_required_roi_placeholders(self):
        if self.ct_data is None:
            return []
        created = []
        base_r = int(getattr(getattr(self, "sp_cal_radius", None), "value", lambda: 18)())
        for tag in self._required_target_labels():
            if self._get_target_spec(tag):
                continue
            coords = self._estimate_target_volume_center(tag)
            if coords is None:
                continue
            x, y, z = [int(v) for v in coords]
            self._set_target_spec(tag, x, y, z, radius=base_r, status="placeholder")
            self._sync_target_roi_pair_from_spec(tag, select_view="coronal")
            created.append(tag)
        self._update_roi_target_status()
        return created

    def _update_roi_target_status(self):
        if not hasattr(self, "lbl_roi_targets"):
            return
        lines = []
        for tag in self._required_target_labels():
            spec = self._get_target_spec(tag)
            if not spec:
                lines.append(f"{tag:<4} : MISSING")
                continue
            x, y, z = int(spec.get("x", 0)), int(spec.get("y", 0)), int(spec.get("z", 0))
            status_txt = str(spec.get("status", "")).upper() or "LOCKED"
            lines.append(f"{tag:<4} : {status_txt:<11} XYZ=({x},{y},{z})")
        self.lbl_roi_targets.setText("\n".join(lines))
    def _canvas_roi_to_volume(self, canvas, roi):
        if canvas is None or roi is None or self.ct_data is None:
            return None
        sx, sy, sz = self.ct_data.shape
        sl = getattr(roi, "slice_index", None)
        if sl is None:
            sl = getattr(canvas, "current_slice", 0)
        if canvas.view == "axial":
            x = int(round(roi.center.x()))
            y = int(round(roi.center.y()))
            z = int(self._axial_disp_to_data_z(int(sl)))
        elif canvas.view == "coronal":
            x = int(round(roi.center.x()))
            y = int(sl)
            z = int(round(roi.center.y()))
        else:
            x = int(sl)
            y = int(round(roi.center.x()))
            z = int(round(roi.center.y()))
        x = int(np.clip(x, 0, sx - 1))
        y = int(np.clip(y, 0, sy - 1))
        z = int(np.clip(z, 0, sz - 1))
        return x, y, z

    def _volume_to_canvas_center(self, view: str, x: int, y: int, z: int):
        if view == "axial":
            return QPointF(float(x), float(y)), int(self._axial_data_to_disp_z(z))
        if view == "coronal":
            return QPointF(float(x), float(z)), int(y)
        return QPointF(float(y), float(z)), int(x)

    def _remove_rois_by_tag(self, tag: str, target_views=None):
        target_views = set(target_views or ["axial", "coronal"])
        for c in self._canvases():
            if c is None or c.view not in target_views:
                continue
            keep = []
            removed_selected = False
            for roi in getattr(c, "spherical_rois", []):
                if getattr(roi, "tag", "") == str(tag):
                    if c.selected_roi is roi:
                        removed_selected = True
                    continue
                keep.append(roi)
            c.spherical_rois = keep
            if removed_selected:
                c.selected_roi = None
                c.hover_roi = None
            c.update()


    def _create_target_roi_pair(self, tag: str, x: int, y: int, z: int, radius: int = 18, select_view: str = "coronal", status: str = "locked"):
        if self.ct_data is None:
            return None, None
        self._set_target_spec(tag, x, y, z, radius=radius, status=status)
        ax_roi, cor_roi = self._sync_target_roi_pair_from_spec(tag, select_view=select_view)
        self._update_roi_target_status()
        return ax_roi, cor_roi

    def _get_target_spec_as_axial_roi(self, tag: str):
        spec = self._get_target_spec(tag)
        if not spec or self.ct_data is None:
            return None
        x = int(spec.get("x", 0))
        y = int(spec.get("y", 0))
        z = int(spec.get("z", 0))
        r = int(max(5, spec.get("radius", 18)))
        ax_pt, ax_sl = self._volume_to_canvas_center("axial", x, y, z)
        roi = SphericalROI(ax_pt.x(), ax_pt.y(), r, tag=str(tag), slice_index=int(ax_sl))
        roi.locked_to_slice = True
        roi.follow_on_scroll = False
        roi.volume_xyz = (x, y, z)
        roi.source_view = "axial"
        roi.auto_status = str(spec.get("status", "locked"))
        return roi

    def _target_based_projection_range(self, tags=None, pad: int = 6):
        if self.ct_data is None:
            return None, None
        tags = list(tags or self._required_target_labels())
        z_vals = []
        for tag in tags:
            spec = self._get_target_spec(tag)
            if spec:
                z_vals.append(int(spec.get("z", 0)))
        if not z_vals:
            return None, None
        zmin = max(0, min(z_vals) - int(pad))
        zmax = min(self.ct_data.shape[2] - 1, max(z_vals) + int(pad))
        return int(zmin), int(zmax)

    def _build_dxa_display_payload(self):
        if self.ct_data is None:
            return None, None
        return self._make_ap_projection(0, int(self.ct_data.shape[2] - 1))

    def _build_dxa_quant_payload(self):
        if self.ct_data is None:
            return None, None
        zmin, zmax = self._target_based_projection_range()
        return self._make_ap_projection(zmin, zmax)

    def _proj_calibration_for_site(self, tag: str):
        tag = str(tag)
        if tag in ("L1", "L2", "L3", "L4"):
            return float(getattr(self, "proj_spine_slope", 1.0)), float(getattr(self, "proj_spine_intercept", 0.0))
        return float(getattr(self, "proj_fn_slope", 1.0)), float(getattr(self, "proj_fn_intercept", 0.0))
    def _lock_target_roi_pair_from_selected(self, tag: str, canvas, roi):
        coords = self._canvas_roi_to_volume(canvas, roi)
        if coords is None:
            return None, None
        x, y, z = coords
        radius = int(getattr(roi, "radius", 18))
        self._set_target_spec(tag, x, y, z, radius=radius, status="locked")
        ax_roi, cor_roi = self._sync_target_roi_pair_from_spec(tag, select_view=getattr(canvas, "view", "coronal"))
        self._update_roi_target_status()
        return ax_roi, cor_roi

    def _lock_roi_to_current_slice(self, roi, canvas=None, tag: str = None):
        c = canvas or self._active_canvas() or self.canvas_axial
        if roi is None or c is None:
            return
        if tag is not None:
            roi.tag = str(tag)
        roi.slice_index = int(c.current_slice)
        roi.locked_to_slice = True
        roi.follow_on_scroll = False
        try:
            self._update_roi_target_status()
        except Exception:
            pass

    def new_update_roi_at_crosshair(self):
        c = self._active_canvas() or self.canvas_coronal or self.canvas_axial
        tag = self.roi_label_combo.currentText() if hasattr(self, "roi_label_combo") else "ROI"
        pt, sl = self._current_crosshair_imgpos(c.view)
        if pt is None:
            return
        rad = int(getattr(getattr(self, "sp_cal_radius", None), "value", lambda: 18)())
        if tag in self._required_target_labels() and self.ct_data is not None:
            if c.view == "axial":
                x, y, z = int(round(pt.x())), int(round(pt.y())), int(self._axial_disp_to_data_z(int(sl)))
            elif c.view == "coronal":
                x, y, z = int(round(pt.x())), int(sl), int(round(pt.y()))
            else:
                x, y, z = int(sl), int(round(pt.x())), int(round(pt.y()))
            self._create_target_roi_pair(tag, x, y, z, radius=rad, select_view=c.view if c.view in ("axial", "coronal") else "coronal", status="manual")
            self.results.append(f"Created locked paired ROI '{tag}' at crosshair from {c.view} view.")
        else:
            roi = SphericalROI(pt.x(), pt.y(), rad, tag=tag, slice_index=int(sl))
            roi.locked_to_slice = True
            roi.follow_on_scroll = False
            c.spherical_rois.append(roi)
            c.selected_roi = roi
            c.hover_roi = roi
            c.update()
            self.results.append(f"Created new locked ROI '{tag}' at crosshair on {c.view} slice {int(sl)}.")
        self._update_roi_target_status()

    def lock_selected_roi_here(self):
        c = self._active_canvas() or self.canvas_coronal or self.canvas_axial
        roi = c.selected_roi if c is not None else None
        if roi is None:
            self.results.append("No ROI selected to lock.")
            return

        existing_tag = str(getattr(roi, "tag", "ROI") or "ROI")
        combo_tag = self.roi_label_combo.currentText() if hasattr(self, "roi_label_combo") else existing_tag

        # Hard rule: locking must not silently relabel an already tagged target ROI.
        # The combo-box label is used only for generic/unassigned ROIs.
        if existing_tag in self._required_target_labels():
            tag = existing_tag
        else:
            tag = combo_tag or existing_tag

        if tag in self._required_target_labels():
            self._lock_target_roi_pair_from_selected(tag, c, roi)
            if existing_tag != tag and existing_tag not in self._required_target_labels():
                self.results.append(
                    f"Locked selected ROI as target '{tag}' at the selected {c.view} position."
                )
            else:
                self.results.append(
                    f"Locked target ROI '{tag}' at the selected {c.view} position without relabeling."
                )
        else:
            self._lock_roi_to_current_slice(roi, c, tag=tag)
            c.update()
            self.results.append(
                f"Locked selected ROI as '{tag}' on {c.view} slice {int(c.current_slice)} without moving it."
            )
        self._update_roi_target_status()

    def snap_selected_roi_to_crosshair(self):
        c = self._active_canvas() or self.canvas_coronal or self.canvas_axial
        roi = c.selected_roi
        if roi is None:
            self.results.append("No ROI selected to snap to crosshair.")
            return
        pt, sl = self._current_crosshair_imgpos(c.view)
        if pt is None:
            return
        roi.center = QPointF(pt)
        roi.slice_index = int(sl)
        roi.locked_to_slice = True
        roi.follow_on_scroll = False
        roi.locked_to_slice = True
        c.update()
        self.results.append(
            f"Snapped ROI '{getattr(roi, 'tag', 'ROI')}' to the current crosshair on {c.view} slice {int(sl)}."
        )
        self._update_roi_target_status()

    def delete_selected_roi(self):

        c=self._active_canvas() or self.canvas_axial
        if c and c.selected_roi in c.spherical_rois:
            tag = getattr(c.selected_roi, "tag", "")
            if tag in self._required_target_labels():
                self._clear_target_spec(tag)
                self._remove_rois_by_tag(tag, target_views=["axial", "coronal"])
                c.selected_roi = None
                c.hover_roi = None
            else:
                c.spherical_rois.remove(c.selected_roi)
                c.selected_roi=None
            c.update()
            self._update_roi_target_status()

    def place_calibration_roi_center(self, tag):
        c = self.canvas_axial
        pt, sl = self._current_crosshair_imgpos('axial')
        if pt is None: return
        rad = int(self.sp_cal_radius.value()) if hasattr(self,'sp_cal_radius') else 18
        roi = SphericalROI(pt.x(), pt.y(), rad, tag=tag, slice_index=int(sl))
        c.spherical_rois.append(roi); c.selected_roi=roi; c.update()
        self._update_roi_target_status()
        self.results.append(f'Tagged ROI as {tag} on slice {sl}.')

    def validate_calibration(self):
        c_fat, roi_fat = self._find_roi_by_tag('FAT')
        c_m, roi_m = self._find_roi_by_tag('MUSCLE')
        if roi_m is None:
            c_m, roi_m = self._find_roi_by_tag('PSOAS_L')
            if roi_m is None:
                c_m, roi_m = self._find_roi_by_tag('PSOAS_R')
        msgs=[]
        ok=True
        for name, cc, rr in [('FAT',c_fat,roi_fat),('MUSCLE',c_m,roi_m)]:
            if rr is None: msgs.append(f'{name}: missing'); ok=False; continue
            res=self._analyze_roi(cc, rr, site=name)
            if not res: msgs.append(f'{name}: failed'); ok=False; continue
            msgs.append(f"{name} HU={res['mean_hu']:.1f}")
        msgs.append(f'Effective calibration: {self.cal_slope_eff:.4f}*HU + {self.cal_intercept_eff:.2f}')
        QMessageBox.information(self, 'Validate Calibration', '\n'.join(msgs))

    def _fmt_result_value(self, value, decimals: int = 1, empty: str = ""):
        if value is None:
            return empty
        try:
            v = float(value)
            if not np.isfinite(v):
                return empty
            return f"{v:.{int(decimals)}f}"
        except Exception:
            return str(value)

    def _style_publication_table(self, table: QTableWidget):
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        table.setWordWrap(False)
        table.setCornerButtonEnabled(False)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        table.setStyleSheet(
            """
            QTableWidget {
                gridline-color: #4a4a4a;
                alternate-background-color: #232323;
                background: #161616;
                color: #f2f2f2;
                font-size: 9.8pt;
            }
            QHeaderView::section {
                background: #2b2b2b;
                color: #ffffff;
                padding: 6px;
                border: 1px solid #444;
                font-weight: 600;
                font-size: 9.8pt;
            }
            """
        )

    def _build_ct_results_table(self) -> QTableWidget:
        cols = [
            "Site", "Slice", "Mean HU", "SD HU", "vBMD (mg/cm³)", "T-score", "Z-score",
            "Voxels", "Mask", "QC", "QC Reasons",
        ]
        rows = [(site, self.vertebral.get(site)) for site in self.site_names if self.vertebral.get(site)]
        table = QTableWidget(len(rows), len(cols))
        table.setHorizontalHeaderLabels(cols)
        self._style_publication_table(table)

        for row_idx, (site, res) in enumerate(rows):
            values = [
                site,
                str(res.get("slice_index", "")),
                self._fmt_result_value(res.get("mean_hu"), 1),
                self._fmt_result_value(res.get("std_hu"), 1),
                self._fmt_result_value(res.get("bmd"), 1),
                self._fmt_result_value(res.get("t_score"), 2),
                self._fmt_result_value(res.get("z_score"), 2),
                str(res.get("nvox", "")),
                str(res.get("mask_status", "")),
                str(res.get("qc_status", "")),
                "; ".join(res.get("qc_reasons", []) or []),
            ]
            for col_idx, value in enumerate(values):
                table.setItem(row_idx, col_idx, QTableWidgetItem(value))

        table.resizeColumnsToContents()
        return table

    def _build_dxa_results_table(self) -> QTableWidget:
        cols = [
            "Region", "Type", "Mean HU", "Pseudo-aBMD", "T-score", "Z-score",
            "Bone Voxels", "All Voxels",
        ]
        dxa = self.dxa_proj_last or {}
        rows = []

        for res in dxa.get("sites", []) or []:
            rows.append(
                [
                    str(res.get("site", "")),
                    "Per-ROI",
                    self._fmt_result_value(res.get("mean_hu"), 1),
                    self._fmt_result_value(res.get("bmd"), 1),
                    self._fmt_result_value(res.get("t_score"), 2),
                    self._fmt_result_value(res.get("z_score"), 2),
                    str(res.get("nvox", "")),
                    str(res.get("nvox_all", "")),
                ]
            )

        comp = dxa.get("composite", {}) or {}
        if comp.get("spine") is not None:
            res = comp["spine"]
            rows.append(
                [
                    "L1-L4 Composite",
                    "Composite",
                    self._fmt_result_value(res.get("mean_hu"), 1),
                    self._fmt_result_value(res.get("bmd"), 1),
                    self._fmt_result_value(res.get("t_score"), 2),
                    self._fmt_result_value(res.get("z_score"), 2),
                    "",
                    "",
                ]
            )
        if comp.get("hips_mean") is not None:
            res = comp["hips_mean"]
            rows.append(
                [
                    "Femoral Neck Mean",
                    "Composite",
                    self._fmt_result_value(res.get("mean_hu"), 1),
                    self._fmt_result_value(res.get("bmd"), 1),
                    self._fmt_result_value(res.get("t_score"), 2),
                    self._fmt_result_value(res.get("z_score"), 2),
                    "",
                    "",
                ]
            )

        table = QTableWidget(len(rows), len(cols))
        table.setHorizontalHeaderLabels(cols)
        self._style_publication_table(table)

        for row_idx, row in enumerate(rows):
            for col_idx, value in enumerate(row):
                table.setItem(row_idx, col_idx, QTableWidgetItem(value))

        table.resizeColumnsToContents()
        return table


    def _site_roi_specs_for_publication(self):
        specs = []
        if self.ct_data is None or self.canvas_axial is None:
            return specs
        roi_map = {}
        for rr in self.canvas_axial.spherical_rois:
            tag = getattr(rr, "tag", "")
            if tag in self.site_names:
                roi_map[tag] = rr
        for tag in self.site_names:
            roi = roi_map.get(tag)
            if roi is None:
                missing.append(name)
                continue
            sl = getattr(roi, "slice_index", None)
            if sl is None:
                sl = int(self.crosshair[2]) if self.ct_data is not None else 0
            res = self.vertebral.get(tag) or {}
            specs.append(
                {
                    "tag": tag,
                    "roi": roi,
                    "slice_index": int(sl),
                    "result": res,
                }
            )
        return specs

    def _get_publication_slice(self, view: str, idx: int) -> np.ndarray:
        if self.ct_data is None:
            return np.zeros((1, 1), np.float32)
        data = self.ct_data
        if view == "axial":
            idx = int(np.clip(idx, 0, data.shape[2] - 1))
            s = data[:, :, idx].T
            if getattr(self, "axial_flip_ud", False):
                s = np.flipud(s)
        elif view == "sagittal":
            idx = int(np.clip(idx, 0, data.shape[0] - 1))
            s = data[idx, :, :].T
            if getattr(self, "sagittal_flip_ud", False):
                s = np.flipud(s)
        else:
            idx = int(np.clip(idx, 0, data.shape[1] - 1))
            s = data[:, idx, :].T
            if getattr(self, "coronal_flip_ud", False):
                s = np.flipud(s)
        return np.asarray(s, np.float32)

    def _draw_publication_label(self, painter: QPainter, box: QRectF, text: str):
        painter.save()
        painter.setPen(QPen(QColor(35, 35, 35), 1))
        painter.setBrush(QColor(255, 255, 255, 235))
        painter.drawRoundedRect(box, 8, 8)
        painter.setPen(QPen(QColor(20, 20, 20), 1))
        painter.drawText(box.adjusted(8, 6, -8, -6), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)
        painter.restore()

    def _make_publication_roi_montage(self, view: str = "axial") -> QPixmap:
        specs = self._site_roi_specs_for_publication()
        if self.ct_data is None or not specs:
            return QPixmap()

        cols = 3
        rows = int(math.ceil(len(specs) / cols))
        panel_w = 520
        panel_h = 420
        margin = 30
        header_h = 70
        footer_h = 22
        W = cols * panel_w + (cols + 1) * margin
        H = rows * panel_h + (rows + 1) * margin + header_h

        pixmap = QPixmap(W, H)
        pixmap.fill(QColor(245, 245, 245))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        painter.setPen(QPen(QColor(20, 20, 20), 1))
        painter.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title = "Axial CT ROI Montage" if view == "axial" else "Sagittal CT ROI Montage"
        painter.drawText(QRectF(margin, 18, W - 2 * margin, 28), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, title)
        painter.setFont(QFont("Arial", 10))
        painter.setPen(QPen(QColor(70, 70, 70), 1))
        subtitle = "ROIs: L1-L4 vertebrae and bilateral femoral necks. Publication export with ROI circles and measured values."
        painter.drawText(QRectF(margin, 44, W - 2 * margin, 20), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, subtitle)

        for i, spec in enumerate(specs):
            row = i // cols
            col = i % cols
            x0 = margin + col * (panel_w + margin)
            y0 = header_h + margin + row * (panel_h + margin)

            panel_rect = QRectF(x0, y0, panel_w, panel_h)
            painter.setPen(QPen(QColor(170, 170, 170), 1))
            painter.setBrush(QColor(255, 255, 255))
            painter.drawRoundedRect(panel_rect, 10, 10)

            roi = spec["roi"]
            tag = spec["tag"]
            sl = int(spec["slice_index"])

            if view == "axial":
                img = self._get_publication_slice("axial", sl)
                cx = float(roi.center.x())
                cy = float(roi.center.y())
                rr = float(roi.radius)
                panel_title = f"{tag}  |  axial slice {sl}"
            else:
                x_idx = int(round(float(roi.center.x())))
                img = self._get_publication_slice("sagittal", x_idx)
                cx = float(roi.center.y())
                cy = float(sl)
                rr = float(roi.radius)
                panel_title = f"{tag}  |  sagittal x {x_idx}"

            c = float(self.ct_win_center)
            wv = max(float(self.ct_win_width), 1.0)
            vmin, vmax = c - wv / 2.0, c + wv / 2.0
            img8 = np.clip((img - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
            img8 = (img8 * 255.0 + 0.5).astype(np.uint8)
            h, w = img8.shape
            qimg = QImage(img8.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
            pm = QPixmap.fromImage(qimg)

            inner_rect = QRectF(x0 + 16, y0 + 42, panel_w - 32, panel_h - 86)
            painter.drawPixmap(inner_rect, pm, QRectF(0, 0, w, h))

            sx = inner_rect.width() / max(w, 1)
            sy = inner_rect.height() / max(h, 1)
            px = inner_rect.left() + cx * sx
            py = inner_rect.top() + cy * sy
            pr = max(6.0, rr * (sx + sy) * 0.5)

            painter.setPen(QPen(QColor(240, 210, 0), 3))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(px, py), pr, pr)

            painter.setFont(QFont("Arial", 11, QFont.Weight.Bold))
            painter.setPen(QPen(QColor(20, 20, 20), 1))
            painter.drawText(QRectF(x0 + 16, y0 + 10, panel_w - 32, 22), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, panel_title)

            res = spec.get("result") or {}
            label_lines = [tag]
            if res:
                mean_hu = res.get("mean_hu")
                bmd = res.get("bmd")
                t_score = res.get("t_score")
                z_score = res.get("z_score")
                who = res.get("who_abs") or res.get("who") or ""
                if mean_hu is not None:
                    label_lines.append(f"HU {float(mean_hu):.1f}")
                if bmd is not None:
                    label_lines.append(f"vBMD {float(bmd):.1f}")
                if t_score is not None and z_score is not None:
                    label_lines.append(f"T {float(t_score):.2f} | Z {float(z_score):.2f}")
                if who:
                    label_lines.append(str(who))
            else:
                label_lines.append("ROI placed")
            label_text = "\n".join(label_lines)

            metrics = painter.fontMetrics()
            lines = label_text.splitlines()
            box_w = max(metrics.horizontalAdvance(line) for line in lines) + 18
            box_h = max(46, len(lines) * (metrics.height() + 1) + 10)

            candidates = [
                QRectF(px + pr + 14, py - box_h * 0.5, box_w, box_h),
                QRectF(px - pr - box_w - 14, py - box_h * 0.5, box_w, box_h),
                QRectF(px - box_w * 0.5, py - pr - box_h - 14, box_w, box_h),
                QRectF(px - box_w * 0.5, py + pr + 14, box_w, box_h),
            ]
            safe_rect = QRectF(inner_rect.left() + 4, inner_rect.top() + 4, inner_rect.width() - 8, inner_rect.height() - 8)
            chosen = candidates[0]
            roi_box = QRectF(px - pr, py - pr, pr * 2, pr * 2)
            for cand in candidates:
                if safe_rect.contains(cand) and not cand.intersects(roi_box):
                    chosen = cand
                    break
            self._draw_publication_label(painter, chosen, label_text)

            painter.setPen(QPen(QColor(120, 120, 120), 1, Qt.PenStyle.DashLine))
            cpt = chosen.center()
            painter.drawLine(QPointF(px, py), cpt)

            painter.setPen(QPen(QColor(95, 95, 95), 1))
            painter.setFont(QFont("Arial", 9))
            footer = "Windowed CT display for publication export"
            painter.drawText(QRectF(x0 + 16, y0 + panel_h - footer_h - 6, panel_w - 32, footer_h), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, footer)

        painter.end()
        return pixmap

    def open_publication_roi_figures(self):
        if self.ct_data is None:
            QMessageBox.information(self, "Publication ROI Figures", "Load and analyze a CT study first.")
            return
        specs = self._site_roi_specs_for_publication()
        if not specs:
            QMessageBox.information(self, "Publication ROI Figures", "No tagged ROIs were found for L1-L4 or femoral necks.")
            return
        dlg = PublicationROIFigureDialog(self, self)
        dlg.exec()

    def open_results_window(self):
        ct_rows = [self.vertebral.get(site) for site in self.site_names if self.vertebral.get(site)]
        self._ensure_dxa_projection_results(force=False)
        dxa_rows = list((self.dxa_proj_last or {}).get("sites", []) or [])
        has_spine_comp = bool(((self.dxa_proj_last or {}).get("composite", {}) or {}).get("spine"))
        has_hip_comp = bool(((self.dxa_proj_last or {}).get("composite", {}) or {}).get("hips_mean"))

        if not ct_rows and not dxa_rows and not has_spine_comp and not has_hip_comp:
            QMessageBox.information(self, "Results", "No CT ROI or DXA projection results are available yet.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("QCT and DXA Results")
        dlg.resize(1280, 820)

        outer = QVBoxLayout(dlg)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        title = QLabel("Compact publication-style results tables")
        title.setStyleSheet("font-size: 13pt; font-weight: 600; color: #f5f5f5;")
        outer.addWidget(title)

        subtitle = QLabel(
            "Axial CT ROI results and areal DXA projection results are shown separately in labeled tables."
        )
        subtitle.setStyleSheet("color: #cfcfcf; font-size: 9.8pt;")
        outer.addWidget(subtitle)

        g_ct = QGroupBox("Axial CT ROI Measurements")
        g_ct.setStyleSheet("QGroupBox { font-size: 11pt; font-weight: 600; }")
        v_ct = QVBoxLayout(g_ct)
        v_ct.setContentsMargins(10, 14, 10, 10)
        t_ct = self._build_ct_results_table()
        t_ct.setMinimumHeight(260)
        v_ct.addWidget(t_ct)
        outer.addWidget(g_ct, 1)

        g_dxa = QGroupBox("Areal DXA Projection Measurements")
        g_dxa.setStyleSheet("QGroupBox { font-size: 11pt; font-weight: 600; }")
        v_dxa = QVBoxLayout(g_dxa)
        v_dxa.setContentsMargins(10, 14, 10, 10)
        t_dxa = self._build_dxa_results_table()
        t_dxa.setMinimumHeight(240)
        v_dxa.addWidget(t_dxa)
        outer.addWidget(g_dxa, 1)

        foot = QLabel(
            "CT table lists per-ROI volumetric results. DXA table lists per-projected ROI areal-style results plus composite summaries."
        )
        foot.setStyleSheet("color: #bdbdbd; font-size: 9.5pt;")
        outer.addWidget(foot)

        row_btn = QHBoxLayout()
        row_btn.addStretch(1)
        btn_fig = QPushButton("ROI Figures…")
        btn_fig.setMinimumHeight(36)
        btn_fig.clicked.connect(self.open_publication_roi_figures)
        btn_close = QPushButton("Close")
        btn_close.setMinimumHeight(36)
        btn_close.clicked.connect(dlg.accept)
        row_btn.addWidget(btn_fig)
        row_btn.addWidget(btn_close)
        outer.addLayout(row_btn)

        dlg.exec()

    def _tab_vertebrae(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setSpacing(12)
        v.setContentsMargins(8, 8, 8, 8)
        g = QGroupBox("Measurement Sites (L1–L4 + Femoral Necks)")
        inner = QVBoxLayout()
        inner.setSpacing(6)

        self.vertebra = QComboBox()
        self.vertebra.addItems(self.site_names)

        self.btn_meas_sel = QPushButton("Measure Selected Site")
        self.btn_meas_sel.setMinimumHeight(44)
        self.btn_meas_sel.clicked.connect(self.measure_vertebra_selected)

        row_btns = QWidget()
        hb = QHBoxLayout(row_btns)
        hb.setSpacing(6)
        for name in self.site_names:
            b = QPushButton(f"Measure {name}")
            b.setMinimumHeight(34)
            b.clicked.connect(lambda _=False, n=name: self.measure_specific(n))
            hb.addWidget(b)

        inner.addWidget(self.vertebra)
        inner.addWidget(self.btn_meas_sel)
        inner.addWidget(row_btns)

        btn_show_all = QPushButton("Show All Sites")
        btn_show_all.setMinimumHeight(36)
        btn_show_all.clicked.connect(self.show_all)
        inner.addWidget(btn_show_all)

        g.setLayout(inner)
        v.addWidget(g)
        v.addStretch(1)
        return w

    # ---- Left-panel widget builders ----
    def _files_box(self):
        g = QGroupBox("Load Data (DICOM)")
        f = QFormLayout()
        f.setContentsMargins(10, 8, 10, 8)
        f.setHorizontalSpacing(12)
        f.setVerticalSpacing(8)

        self.btn_load_zip = QPushButton("Load PET/CT DICOM ZIP…")
        self.btn_load_zip.setMinimumHeight(36)
        self.btn_load_zip.clicked.connect(self.load_petct_dicom_zip)

        self.btn_load_ct_folder = QPushButton("Load CT DICOM Folder…")
        self.btn_load_ct_folder.setMinimumHeight(36)
        self.btn_load_ct_folder.clicked.connect(lambda: self.load_dicom_folder(kind="ct"))

        f.addRow(self.btn_load_zip)
        f.addRow(self.btn_load_ct_folder)

        note = QLabel(
            "Tip: if the ZIP/folder has multiple series, a selector will appear."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #8a8a8a;")
        f.addRow(note)

        g.setLayout(f)
        return g

    def _display_box(self):
        g = QGroupBox("Display / Fusion")
        v = QVBoxLayout()
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(8)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.r_ct = QRadioButton("CT")
        self.r_pet = QRadioButton("PET")
        self.r_fuse = QRadioButton("Fusion")
        self.r_ct.setChecked(True)
        for w2 in (self.r_ct, self.r_pet, self.r_fuse):
            w2.toggled.connect(self._update_mode)
            row.addWidget(w2)
        v.addLayout(row)
        form = QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        self.chk_overlay = QCheckBox("Show coords/HU overlay")
        self.chk_overlay.setChecked(self.show_overlay)
        self.chk_overlay.toggled.connect(
            lambda b: (setattr(self, "show_overlay", bool(b)), self._update_all())
        )
        self.sp_wc = QDoubleSpinBox()
        self.sp_wc.setRange(-1000.0, 3000.0)
        self.sp_wc.setDecimals(1)
        self.sp_wc.setValue(self.ct_win_center)
        self.sp_ww = QDoubleSpinBox()
        self.sp_ww.setRange(1.0, 4000.0)
        self.sp_ww.setDecimals(1)
        self.sp_ww.setValue(self.ct_win_width)
        self.sp_wc.valueChanged.connect(
            lambda v2: (setattr(self, "ct_win_center", float(v2)), self._update_all())
        )
        self.sp_ww.valueChanged.connect(
            lambda v2: (setattr(self, "ct_win_width", float(v2)), self._update_all())
        )
        self.alpha = QSlider(Qt.Orientation.Horizontal)
        self.alpha.setRange(0, 100)
        self.alpha.setValue(int(self.pet_alpha * 100))
        self.alpha.valueChanged.connect(
            lambda val: (setattr(self, "pet_alpha", val / 100.0), self._update_all())
        )
        self.lowpct = QDoubleSpinBox()
        self.lowpct.setDecimals(1)
        self.lowpct.setRange(0.0, 30.0)
        self.lowpct.setSingleStep(0.5)
        self.lowpct.setValue(self.pet_low_pct)
        self.lowpct.valueChanged.connect(
            lambda vval: (setattr(self, "pet_low_pct", float(vval)), self._update_all())
        )
        self.chk_axial = QCheckBox("Axial vertical flip (UD)")
        self.chk_axial.setChecked(self.axial_flip_ud)
        self.chk_axial.toggled.connect(
            lambda vval: (
                setattr(self, "axial_flip_ud", bool(vval)),
                self._sync_crosshairs(*self.crosshair),
            )
        )
        self.chk_axialZ = QCheckBox("Invert axial slice order (Z)")
        self.chk_axialZ.setChecked(self.axial_invert_z)
        self.chk_axialZ.toggled.connect(
            lambda vval: (
                setattr(self, "axial_invert_z", bool(vval)),
                self._sync_crosshairs(*self.crosshair),
            )
        )
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        self.lbl_stage = QLabel("")
        form.addRow(self.chk_overlay)
        form.addRow("CT Window Center:", self.sp_wc)
        form.addRow("CT Window Width:", self.sp_ww)
        form.addRow("PET Alpha:", self.alpha)
        form.addRow("PET Low %ile:", self.lowpct)
        form.addRow("Status:", self.lbl_stage)
        form.addRow(self.progress)
        v.addLayout(form)
        g.setLayout(v)
        return g

    def _image_controls(self):
        g = QGroupBox("Image Controls")
        v = QVBoxLayout()
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(8)
        top = QHBoxLayout()
        bot = QHBoxLayout()
        top.setSpacing(6)
        bot.setSpacing(6)
        self.chk_sync = QCheckBox("Sync view transforms")
        self.chk_sync.setChecked(True)
        self.chk_sync.toggled.connect(lambda val: setattr(self, "sync_views", bool(val)))
        self.btn_pan = self._tool_btn(
            "Pan", lambda: self.set_roi_mode("Pan" if self.btn_pan.isChecked() else None), True
        )
        self.btn_cross = self._tool_btn(
            "Crosshair",
            lambda: self.set_roi_mode("Cross" if self.btn_cross.isChecked() else None),
            True,
        )
        for w2 in [
            self.chk_sync,
            self._tool_btn("Zoom+", self.zoom_all_in),
            self._tool_btn("Zoom-", self.zoom_all_out),
            self.btn_pan,
            self.btn_cross,
            self._tool_btn("Reset", self.reset_all),
        ]:
            top.addWidget(w2)
        v.addLayout(top)
        g.setLayout(v)
        return g

    def _roi_tools(self):
        g = QGroupBox("ROI Tools")
        v = QVBoxLayout()
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(8)
        ops = QGridLayout()
        ops.setSpacing(6)
        self.btn_sph = QPushButton("Draw Spherical ROI")
        self.btn_sph.setCheckable(True)
        self.btn_sph.setMinimumHeight(34)
        self.btn_sph.clicked.connect(
            lambda: self.set_roi_mode("Spherical" if self.btn_sph.isChecked() else None)
        )
        v.addWidget(self.btn_sph)
        self.btn_copy_left = self._tool_btn("Copy", self.copy_roi)
        self.btn_paste_left = self._tool_btn("Paste", self.paste_roi)
        self.btn_clear_left = self._tool_btn("Clear", self.clear_rois)
        self.btn_delete_left = self._tool_btn("Delete Selected ROI", self.delete_selected_roi)
        ops.addWidget(self.btn_copy_left, 0, 0)
        ops.addWidget(self.btn_paste_left, 0, 1)
        ops.addWidget(self.btn_clear_left, 1, 0)
        ops.addWidget(self.btn_delete_left, 1, 1)
        v.addLayout(ops)
        self.chk_altcreate = QCheckBox("Use Alt+click to create ROI (safer)")
        self.chk_altcreate.setChecked(True)
        self.chk_altcreate.toggled.connect(
            lambda b: setattr(self, "require_alt_for_new_roi", bool(b))
        )
        v.addWidget(self.chk_altcreate)
        g.setLayout(v)
        return g

    def _view_box(self, title, canvas):
        box = QGroupBox(title)
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.valueChanged.connect(canvas.set_slice)
        slider.valueChanged.connect(lambda val, vw=canvas.view: self._on_slider_changed(vw, int(val)))
        canvas.slice_scrolled.connect(lambda d, s=slider: s.setValue(s.value() + d))
        setattr(self, f"slider_{canvas.view}", slider)
        layout.addWidget(canvas)
        layout.addWidget(slider)
        box.setLayout(layout)
        return box

    def _tool_btn(self, text, slot, checkable=False):
        b = QToolButton()
        b.setText(text)
        b.setCheckable(checkable)
        b.clicked.connect(slot)
        b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        b.setMinimumHeight(36)
        return b

    # ----------------------------------------------------------------
    # Orientation helpers
    # ----------------------------------------------------------------
    def toggle_axial_orientation(self):
        self.axial_flip_ud = not self.axial_flip_ud
        self.axial_invert_z = not self.axial_invert_z
        if hasattr(self, "chk_axial"):
            self.chk_axial.blockSignals(True)
            self.chk_axial.setChecked(self.axial_flip_ud)
            self.chk_axial.blockSignals(False)
        if hasattr(self, "chk_axialZ"):
            self.chk_axialZ.blockSignals(True)
            self.chk_axialZ.setChecked(self.axial_invert_z)
            self.chk_axialZ.blockSignals(False)
        self._sync_crosshairs(*self.crosshair)

    # ----------------------------------------------------------------
    # DICOM loading helpers (all as class methods)
    # ----------------------------------------------------------------
    @staticmethod
    def _parse_multi_float(s):
        try:
            if s is None:
                return None
            if isinstance(s, (list, tuple)):
                return tuple(float(x) for x in s)
            s = str(s).strip()
            if not s:
                return None
            parts = re.split(r"[\\, ]+", s)
            vals = [float(p) for p in parts if p != ""]
            return tuple(vals) if vals else None
        except Exception:
            return None

    def _read_meta_firstfile(self, file_path: str) -> Dict:
        meta: Dict = {}
        if SITK_AVAILABLE:
            try:
                r = sitk.ImageFileReader()
                r.SetFileName(file_path)
                r.LoadPrivateTagsOn()
                r.ReadImageInformation()

                def g(tag):
                    return r.GetMetaData(tag) if r.HasMetaDataKey(tag) else ""

                meta["modality"] = (g("0008|0060") or "").strip().upper()
                meta["series_desc"] = (g("0008|103e") or "").strip()
                meta["protocol"] = (g("0018|1030") or "").strip()
                meta["kernel"] = (g("0018|1210") or "").strip()
                meta["recon"] = (g("0018|9315") or "").strip()
                meta["slice_thickness"] = (
                    float(g("0018|0050")) if (g("0018|0050") or "").strip() else None
                )
                meta["spacing_between_slices"] = (
                    float(g("0018|0088")) if (g("0018|0088") or "").strip() else None
                )
                meta["pixel_spacing"] = self._parse_multi_float(g("0028|0030"))
                meta["rows"] = int(g("0028|0010")) if (g("0028|0010") or "").strip() else None
                meta["cols"] = int(g("0028|0011")) if (g("0028|0011") or "").strip() else None
                meta["image_type"] = (g("0008|0008") or "").strip()
                meta["corrected_image"] = (g("0028|0051") or "").strip()
                meta["series_number"] = (g("0020|0011") or "").strip()
                meta["manufacturer"] = (g("0008|0070") or "").strip()
                meta["model"] = (g("0008|1090") or "").strip()
                return meta
            except Exception:
                pass

        if PYDICOM_AVAILABLE:
            try:
                ds = pydicom.dcmread(file_path, stop_before_pixels=True, force=True)
                meta["modality"] = str(getattr(ds, "Modality", "")).strip().upper()
                meta["series_desc"] = str(getattr(ds, "SeriesDescription", "")).strip()
                meta["protocol"] = str(getattr(ds, "ProtocolName", "")).strip()
                meta["kernel"] = str(getattr(ds, "ConvolutionKernel", "")).strip()
                meta["slice_thickness"] = (
                    float(getattr(ds, "SliceThickness", None))
                    if getattr(ds, "SliceThickness", None) is not None
                    else None
                )
                ps = getattr(ds, "PixelSpacing", None)
                meta["pixel_spacing"] = tuple(float(x) for x in ps) if ps else None
                meta["rows"] = int(getattr(ds, "Rows", 0)) or None
                meta["cols"] = int(getattr(ds, "Columns", 0)) or None
                meta["image_type"] = (
                    "\\ ".join(getattr(ds, "ImageType", []))
                    if hasattr(ds, "ImageType")
                    else ""
                )
                meta["corrected_image"] = (
                    "\\ ".join(getattr(ds, "CorrectedImage", []))
                    if hasattr(ds, "CorrectedImage")
                    else ""
                )
                meta["series_number"] = str(getattr(ds, "SeriesNumber", "")).strip()
                meta["manufacturer"] = str(getattr(ds, "Manufacturer", "")).strip()
                meta["model"] = str(getattr(ds, "ManufacturerModelName", "")).strip()
            except Exception:
                pass
        return meta

    def _dicom_series_in_tree(self, root_dir: str) -> List[Dict]:
        series = []
        root_dir = os.path.abspath(root_dir)
        if not SITK_AVAILABLE:
            return series
        for d, _sub, _files in os.walk(root_dir):
            try:
                ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(d)
            except Exception:
                ids = None
            if not ids:
                continue
            for sid in ids:
                try:
                    fnames = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(d, sid)
                    if not fnames:
                        continue
                    meta = self._read_meta_firstfile(fnames[0])
                    modality = (meta.get("modality") or "UNK").upper()
                    item = {
                        "dir": d, "series_id": sid, "n_files": len(fnames),
                        "modality": modality, "files": fnames,
                    }
                    item.update(meta)
                    series.append(item)
                except Exception:
                    continue
        return series

    @staticmethod
    def _norm_text(s: str) -> str:
        return (s or "").strip().lower()

    def _score_ct(self, s: Dict) -> float:
        if (s.get("modality") or "").upper() != "CT":
            return -1e9
        ps = s.get("pixel_spacing") or None
        if ps and len(ps) >= 2:
            pixel_area = float(ps[0]) * float(ps[1])
        else:
            pixel_area = 10.0
        thk = s.get("spacing_between_slices") or s.get("slice_thickness") or 10.0
        try:
            thk = float(thk)
        except Exception:
            thk = 10.0
        n = float(s.get("n_files") or 0)
        rows = float(s.get("rows") or 0)
        cols = float(s.get("cols") or 0)
        mat = rows * cols
        score = (
            (1000.0 / max(pixel_area, 1e-6))
            + (600.0 / max(thk, 1e-6))
            + 0.05 * n
            + 0.000002 * mat
        )
        desc = self._norm_text(s.get("series_desc"))
        if "thin" in desc or "bone" in desc or "high" in desc:
            score += 20.0
        return float(score)

    def _score_pet(self, s: Dict) -> float:
        mod = (s.get("modality") or "").upper()
        if mod not in ("PT", "NM", "PET"):
            return -1e9
        desc = self._norm_text(s.get("series_desc"))
        proto = self._norm_text(s.get("protocol"))
        corr = self._norm_text(s.get("corrected_image"))
        itype = self._norm_text(s.get("image_type"))

        score = 0.0
        n = float(s.get("n_files") or 0)
        score += 0.02 * n

        ac_hits = 0
        if "attn" in corr:
            ac_hits += 3
        if "atten" in desc or "atten" in proto:
            ac_hits += 1
        if re.search(r"(^|\s)ac($|\s)", desc):
            ac_hits += 1
        if "nac" in desc or "nonac" in desc or "uncorr" in desc or "uncorrected" in desc:
            score -= 800.0
        score += 500.0 * ac_hits
        if "decy" in corr:
            score += 200.0
        if "scat" in corr:
            score += 120.0
        if "norm" in corr:
            score += 80.0
        if "recon" in itype or "derived" in itype:
            score += 50.0
        if "delayed" in desc or "late" in desc:
            score -= 30.0
        return float(score)

    def _pick_best_ct_pet(self, series_list: List[Dict]) -> Tuple[Optional[str], Optional[str]]:
        best_ct = None
        best_ct_sc = -1e18
        best_pet = None
        best_pet_sc = -1e18
        for s in series_list:
            s["ct_score"] = self._score_ct(s)
            s["pet_score"] = self._score_pet(s)
            if s["ct_score"] > best_ct_sc:
                best_ct_sc = s["ct_score"]
                best_ct = s.get("series_id")
            if s["pet_score"] > best_pet_sc:
                best_pet_sc = s["pet_score"]
                best_pet = s.get("series_id")
        return best_ct, best_pet

    def _to_lps(self, img):
        if not SITK_AVAILABLE or img is None:
            return img
        try:
            f = sitk.DICOMOrientImageFilter()
            f.SetDesiredCoordinateOrientation("LPS")
            return f.Execute(img)
        except Exception:
            return img

    def _load_sitk_series(self, series_item: Dict):
        if not SITK_AVAILABLE:
            raise RuntimeError("SimpleITK is required to load DICOM series.")
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(series_item["files"])
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        img = reader.Execute()
        return self._to_lps(img)

    def _clear_temp_nifti(self):
        try:
            if self._temp_nifti_dir and os.path.isdir(self._temp_nifti_dir):
                shutil.rmtree(self._temp_nifti_dir, ignore_errors=True)
        except Exception:
            pass
        self._temp_nifti_dir = None

    def _ensure_temp_nifti_dir(self) -> str:
        if not self._temp_nifti_dir:
            self._temp_nifti_dir = tempfile.mkdtemp(prefix="petct_dicom_as_nifti_")
        return self._temp_nifti_dir

    def _set_loaded_volume_from_sitk(self, kind: str, img):
        arr = sitk.GetArrayFromImage(img).transpose(2, 1, 0)  # x,y,z
        if kind == "ct":
            self.ct_img = img
            self.ct_data = arr.astype(np.float32)
            self.ct_spacing = img.GetSpacing()
        else:
            self.pet_img = img
            self.pet_data = arr.astype(np.float32)

        # temp nifti for tools that require a path
        try:
            td = self._ensure_temp_nifti_dir()
            out_path = os.path.join(td, f"{kind}.nii.gz")
            sitk.WriteImage(img, out_path)
            if kind == "ct":
                self.ct_path = out_path
            else:
                self.pet_path = out_path
        except Exception:
            if kind == "ct":
                self.ct_path = ""
            else:
                self.pet_path = ""

        if kind == "ct":
            self.image_data = self.ct_data
            self.slider_axial.setMaximum(self.ct_data.shape[2] - 1)
            self.slider_coronal.setMaximum(self.ct_data.shape[1] - 1)
            self.slider_sagittal.setMaximum(self.ct_data.shape[0] - 1)

        self._update_all()
        self._update_mode()

    # ---- DICOM loaders ----
    def load_petct_dicom_zip(self):
        if not SITK_AVAILABLE:
            QMessageBox.critical(self, "DICOM", "SimpleITK is required to load DICOM series.")
            return
        zip_path, _ = QFileDialog.getOpenFileName(
            self, "Select zipped DICOM folder (PET+CT)", "", "ZIP Files (*.zip)"
        )
        if not zip_path:
            return
        tmp_dir = tempfile.mkdtemp(prefix="dicomzip_")
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmp_dir)

            series = self._dicom_series_in_tree(tmp_dir)
            if not series:
                raise RuntimeError("No DICOM series found inside the ZIP.")

            pre_ct, pre_pet = self._pick_best_ct_pet(series)
            ct_candidates = [s for s in series if (s.get("modality") or "").upper() == "CT"]
            pet_candidates = [
                s for s in series if (s.get("modality") or "").upper() in ("PT", "NM", "PET")
            ]

            if len(ct_candidates) == 1 and len(pet_candidates) <= 1:
                ct_uid = ct_candidates[0].get("series_id")
                pet_uid = pet_candidates[0].get("series_id") if pet_candidates else None
            else:
                dlg = SeriesSelectionDialog(
                    series, preselect_ct_uid=pre_ct, preselect_pet_uid=pre_pet, parent=self,
                )
                if dlg.exec() != QDialog.DialogCode.Accepted:
                    return
                ct_uid, pet_uid = dlg.get_selection()

            ct_item = next((s for s in series if s.get("series_id") == ct_uid), None)
            if not ct_item:
                raise RuntimeError("No CT series selected.")
            self._set_loaded_volume_from_sitk("ct", self._load_sitk_series(ct_item))

            pet_item = next((s for s in series if s.get("series_id") == pet_uid), None)
            if pet_item:
                self._set_loaded_volume_from_sitk("pet", self._load_sitk_series(pet_item))
            else:
                self.pet_data = None
                self.pet_img = None
                self.pet_path = ""

            QMessageBox.information(
                self,
                "DICOM loaded",
                f"CT: {self.ct_data.shape}\nPET: {None if self.pet_data is None else self.pet_data.shape}",
            )
            self._dicom_extract_dir = tmp_dir
            tmp_dir = None

        except Exception as e:
            QMessageBox.critical(self, "DICOM error", f"Failed to load DICOM from ZIP:\n{e}")
        finally:
            try:
                if tmp_dir and os.path.isdir(tmp_dir):
                    shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

    def load_dicom_folder(self, kind: str = "ct"):
        if not SITK_AVAILABLE:
            QMessageBox.critical(self, "DICOM", "SimpleITK is required to load DICOM series.")
            return
        root = QFileDialog.getExistingDirectory(self, "Select DICOM folder")
        if not root:
            return
        try:
            series = self._dicom_series_in_tree(root)
            if not series:
                raise RuntimeError("No DICOM series found in the selected folder.")

            pre_ct, pre_pet = self._pick_best_ct_pet(series)
            dlg = SeriesSelectionDialog(
                series, preselect_ct_uid=pre_ct, preselect_pet_uid=pre_pet, parent=self,
            )
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            ct_uid, pet_uid = dlg.get_selection()

            sel_uid = ct_uid if kind.lower() == "ct" else pet_uid
            sel = next((s for s in series if s.get("series_id") == sel_uid), None)
            if not sel:
                raise RuntimeError("No matching series selected.")
            self._set_loaded_volume_from_sitk(
                "ct" if kind.lower() == "ct" else "pet",
                self._load_sitk_series(sel),
            )
            QMessageBox.information(self, "DICOM loaded", f"{kind.upper()} loaded.")
        except Exception as e:
            QMessageBox.critical(self, "DICOM error", f"Failed to load DICOM folder:\n{e}")

    # ----------------------------------------------------------------
    # Mode / display
    # ----------------------------------------------------------------
    def _update_pet_radio_state(self):
        enabled = self.registered_pet is not None
        if hasattr(self, "r_pet"):
            self.r_pet.setEnabled(bool(enabled))
        if hasattr(self, "r_fuse"):
            self.r_fuse.setEnabled(bool(enabled))

    def _update_mode(self):
        show = "CT"
        if hasattr(self, "r_fuse") and self.r_fuse.isChecked():
            show = "Fusion"
        elif hasattr(self, "r_pet") and self.r_pet.isChecked():
            show = "PET"

        if show in ("PET", "Fusion") and self.registered_pet is None:
            if hasattr(self, "r_ct"):
                self.r_ct.setChecked(True)
            self._update_pet_radio_state()
            show = "CT"

        self.fusion_mode = show
        self.image_data = self.ct_data if show != "PET" else self.registered_pet
        base = self.ct_data if self.ct_data is not None else self.image_data
        if base is not None:
            s = base.shape
            self._updating_gui = True
            self.slider_axial.setMaximum(s[2] - 1)
            self.slider_coronal.setMaximum(s[1] - 1)
            self.slider_sagittal.setMaximum(s[0] - 1)
            self._updating_gui = False
            self._sync_crosshairs(s[0] // 2, s[1] // 2, s[2] // 2)
        self._update_all()

    def _on_slider_changed(self, view: str, val: int):
        if (self.ct_data is None and self.image_data is None) or self._updating_gui:
            return
        s = self.ct_data.shape if self.ct_data is not None else self.image_data.shape
        x, y, z = self.crosshair
        if view == "axial":
            z = int(np.clip(val, 0, s[2] - 1))
        elif view == "coronal":
            y = int(np.clip(val, 0, s[1] - 1))
        else:
            x = int(np.clip(val, 0, s[0] - 1))
        self._sync_crosshairs(x, y, z)
        self._qc_overlay_cache = None

    def _update_all(self):
        for c in self._canvases():
            if c is not None:
                c.update_image()

    def _sync_crosshairs(self, x, y, z):
        if self.ct_data is None and self.image_data is None:
            return
        s = self.ct_data.shape if self.ct_data is not None else self.image_data.shape
        x = int(np.clip(x, 0, s[0] - 1))
        y = int(np.clip(y, 0, s[1] - 1))
        z = int(np.clip(z, 0, s[2] - 1))
        self.crosshair = [x, y, z]

        self._updating_gui = True
        try:
            self.slider_axial.setValue(z)
            self.slider_coronal.setValue(y)
            self.slider_sagittal.setValue(x)
            self.canvas_axial.crosshair_pos = QPointF(x, y)
            self.canvas_coronal.crosshair_pos = QPointF(x, z)
            self.canvas_sagittal.crosshair_pos = QPointF(y, z)
        finally:
            self._updating_gui = False

        self._update_all()

    # ---- View transform helpers ----
    def _apply_views(self, fn):
        if self.sync_views:
            for c in self._canvases():
                if c is not None:
                    fn(c)
        else:
            c = self._active_canvas()
            if c is not None:
                fn(c)

    def zoom_all_in(self):
        self._apply_views(lambda c: c.zoom_in())

    def zoom_all_out(self):
        self._apply_views(lambda c: c.zoom_out())

    def reset_all(self):
        self._apply_views(lambda c: c.reset_view())

    def flip_all_h(self):
        self._apply_views(lambda c: c.flip(True))

    def flip_all_v(self):
        self._apply_views(lambda c: c.flip(False))

    def rotate_all_cw(self):
        self._apply_views(lambda c: c.rotate(-90))

    def rotate_all_ccw(self):
        self._apply_views(lambda c: c.rotate(90))

    def _canvases(self):
        return [
            getattr(self, "canvas_axial", None),
            getattr(self, "canvas_coronal", None),
            getattr(self, "canvas_sagittal", None),
        ]

    def _set_active_canvas(self, canvas):
        self._last_active_canvas = canvas

    def _active_canvas(self):
        return self._last_active_canvas or getattr(self, "canvas_axial", None)

    def set_roi_mode(self, mode):
        self.roi_mode = mode
        if hasattr(self, "btn_sph"):
            self.btn_sph.setChecked(mode == "Spherical")
        if hasattr(self, "btn_pan"):
            self.btn_pan.setChecked(mode == "Pan")
        if hasattr(self, "btn_cross"):
            self.btn_cross.setChecked(mode == "Cross")
        for c in self._canvases():
            if c is not None:
                c.setCursor(
                    Qt.CursorShape.OpenHandCursor
                    if mode == "Pan"
                    else Qt.CursorShape.ArrowCursor
                )

    def copy_roi(self):
        c = self._active_canvas()
        self.roi_clipboard = c.selected_roi.copy() if c and c.selected_roi else None
        if self.roi_clipboard is None:
            QMessageBox.information(self, "Copy ROI", "Select an ROI first.")

    def paste_roi(self):
        c = self._active_canvas()
        if c and self.roi_clipboard:
            new_roi = self.roi_clipboard.copy()
            new_roi.center = new_roi.center + QPointF(10, 10)
            c.spherical_rois.append(new_roi)
            c.selected_roi = new_roi
            c.update()
        else:
            QMessageBox.information(self, "Paste ROI", "Clipboard empty.")

    def clear_rois(self):
        for c in self._canvases():
            if c is not None:
                c.spherical_rois = []
                c.selected_roi = None
                c.update()

    def _get_slice_for_canvas(self, canvas, use_pet=False):
        data = self.registered_pet if use_pet else self.ct_data
        if data is None:
            return None
        if canvas.view == "axial":
            idx = int(np.clip(canvas.current_slice, 0, data.shape[2] - 1))
            s = data[:, :, idx].T
        elif canvas.view == "coronal":
            idx = int(np.clip(canvas.current_slice, 0, data.shape[1] - 1))
            s = data[:, idx, :].T
        else:
            idx = int(np.clip(canvas.current_slice, 0, data.shape[0] - 1))
            s = data[idx, :, :].T
        if canvas.flip_h:
            s = np.fliplr(s)
        if canvas.flip_v:
            s = np.flipud(s)
        if canvas.rotation_angle != 0 and ndi_rotate is not None:
            s = ndi_rotate(
                s, canvas.rotation_angle, reshape=False, order=1,
                mode="constant", cval=np.min(s) if s.size > 0 else 0,
            )
        return s

    # ----------------------------------------------------------------
    # FIX 1: SINGLE unified _find_roi_by_tag that ALWAYS returns (canvas, roi)
    # All callers must unpack: c, roi = self._find_roi_by_tag(tag)
    # ----------------------------------------------------------------
    def _find_roi_by_tag(self, tag: str) -> Tuple[Optional['ImageCanvas'], Optional[SphericalROI]]:
        """Search all canvases for an ROI with the given tag.
        Returns (canvas, roi) tuple. Both are None if not found."""
        if tag in self._required_target_labels():
            for c in [getattr(self, "canvas_axial", None), getattr(self, "canvas_coronal", None), getattr(self, "canvas_sagittal", None)]:
                if c is None:
                    continue
                for rr in reversed(c.spherical_rois):
                    if getattr(rr, "tag", "") == tag:
                        return c, rr
            spec = self._get_target_spec(tag)
            if spec:
                self._sync_target_roi_pair_from_spec(tag, select_view="coronal")
                for c in [self.canvas_axial, self.canvas_coronal]:
                    for rr in reversed(c.spherical_rois):
                        if getattr(rr, "tag", "") == tag:
                            return c, rr
        for c in self._canvases():
            if c is None:
                continue
            for rr in reversed(c.spherical_rois):
                if getattr(rr, "tag", "") == tag:
                    return c, rr
        return None, None

    # ----------------------------------------------------------------
    # ROI analysis
    # ----------------------------------------------------------------
    def _get_site_mask_3d(self, site: str) -> Optional[np.ndarray]:
        if site in ("L1", "L2", "L3", "L4", "L5"):
            return self.vertebra_masks.get(site)
        if site == "FN_L":
            return self.extra_masks.get("FEMUR_L")
        if site == "FN_R":
            return self.extra_masks.get("FEMUR_R")
        return None

    def _roi_mask_on_slice(self, shape_hw, roi, view="axial"):
        h, w = shape_hw
        rows, cols = np.ogrid[:h, :w]
        sx, sy, sz = self.ct_spacing if self.ct_spacing else (1, 1, 1)
        pixel_ar = 1.0
        if view == "axial":
            pixel_ar = sy / sx
        elif view == "coronal":
            pixel_ar = sz / sx
        else:
            pixel_ar = sz / sy
        r = int(max(5, roi.radius))
        cx = int(round(roi.center.x()))
        cy = int(round(roi.center.y()))
        return ((rows - cy) ** 2 + ((cols - cx) / pixel_ar) ** 2) <= (r ** 2)


    def _axial_disp_to_data_z(self, z_disp: int) -> int:
        if self.ct_data is None:
            return int(z_disp)
        z_disp = int(np.clip(z_disp, 0, self.ct_data.shape[2] - 1))
        return (self.ct_data.shape[2] - 1 - z_disp) if self.axial_invert_z else z_disp

    def _axial_data_to_disp_z(self, z_data: int) -> int:
        if self.ct_data is None:
            return int(z_data)
        z_data = int(np.clip(z_data, 0, self.ct_data.shape[2] - 1))
        return (self.ct_data.shape[2] - 1 - z_data) if self.axial_invert_z else z_data

    def _new_axial_roi(self, row: int, col: int, z_data: int, r: int, tag: str):
        if self.ct_data is None:
            raise RuntimeError("Load CT first.")
        z_data = int(np.clip(z_data, 0, self.ct_data.shape[2] - 1))
        row = int(np.clip(row, 0, self.ct_data.shape[0] - 1))
        col = int(np.clip(col, 0, self.ct_data.shape[1] - 1))
        r = int(max(5, r))
        disp_z = self._axial_data_to_disp_z(z_data)
        self.slider_axial.setValue(int(disp_z))
        if tag in self._required_target_labels():
            ax_roi, _cor_roi = self._create_target_roi_pair(tag, int(row), int(col), int(z_data), radius=r, select_view="axial")
            self.canvas_axial.selected_roi = ax_roi
            self.canvas_axial.hover_roi = ax_roi
            self.canvas_axial.update()
            return ax_roi
        c = self.canvas_axial
        roi = SphericalROI(row, col, r, tag=tag, slice_index=int(disp_z))
        roi.locked_to_slice = True
        roi.follow_on_scroll = False
        c.spherical_rois.append(roi)
        c.selected_roi = roi
        c.hover_roi = roi
        c.update()
        return roi

    def _place_femoral_neck_fallback_from_mask_array(self, marr: np.ndarray, tag: str, shrink: float = 0.34):
        if self.ct_data is None:
            raise RuntimeError("Load CT first.")
        marr = (marr > 0).astype(np.uint8)
        if marr.shape != self.ct_data.shape:
            raise RuntimeError(f"Femur fallback mask shape {marr.shape} != CT shape {self.ct_data.shape}")
        idx = np.argwhere(marr > 0)
        if idx.size == 0:
            raise RuntimeError(f"No voxels in fallback mask for {tag}")

        zs = np.unique(idx[:, 2])
        zs.sort()
        areas = np.array([int(marr[:, :, int(z)].sum()) for z in zs], dtype=np.float64)
        if areas.size >= 5:
            ker = np.ones(5, dtype=np.float64) / 5.0
            areas_s = np.convolve(areas, ker, mode="same")
        else:
            areas_s = areas.copy()

        edge_n = max(1, areas_s.size // 5)
        low_mean = float(np.mean(areas_s[:edge_n]))
        high_mean = float(np.mean(areas_s[-edge_n:]))
        search_from_top = high_mean >= low_mean

        if search_from_top:
            start = max(0, areas_s.size - max(4, areas_s.size // 3))
            stop = areas_s.size
        else:
            start = 0
            stop = min(areas_s.size, max(4, areas_s.size // 3))

        cand_idx = np.arange(start, stop, dtype=int)
        if cand_idx.size == 0:
            cand_idx = np.arange(areas_s.size, dtype=int)

        best = None
        for ii in cand_idx:
            z = int(zs[ii])
            sl = marr[:, :, z].astype(bool)
            if int(sl.sum()) < 25:
                continue
            yx = np.argwhere(sl)
            x0, x1 = int(yx[:, 1].min()), int(yx[:, 1].max())
            y0, y1 = int(yx[:, 0].min()), int(yx[:, 0].max())
            width = float(x1 - x0 + 1)
            height = float(y1 - y0 + 1)
            aspect_penalty = abs(width - height) / max(width + height, 1.0)
            score = areas_s[ii] + 0.35 * width + 0.10 * height - 10.0 * aspect_penalty
            if best is None or score < best[0]:
                best = (score, z, sl, width, height)

        if best is None:
            return self._place_fallback_roi_from_mask_array(marr, tag=tag, shrink=0.32)

        _, z_best, sl, width, height = best
        if binary_erosion is not None:
            tmp = binary_erosion(sl, iterations=1)
            if int(tmp.sum()) >= max(12, int(0.20 * max(int(sl.sum()), 1))):
                sl = tmp

        yx = np.argwhere(sl)
        cy = int(np.round(np.mean(yx[:, 0])))
        cx = int(np.round(np.mean(yx[:, 1])))
        r_box = 0.28 * min(width, height)
        r_area = math.sqrt(float(sl.sum()) / max(math.pi, 1e-6)) * float(shrink)
        r = int(max(6, min(16, min(r_box, r_area))))
        return self._new_axial_roi(cy, cx, z_best, r, tag)

    def _place_femoral_neck_fallback_from_mask_path(self, mask_path: str, tag: str, shrink: float = 0.34):
        if not SITK_AVAILABLE:
            raise RuntimeError("SimpleITK required for fallback femur placement.")
        if self.ct_data is None or self.ct_img is None:
            raise RuntimeError("Load CT first.")
        m = sitk.ReadImage(mask_path)
        m = self._to_lps(m)
        m = self._resample_like(m, self.ct_img, nearest=True)
        marr = sitk.GetArrayFromImage(m).transpose(2, 1, 0)
        marr = (marr > 0.5).astype(np.uint8)
        return self._place_femoral_neck_fallback_from_mask_array(marr, tag=tag, shrink=shrink)


    # FIX 2: _analyze_roi now uses ROI's stored slice_index when available
    def _analyze_roi(self, canvas, roi, site: Optional[str] = None):
        if self.ct_data is None or canvas is None or roi is None:
            return None

        site = site or getattr(roi, "tag", None) or "ROI"

        ct_slice = None
        slice_idx = 0
        if canvas.view == "axial":
            # FIX: use ROI's stored slice_index if available, not canvas.current_slice
            if getattr(roi, 'slice_index', None) is not None:
                z_disp = int(np.clip(roi.slice_index, 0, self.ct_data.shape[2] - 1))
            else:
                z_disp = int(np.clip(canvas.current_slice, 0, self.ct_data.shape[2] - 1))
            slice_idx = self._axial_disp_to_data_z(z_disp)
            ct_slice = self.ct_data[:, :, slice_idx]
        elif canvas.view == "coronal":
            if getattr(roi, 'slice_index', None) is not None:
                slice_idx = int(np.clip(roi.slice_index, 0, self.ct_data.shape[1] - 1))
            else:
                slice_idx = int(np.clip(canvas.current_slice, 0, self.ct_data.shape[1] - 1))
            ct_slice = self.ct_data[:, slice_idx, :]
        else:
            if getattr(roi, 'slice_index', None) is not None:
                slice_idx = int(np.clip(roi.slice_index, 0, self.ct_data.shape[0] - 1))
            else:
                slice_idx = int(np.clip(canvas.current_slice, 0, self.ct_data.shape[0] - 1))
            ct_slice = self.ct_data[slice_idx, :, :]

        if ct_slice is None:
            return None

        i0 = int(round(roi.center.x()))
        i1 = int(round(roi.center.y()))
        h, w = ct_slice.shape
        if not (0 <= i0 < h and 0 <= i1 < w):
            return None

        rows, cols = np.ogrid[:h, :w]
        sx, sy, sz = self.ct_spacing if self.ct_spacing else (1, 1, 1)
        pixel_ar = 1.0
        if canvas.view == "axial":
            pixel_ar = sy / max(sx, 1e-6)
        elif canvas.view == "coronal":
            pixel_ar = sz / max(sx, 1e-6)
        elif canvas.view == "sagittal":
            pixel_ar = sz / max(sy, 1e-6)

        r = int(max(5, roi.radius))
        mask_roi = ((rows - i0) ** 2 + ((cols - i1) / pixel_ar) ** 2) <= (r ** 2)
        if mask_roi.sum() == 0:
            return None

        is_bone_site = str(site).startswith("L") or str(site).startswith("FN")
        site_mask3d = self._get_site_mask_3d(site)
        mask_site2d = None
        if isinstance(site_mask3d, np.ndarray) and site_mask3d.shape == self.ct_data.shape:
            if canvas.view == "axial":
                mask_site2d = site_mask3d[:, :, slice_idx].astype(bool)
            elif canvas.view == "coronal":
                mask_site2d = site_mask3d[:, slice_idx, :].astype(bool)
            else:
                mask_site2d = site_mask3d[slice_idx, :, :].astype(bool)

        overlap = None
        if mask_site2d is not None and mask_roi.sum() > 0:
            overlap = float((mask_roi & mask_site2d).sum()) / float(mask_roi.sum())

        # FIX: for bone sites without a seg mask, still allow measurement if there's no mask yet
        # (e.g. manual ROI before running TotalSegmentator)
        if is_bone_site and mask_site2d is not None and (overlap is None or overlap < max(float(self.qc_overlap_min), 0.35)):
            return None

        mask = mask_roi & mask_site2d if mask_site2d is not None else mask_roi
        mask_status_parts = ["segmask" if mask_site2d is not None else "roi_only"]

        if self.mask_enable:
            within = (ct_slice >= self.mask_hu_min) & (ct_slice <= self.mask_hu_max)
            cand = mask & within
            if cand.sum() >= max(10, int(0.05 * max(mask.sum(), 1))):
                mask = cand
                mask_status_parts.append("hu_on")
            else:
                mask_status_parts.append("hu_fallback")

        if getattr(self, "mask_erode_px", 0) > 0 and binary_erosion is not None:
            cand = binary_erosion(mask, iterations=int(self.mask_erode_px))
            if cand.sum() >= max(10, int(0.05 * max(mask.sum(), 1))):
                mask = cand
                mask_status_parts.append(f"erode{int(self.mask_erode_px)}")

        if is_bone_site and getattr(self, "auto_cortex", False) and binary_erosion is not None:
            base = mask.copy()
            best = base
            for it in range(int(self.cortex_min_px), int(self.cortex_max_px) + 1):
                cand = binary_erosion(base, iterations=it)
                if cand.sum() < 50:
                    break
                hi_frac = float((ct_slice[cand] >= float(self.cortex_hu_thr)).sum()) / float(cand.sum())
                best = cand
                if hi_frac <= float(self.qc_cortex_hi_frac_thr):
                    break
            if best.sum() >= 50 and best.sum() < mask.sum():
                mask = best
                mask_status_parts.append("cortex_auto")

        vals = ct_slice[mask]
        if vals.size == 0:
            return None

        mean_hu = float(np.mean(vals))
        std_hu = float(np.std(vals))
        if is_bone_site and (mean_hu < -20.0 or mean_hu > 500.0):
            return None

        bmd = max(0.0, self.cal_slope_eff * mean_hu + self.cal_intercept_eff)
        if not is_bone_site:
            t = float('nan')
            z_score = float('nan')
            who = 'N/A'
        else:
            ya_mu, ya_sd = young_adult_norms(self.patient_sex)
            t = (bmd - ya_mu) / ya_sd
            z_mu, z_sd = age_matched_norms(self.patient_age, self.patient_sex)
            z_score = (bmd - z_mu) / z_sd
            who = "Normal" if t >= -1.0 else ("Osteopenia" if t >= -2.5 else "Osteoporosis")

        qc_reasons: List[str] = []
        metal_frac = float((vals >= float(self.qc_metal_hu_thr)).sum()) / float(vals.size) if vals.size else 0.0
        if metal_frac > float(self.qc_metal_frac_thr):
            qc_reasons.append("MetalArtifact")
        if overlap is not None and overlap < float(self.qc_overlap_min):
            qc_reasons.append("LowROIMaskOverlap")
        if str(site) in ("FN_L", "FN_R"):
            key = "FEMUR_L" if str(site) == "FN_L" else "FEMUR_R"
            if self._femur_cropped.get(key, False):
                qc_reasons.append("FemurCropped")

        qc_status = "PASS" if not qc_reasons else "FAIL"
        return {
            "mean_hu": mean_hu,
            "std_hu": std_hu,
            "bmd": bmd,
            "t_score": t,
            "z_score": z_score,
            "who_classification": who,
            "nvox": int(mask.sum()),
            "mask_status": "|".join(mask_status_parts),
            "qc_status": qc_status,
            "qc_reasons": qc_reasons,
            "qc_overlap": overlap,
            "qc_metal_frac": metal_frac,
            "site": str(site),
        }

    # ----------------------------------------------------------------
    # Measurement actions
    # ----------------------------------------------------------------
    def measure_vertebra_selected(self):
        v = self.vertebra.currentText()
        self.measure_specific(v)

    def measure_specific(self, name: str):
        c = self._active_canvas()
        if not (c and c.spherical_rois):
            QMessageBox.warning(self, "No ROI", "Draw or auto-place ROI then re-try.")
            return
        roi = None
        for rr in reversed(c.spherical_rois):
            if getattr(rr, "tag", "") == name:
                roi = rr
                break
        roi = roi or c.selected_roi or c.spherical_rois[-1]

        # Navigate to the ROI's slice before measuring
        if roi and getattr(roi, "slice_index", None) is not None:
            self.slider_axial.setValue(int(roi.slice_index))

        res = self._analyze_roi(c, roi, site=name)
        if res:
            self.vertebral[name] = res
            self._display_results(res, name)
        else:
            QMessageBox.warning(self, "Measure", f"Measurement failed for {name}.")

    def measure_all_sites(self):
        """Measure all sites from canonical locked target specs on axial CT."""
        if self.ct_data is None:
            QMessageBox.warning(self, "Measure All", "Load CT first.")
            return
        measured = 0
        missing = []
        for name in self.site_names:
            roi_ax = self._get_target_spec_as_axial_roi(name)
            if roi_ax is None:
                missing.append(name)
                continue
            if getattr(roi_ax, "slice_index", None) is not None:
                self.slider_axial.setValue(int(roi_ax.slice_index))
            res = self._analyze_roi(self.canvas_axial, roi_ax, site=name)
            if res:
                self.vertebral[name] = res
                self._display_results(res, name)
                measured += 1
            else:
                missing.append(name)

        if measured > 0:
            msg = f"Measured {measured} sites. Results in panel."
            if missing:
                msg += "\nMissing/failed ROI tags: " + ", ".join(missing)
            QMessageBox.information(self, "Measure All", msg)
        else:
            detail = ("\nMissing/failed ROI tags: " + ", ".join(missing)) if missing else ""
            QMessageBox.warning(self, "Measure All", "No measurements succeeded." + detail)

    def _display_results(self, res, name: str):
        if str(name).startswith("L") or str(name).startswith("FN"):
            line = (
                f"{name}: HU={res['mean_hu']:.1f}  BMD={res['bmd']:.1f}  "
                f"T={res['t_score']:.2f}  Z={res.get('z_score', float('nan')):.2f}  "
                f"Class={res['who_classification']}"
            )
        else:
            line = f"{name}: HU={res['mean_hu']:.1f}  BMD={res['bmd']:.1f}  Class=N/A"
        cur = self.results.toPlainText()
        header1, header2 = "SITE RESULTS", "-" * 40
        if not cur.strip():
            self.results.setPlainText(header1 + "\n" + header2 + "\n" + line)
            return
        lines = cur.splitlines()
        out = []
        replaced = False
        seen = set()
        for L in lines:
            key = L.split(":", 1)[0].strip()
            if key in self.site_names:
                if key == name and not replaced:
                    out.append(line)
                    replaced = True
                    seen.add(key)
                elif key not in seen:
                    out.append(L)
                    seen.add(key)
            else:
                out.append(L)
        if not replaced:
            out.append(line)
        body = [l for l in out if l and l.split(":", 1)[0].strip() in set(self.site_names)]
        header = [l for l in out if l not in body]
        ordered = []
        for vname in self.site_names:
            for l in body:
                if l.startswith(vname + ":"):
                    ordered.append(l)
        self.results.setPlainText("\n".join(header + ordered))

    # ----------------------------------------------------------------
    # Auto segmentation & placement
    # ----------------------------------------------------------------
    def _resample_like(self, mov_img, ref_img, nearest=True):
        interp = sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear
        # Use the reference-image overload. The previous 9-argument form mixed
        # overload signatures and fails on some SimpleITK builds.
        return sitk.Resample(
            mov_img,
            ref_img,
            sitk.Transform(),
            interp,
            0.0,
            mov_img.GetPixelID(),
        )

    def _place_from_mask_path(self, mask_path: str, name: str, shrink: float):
        if not SITK_AVAILABLE:
            raise RuntimeError("SimpleITK required for mask resampling.")
        m = sitk.ReadImage(mask_path)
        m = self._to_lps(m)
        if self.ct_img is not None:
            m = self._resample_like(m, self.ct_img, nearest=True)
        marr = sitk.GetArrayFromImage(m).transpose(2, 1, 0)
        return self._place_from_mask_array(marr > 0.5, name, shrink)

    def _place_from_mask_array(self, mask: np.ndarray, name: str, shrink: float):
        if self.ct_data is None:
            raise RuntimeError("Load CT first.")
        mask = (mask > 0).astype(np.uint8)
        if mask.shape != self.ct_data.shape:
            raise RuntimeError(f"Mask shape {mask.shape} != CT shape {self.ct_data.shape}")
        self.vertebra_masks[name] = mask.copy()
        idx = np.argwhere(mask > 0)
        if idx.size == 0:
            raise RuntimeError(f"No voxels in {name} mask")
        zs = np.unique(idx[:, 2])
        zs.sort()
        areas = np.array([int(mask[:, :, int(z)].sum()) for z in zs], dtype=np.int64)
        cz = int(zs[int(np.argmax(areas))])
        sl = mask[:, :, cz].astype(bool)
        if sl.sum() < 10:
            cz = int(zs[len(zs) // 2])
            sl = mask[:, :, cz].astype(bool)
        if sl.sum() < 10:
            raise RuntimeError(f"{name}: mask too small on chosen slice")
        if binary_erosion is not None:
            tmp = binary_erosion(sl, iterations=1)
            if tmp.sum() >= 10:
                sl = tmp
        yx = np.argwhere(sl)
        cy = int(np.mean(yx[:, 0]))
        cx = int(np.mean(yx[:, 1]))
        area = float(sl.sum())
        r_eq = math.sqrt(area / math.pi)
        r = int(max(8, min(28, r_eq * shrink)))
        return self._new_axial_roi(cy, cx, cz, r, name)

    def _place_simple_roi_from_mask_path(self, mask_path: str, tag: str, shrink: float = 0.55):
        if not SITK_AVAILABLE:
            raise RuntimeError("SimpleITK required for mask resampling.")
        if self.ct_data is None or self.ct_img is None:
            raise RuntimeError("Load CT first.")
        m = sitk.ReadImage(mask_path)
        m = self._to_lps(m)
        m = self._resample_like(m, self.ct_img, nearest=True)
        marr = sitk.GetArrayFromImage(m).transpose(2, 1, 0)
        marr = (marr > 0.5).astype(np.uint8)
        idx = np.argwhere(marr > 0)
        if idx.size == 0:
            raise RuntimeError(f"No voxels in {tag} mask")
        cz = int(np.median(idx[:, 2]))
        sl = marr[:, :, cz].astype(bool)
        if sl.sum() < 20:
            zs = np.unique(idx[:, 2])
            zs.sort()
            cz = int(zs[len(zs) // 2])
            sl = marr[:, :, cz].astype(bool)
        if sl.sum() < 20:
            raise RuntimeError(f"{tag}: mask too small on chosen slice")
        if binary_erosion is not None:
            tmp = binary_erosion(sl, iterations=1)
            if tmp.sum() >= 20:
                sl = tmp
        yx = np.argwhere(sl)
        cy = int(np.mean(yx[:, 0]))
        cx = int(np.mean(yx[:, 1]))
        area = float(sl.sum())
        r_eq = math.sqrt(area / max(math.pi, 1e-6))
        r = int(max(10, min(30, r_eq * float(shrink))))
        roi = self._new_axial_roi(cy, cx, cz, r, tag)
        # Cache mask
        self.extra_masks[tag] = marr.copy()
        return roi

    def _place_femoral_neck_roi_from_femur_mask_path(
        self, mask_path: str, tag: str, shrink: float = 0.45
    ):
        if not SITK_AVAILABLE:
            raise RuntimeError("SimpleITK required for mask resampling.")
        if self.ct_data is None or self.ct_img is None:
            raise RuntimeError("Load CT first.")
        if tag not in ("FN_L", "FN_R"):
            raise ValueError("tag must be FN_L or FN_R")

        femur_key = "FEMUR_L" if tag == "FN_L" else "FEMUR_R"

        m = sitk.ReadImage(mask_path)
        m = self._to_lps(m)
        m = self._resample_like(m, self.ct_img, nearest=True)
        marr = sitk.GetArrayFromImage(m).transpose(2, 1, 0)
        marr = (marr > 0.5).astype(np.uint8)

        if marr.shape != self.ct_data.shape:
            raise RuntimeError(f"Femur mask shape {marr.shape} != CT shape {self.ct_data.shape}")

        self.extra_masks[femur_key] = marr.copy()

        idx = np.argwhere(marr > 0)
        if idx.size == 0:
            raise RuntimeError(f"No voxels in {tag} femur mask")

        sx, sy, sz = self.ct_spacing if self.ct_spacing else (1.0, 1.0, 1.0)
        touch = bool(
            (idx[:, 0].min() == 0) or (idx[:, 0].max() == marr.shape[0] - 1)
            or (idx[:, 1].min() == 0) or (idx[:, 1].max() == marr.shape[1] - 1)
            or (idx[:, 2].min() == 0) or (idx[:, 2].max() == marr.shape[2] - 1)
        )
        z_span_mm = float((idx[:, 2].max() - idx[:, 2].min() + 1) * sz)
        self._femur_cropped[femur_key] = bool(touch or (z_span_mm < float(self.qc_femur_min_z_mm)))

        # Area profile
        zs = np.unique(idx[:, 2])
        zs.sort()
        areas = np.array([int(marr[:, :, int(z)].sum()) for z in zs], dtype=np.int64)
        if areas.size < 5:
            raise RuntimeError(f"{tag}: femur mask too small")

        if areas.size >= 5:
            ker = np.ones(5, dtype=np.float64) / 5.0
            areas_s = np.convolve(areas.astype(np.float64), ker, mode="same")
        else:
            areas_s = areas.astype(np.float64)

        max_i = int(np.argmax(areas_s))

        min_off = max(1, int(round(15.0 / max(sz, 1e-6))))
        max_off = max(min_off + 1, int(round(90.0 / max(sz, 1e-6))))
        look_ahead = max(1, int(round(40.0 / max(sz, 1e-6))))

        def pick_candidate(direction: int):
            start_i = max_i + direction * min_off
            end_i = max_i + direction * max_off
            if direction < 0:
                lo = max(0, end_i)
                hi = min(areas_s.size - 1, start_i)
                cand_range = range(hi, lo - 1, -1)
            else:
                lo = max(0, start_i)
                hi = min(areas_s.size - 1, end_i)
                cand_range = range(lo, hi + 1)

            best_i = None
            best_area = None
            for ii in cand_range:
                a = areas_s[ii]
                if best_area is None or a < best_area:
                    best_area = a
                    best_i = ii

            if best_i is None:
                return None

            if direction < 0:
                after_lo = max(0, best_i - look_ahead)
                after = areas_s[after_lo:best_i]
            else:
                after_hi = min(areas_s.size - 1, best_i + look_ahead)
                after = areas_s[best_i + 1: after_hi + 1]

            score = 0.0
            if after.size:
                score = float(after.max() / max(best_area, 1e-6))
            return (best_i, float(best_area), score)

        c1 = pick_candidate(-1)
        c2 = pick_candidate(+1)

        chosen = None
        if c1 and c2:
            chosen = c1 if c1[2] >= c2[2] else c2
        else:
            chosen = c1 or c2

        if chosen is None:
            if max_i < areas_s.size // 2:
                neck_i = int(min(areas_s.size - 1, max_i + max(1, areas_s.size // 6)))
            else:
                neck_i = int(max(0, max_i - max(1, areas_s.size // 6)))
        else:
            neck_i = int(chosen[0])

        cz = int(zs[neck_i])
        sl = marr[:, :, cz].astype(bool)
        if sl.sum() < 50:
            raise RuntimeError(f"{tag}: chosen femoral neck slice too small")

        if binary_erosion is not None:
            tmp = binary_erosion(sl, iterations=1)
            if tmp.sum() >= 50:
                sl = tmp

        yx = np.argwhere(sl)
        cy = int(np.mean(yx[:, 0]))
        cx = int(np.mean(yx[:, 1]))
        area = float(sl.sum())
        r_eq = math.sqrt(area / math.pi)
        r = int(max(10, min(26, r_eq * float(shrink))))

        return self._new_axial_roi(cy, cx, cz, r, tag)

    def _auto_place_fat_roi(self, disp_z: int, tag: str = "FAT", r_mm: float = 15.0):
        if self.ct_data is None:
            raise RuntimeError("Load CT first.")
        z_disp = int(np.clip(disp_z, 0, self.ct_data.shape[2] - 1))
        z_real = (self.ct_data.shape[2] - 1 - z_disp) if self.axial_invert_z else z_disp
        ct_slice = self.ct_data[:, :, z_real]

        fat = (ct_slice >= -190.0) & (ct_slice <= -30.0)
        body = ct_slice > -300.0
        inner = body
        try:
            if binary_erosion is not None:
                inner = binary_erosion(body, iterations=10)
        except Exception:
            inner = body
        fat = fat & inner

        # Exclude vertebra and psoas masks
        try:
            excl = np.zeros_like(fat, dtype=bool)
            for vname in ["L1", "L2", "L3", "L4", "L5"]:
                vm = self.vertebra_masks.get(vname)
                if vm is not None and vm.shape == self.ct_data.shape:
                    excl |= vm[:, :, z_real] > 0
            for key in ["PSOAS_L", "PSOAS_R"]:
                pm = self.extra_masks.get(key)
                if pm is not None and pm.shape == self.ct_data.shape:
                    excl |= pm[:, :, z_real] > 0
            if binary_erosion is not None:
                try:
                    excl = ~binary_erosion(~excl, iterations=4)
                except Exception:
                    pass
            fat = fat & (~excl)
        except Exception:
            pass

        pts = np.argwhere(fat)
        if pts.size < 200:
            raise RuntimeError("Could not find sufficient fat pixels. Try manual ROI.")

        if ndi_label is not None:
            try:
                lbl, n = ndi_label(fat.astype(np.uint8))
                if n > 1:
                    sizes = np.bincount(lbl.ravel())
                    sizes[0] = 0
                    k = int(np.argmax(sizes))
                    pts = np.argwhere(lbl == k)
            except Exception:
                pass

        cy = int(np.mean(pts[:, 0]))
        cx = int(np.mean(pts[:, 1]))
        sx, sy, _ = self.ct_spacing if self.ct_spacing is not None else (1.0, 1.0, 1.0)
        px = max(min(sx, sy), 1e-6)
        r_px = int(max(10, min(35, round(float(r_mm) / px))))

        self.slider_axial.setValue(int(z_disp))
        c = self.canvas_axial
        roi = SphericalROI(cy, cx, r_px, tag=tag, slice_index=int(z_disp))
        c.spherical_rois.append(roi)
        c.selected_roi = roi
        c.update()
        return roi


    def _check_totalseg_environment(self):
        lines = []
        lines.append(f"Python: {sys.version.split()[0]}")
        lines.append(f"SimpleITK: {'available' if SITK_AVAILABLE else 'missing'}")
        lines.append(f"nibabel: {'available' if HAVE_NIB else 'missing'}")
        lines.append(f"TotalSegmentator import: {'available' if TOTALSEG_AVAILABLE else 'missing'}")
        lines.append(f"TotalSegmentator Python API: {'available' if TOTALSEG_API else 'missing'}")
        try:
            import torch
            torch_ver = getattr(torch, "__version__", "unknown")
            cuda_ok = bool(getattr(torch.cuda, "is_available", lambda: False)())
            lines.append(f"PyTorch: {torch_ver} | CUDA available: {'yes' if cuda_ok else 'no'}")
        except Exception as e:
            lines.append(f"PyTorch: missing ({e})")
        cli_hits = []
        for exe in ["TotalSegmentator", "totalsegmentator"]:
            p = shutil.which(exe)
            if p:
                cli_hits.append(f"{exe} -> {p}")
        lines.append("CLI: " + ("; ".join(cli_hits) if cli_hits else "not found in PATH"))
        lines.append("")
        lines.append("Expected setup:")
        lines.append("  - Python >= 3.9")
        lines.append("  - PyTorch >= 2.0")
        lines.append("  - pip install TotalSegmentator")
        lines.append("")
        lines.append("This application writes a temporary CT NIfTI automatically and runs TotalSegmentator on that CT. After auto-placement, you can still drag ROIs to correct them.")
        QMessageBox.information(self, "TotalSegmentator Setup", "\n".join(lines))

    def _study_cache_dir(self) -> Optional[str]:
        if self.ct_data is None or not self.seg_use_cache:
            return None
        try:
            os.makedirs(self.seg_cache_root, exist_ok=True)
            parts = [
                str(self.ct_data.shape),
                ",".join(f"{float(s):.4f}" for s in (self.ct_spacing or (1.0, 1.0, 1.0))),
                f"{float(np.nanmean(self.ct_data)):.3f}",
                f"{float(np.nanstd(self.ct_data)):.3f}",
            ]
            key = re.sub(r"[^A-Za-z0-9_.-]+", "_", "_".join(parts))
            out_dir = os.path.join(self.seg_cache_root, key)
            os.makedirs(out_dir, exist_ok=True)
            return out_dir
        except Exception:
            return None

    def _collect_auto_roi_tags(self) -> List[str]:
        return ["L1", "L2", "L3", "L4", "FN_L", "FN_R", "FAT", "MUSCLE", "PSOAS_L", "PSOAS_R"]

    def _clear_existing_auto_rois(self):
        tags = set(self._collect_auto_roi_tags())
        for c in self._canvases():
            kept = [r for r in c.spherical_rois if getattr(r, "tag", "") not in tags]
            c.spherical_rois = kept
            if c.selected_roi is not None and getattr(c.selected_roi, "tag", "") in tags:
                c.selected_roi = None
            if c.hover_roi is not None and getattr(c.hover_roi, "tag", "") in tags:
                c.hover_roi = None
            c.update()
        for name in self.site_names:
            self.vertebral[name] = None
        self.dxa_proj_last = None

    def _mask_file_ready(self, path: Optional[str]) -> bool:
        return bool(path and os.path.exists(path) and os.path.getsize(path) > 0)

    def _cached_totalseg_files(self, out_dir: str) -> Dict[str, str]:
        files = {}
        all_masks = []
        for ext in ("*.nii.gz", "*.nii"):
            all_masks.extend(glob(os.path.join(out_dir, "**", ext), recursive=True))
        lookup = {os.path.basename(p).lower(): p for p in all_masks}

        def find_mask(candidates):
            for cand in candidates:
                c = cand.lower()
                if c in lookup:
                    return lookup[c]
                for base, full in lookup.items():
                    if c in base:
                        return full
            return None

        for name in ["L1", "L2", "L3", "L4", "L5"]:
            hit = find_mask([
                f"vertebrae_{name}.nii.gz", f"vertebrae_{name}.nii",
                f"vertebra_{name}.nii.gz", f"vertebra_{name}.nii",
                f"{name}.nii.gz", f"{name}.nii",
            ])
            if hit:
                files[name] = hit

        for key, label in [("PSOAS_L", "iliopsoas_left"), ("PSOAS_R", "iliopsoas_right")]:
            hit = find_mask([f"{label}.nii.gz", f"{label}.nii", label])
            if hit:
                files[key] = hit

        for key, label in [("FEMUR_L", "femur_left"), ("FEMUR_R", "femur_right")]:
            hit = find_mask([f"{label}.nii.gz", f"{label}.nii", label])
            if hit:
                files[key] = hit
        return files

    def _required_totalseg_files_present(self, files: Dict[str, str]) -> bool:
        needed = ["L1", "L2", "L3", "L4", "FEMUR_L", "FEMUR_R"]
        return all(self._mask_file_ready(files.get(k)) for k in needed)

    def _validate_auto_roi_summary(self) -> Dict[str, dict]:
        summary = {}
        for tag in ["L1", "L2", "L3", "L4", "FN_L", "FN_R"]:
            _c, roi = self._find_roi_by_tag(tag)
            if roi is None:
                summary[tag] = {"ok": False, "reason": "missing ROI"}
                continue
            sl = getattr(roi, "slice_index", None)
            if sl is None:
                summary[tag] = {"ok": False, "reason": "missing slice index"}
                continue
            summary[tag] = {
                "ok": True,
                "slice": int(sl),
                "x": float(roi.center.x()),
                "y": float(roi.center.y()),
                "radius": int(roi.radius),
            }
        return summary


    def auto_place_with_totalseg(self):
        if self.ct_data is None or self.ct_img is None:
            QMessageBox.information(self, "CT needed", "Load CT first.")
            return
        if not TOTALSEG_AVAILABLE:
            QMessageBox.warning(
                self,
                "TotalSegmentator",
                "TotalSegmentator is not installed in this Python environment. Click 'Check TotalSegmentator Setup' for the expected installation.",
            )
            return
        try:
            td = self._ensure_temp_nifti_dir()
            out_path = os.path.join(td, "ct_for_totalseg.nii.gz")
            sitk.WriteImage(self.ct_img, out_path)
            self.ct_path = out_path
            self._last_totalseg_ct_path = out_path
            self.results.append(f"[Auto-Seg] Exported CT NIfTI: {out_path}")
        except Exception as e:
            QMessageBox.critical(self, "CT export failed", f"Could not prepare CT NIfTI for TotalSegmentator.\n{e}")
            return

        if self.seg_auto_clear:
            self._clear_existing_auto_rois()

        cache_dir = self._study_cache_dir()
        out_dir = cache_dir or tempfile.mkdtemp(prefix="totalseg_")
        self._last_totalseg_out_dir = out_dir
        self.results.append(f"[Auto-Seg] Output folder: {out_dir}")
        cached = self._cached_totalseg_files(out_dir)
        if self._required_totalseg_files_present(cached):
            self._set_busy(True, "Using cached TotalSegmentator masks…")
            QTimer.singleShot(50, lambda: self._on_totalseg_done(cached, out_dir))
            return

        self._seg_worker = SegWorker(self.ct_path, out_dir, fast=self.seg_fast)
        self._seg_worker.stage.connect(lambda s: self._set_busy(True, s))
        self._seg_worker.ok.connect(lambda files: self._on_totalseg_done(files, out_dir))
        self._seg_worker.fail.connect(self._on_totalseg_fail)
        self._set_busy(True, f"Launching TotalSegmentator on {os.path.basename(self.ct_path)} …")
        self._seg_worker.start()

    def run_full_auto_workflow(self):

        if self.ct_data is None:
            QMessageBox.warning(self, "Auto Workflow", "Load CT first.")
            return
        self._open_dxa_after_autoplace = True
        self.auto_place_with_totalseg()

    def _on_totalseg_fail(self, msg: str):
        self._open_dxa_after_autoplace = False
        self._set_busy(False, "Failed")
        try:
            self.results.append("[Auto-Seg] Segmentation failed.")
            if getattr(self, "_last_totalseg_ct_path", None):
                self.results.append(f"[Auto-Seg] CT NIfTI: {self._last_totalseg_ct_path}")
            if getattr(self, "_last_totalseg_out_dir", None):
                self.results.append(f"[Auto-Seg] Output dir: {self._last_totalseg_out_dir}")
            self.results.append(msg)
        except Exception:
            pass
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("TotalSegmentator")
        box.setText("Segmentation failed.")
        box.setInformativeText("The CT was exported to a temporary NIfTI and the CLI returned an error. Open Details to see the exact command, stdout, stderr, and temp paths.")
        box.setDetailedText(msg)
        box.exec()



    def _place_fallback_roi_from_mask_path(self, mask_path: str, tag: str, shrink: float = 0.50):
        if not SITK_AVAILABLE:
            raise RuntimeError("SimpleITK required for fallback mask placement.")
        if self.ct_data is None or self.ct_img is None:
            raise RuntimeError("Load CT first.")
        m = sitk.ReadImage(mask_path)
        m = self._to_lps(m)
        m = self._resample_like(m, self.ct_img, nearest=True)
        marr = sitk.GetArrayFromImage(m).transpose(2, 1, 0)
        marr = (marr > 0.5).astype(np.uint8)
        return self._place_fallback_roi_from_mask_array(marr, tag=tag, shrink=shrink)

    def _place_emergency_roi_from_mask_array(self, marr: np.ndarray, tag: str, shrink: float = 0.40):
        if self.ct_data is None:
            raise RuntimeError("Load CT first.")
        marr = (marr > 0).astype(np.uint8)
        if marr.shape != self.ct_data.shape:
            raise RuntimeError(f"Emergency mask shape {marr.shape} != CT shape {self.ct_data.shape}")
        idx = np.argwhere(marr > 0)
        if idx.size == 0:
            raise RuntimeError(f"No voxels in emergency mask for {tag}")

        zs = np.unique(idx[:, 2])
        zs.sort()
        target_z = int(np.median(zs))
        sl = marr[:, :, target_z].astype(bool)
        if int(sl.sum()) < 10:
            # choose closest non-empty slice to median
            best = None
            best_dist = None
            for z in zs:
                s = marr[:, :, int(z)].astype(bool)
                a = int(s.sum())
                if a < 10:
                    continue
                dist = abs(int(z) - target_z)
                if best is None or dist < best_dist:
                    best = (int(z), s)
                    best_dist = dist
            if best is None:
                raise RuntimeError(f"{tag}: emergency slice selection failed")
            target_z, sl = best

        yx = np.argwhere(sl)
        if yx.size == 0:
            raise RuntimeError(f"{tag}: emergency slice empty")

        cy = int(np.round(np.mean(yx[:, 0])))
        cx = int(np.round(np.mean(yx[:, 1])))
        x0, x1 = int(yx[:, 1].min()), int(yx[:, 1].max())
        y0, y1 = int(yx[:, 0].min()), int(yx[:, 0].max())
        box_r = 0.5 * min((x1 - x0 + 1), (y1 - y0 + 1))
        area_r = math.sqrt(float(sl.sum()) / max(math.pi, 1e-6))
        r = int(max(6, min(22, min(box_r, area_r) * float(shrink))))
        return self._new_axial_roi(cy, cx, target_z, r, tag)

    def _place_emergency_roi_from_mask_path(self, mask_path: str, tag: str, shrink: float = 0.40):
        if not SITK_AVAILABLE:
            raise RuntimeError("SimpleITK required for emergency mask placement.")
        if self.ct_data is None or self.ct_img is None:
            raise RuntimeError("Load CT first.")
        m = sitk.ReadImage(mask_path)
        m = self._to_lps(m)
        m = self._resample_like(m, self.ct_img, nearest=True)
        marr = sitk.GetArrayFromImage(m).transpose(2, 1, 0)
        marr = (marr > 0.5).astype(np.uint8)
        return self._place_emergency_roi_from_mask_array(marr, tag=tag, shrink=shrink)

    def _on_totalseg_done(self, files: dict, out_dir: str):
        self._set_busy(False, "Done — placing ROIs…")
        placed = 0
        placement_notes = []

        for name in ["L1", "L2", "L3", "L4"]:
            path = files.get(name)
            if not path:
                placement_notes.append(f"{name}: mask missing")
                continue
            try:
                roi = self._place_from_mask_path(path, name=name, shrink=self.seg_shrink)
                if roi:
                    placed += 1
                    placement_notes.append(f"{name}: placed")
                    self._mark_target_status(name, "auto")
                    continue
            except Exception as e:
                logging.warning("place_from_mask failed for %s: %s", name, e)
                placement_notes.append(f"{name}: primary placement failed ({e})")
            try:
                roi = self._place_fallback_roi_from_mask_path(
                    path,
                    tag=name,
                    shrink=max(0.42, float(self.seg_shrink) * 0.85),
                )
                if roi:
                    placed += 1
                    placement_notes.append(f"{name}: fallback placed")
                    self._mark_target_status(name, "fallback")
            except Exception as e:
                logging.warning("fallback vertebra placement failed for %s: %s", name, e)
                placement_notes.append(f"{name}: fallback failed ({e})")
                try:
                    self._place_emergency_roi_from_mask_path(path, tag=name, shrink=0.38)
                    placed += 1
                    placement_notes.append(f"{name}: emergency centroid fallback placed")
                    self._mark_target_status(name, "fallback")
                except Exception as e3:
                    placement_notes.append(f"{name}: emergency fallback failed ({e3})")

        psoas_placed = 0
        if self.seg_place_psoas:
            for key in ["PSOAS_L", "PSOAS_R"]:
                p = files.get(key)
                if not p:
                    placement_notes.append(f"{key}: mask missing")
                    continue
                try:
                    self._place_simple_roi_from_mask_path(p, tag=key, shrink=0.50)
                    psoas_placed += 1
                    placement_notes.append(f"{key}: placed")
                except Exception as e:
                    logging.warning("place_psoas failed for %s: %s", key, e)
                    placement_notes.append(f"{key}: primary placement failed ({e})")
                    try:
                        self._place_fallback_roi_from_mask_path(p, tag=key, shrink=0.45)
                        psoas_placed += 1
                        placement_notes.append(f"{key}: fallback placed")
                    except Exception as e2:
                        logging.warning("fallback psoas placement failed for %s: %s", key, e2)
                        placement_notes.append(f"{key}: fallback failed ({e2})")
                        try:
                            self._place_emergency_roi_from_mask_path(p, tag=key, shrink=0.40)
                            psoas_placed += 1
                            placement_notes.append(f"{key}: emergency centroid fallback placed")
                        except Exception as e3:
                            placement_notes.append(f"{key}: emergency fallback failed ({e3})")

        hip_placed = 0
        for site, key in [("FN_L", "FEMUR_L"), ("FN_R", "FEMUR_R")]:
            p = files.get(key)
            if not p:
                placement_notes.append(f"{site}: mask missing")
                continue
            try:
                self._place_femoral_neck_roi_from_femur_mask_path(p, tag=site, shrink=0.45)
                hip_placed += 1
                placement_notes.append(f"{site}: placed")
                self._mark_target_status(site, "auto")
            except Exception as e:
                logging.warning("place_femoral_neck failed for %s: %s", site, e)
                placement_notes.append(f"{site}: primary placement failed ({e})")
                try:
                    self._place_femoral_neck_fallback_from_mask_path(p, tag=site, shrink=0.34)
                    hip_placed += 1
                    placement_notes.append(f"{site}: neck-corridor fallback placed from femur mask")
                    self._mark_target_status(site, "fallback")
                except Exception as e2:
                    logging.warning("fallback femoral placement failed for %s: %s", site, e2)
                    placement_notes.append(f"{site}: fallback failed ({e2})")
                    try:
                        self._place_emergency_roi_from_mask_path(p, tag=site, shrink=0.36)
                        hip_placed += 1
                        placement_notes.append(f"{site}: emergency centroid fallback placed")
                        self._mark_target_status(site, "fallback")
                    except Exception as e3:
                        placement_notes.append(f"{site}: emergency fallback failed ({e3})")

        fat_placed = 0
        if self.seg_place_fat:
            try:
                z_disp = int(self.slider_axial.value())
                if self.vertebra_masks.get("L3") is not None:
                    idx = np.argwhere(self.vertebra_masks["L3"] > 0)
                    if idx.size > 0:
                        cz = int(np.median(idx[:, 2]))
                        z_disp = self._axial_data_to_disp_z(cz)
                self._auto_place_fat_roi(int(z_disp), tag="FAT", r_mm=15.0)
                fat_placed = 1
                placement_notes.append("FAT: placed")
            except Exception as e:
                logging.warning("place_fat failed: %s", e)
                placement_notes.append(f"FAT: failed ({e})")

        missing_required = []
        c_ax = self.canvas_axial
        if c_ax is not None:
            present = {getattr(r, "tag", "") for _cc, r in self._iter_all_roi_pairs()}
            missing_required = [t for t in self._required_target_labels() if t not in present]
        if missing_required:
            created = self._ensure_required_roi_placeholders()
            if created:
                placement_notes.append("Manual placeholders created for: " + ", ".join(created))

        self._update_roi_target_status()
        self._set_busy(False, "Placement complete")

        measured = 0
        for name in self.site_names:
            c_src, roi_src = self._find_latest_roi_by_exact_tag_anywhere(name)
            if roi_src is None:
                continue
            if c_src is self.canvas_axial:
                roi_ax = roi_src
            else:
                coords = self._canvas_roi_to_volume(c_src, roi_src)
                if coords is None:
                    continue
                x, y, z = coords
                ax_pt, ax_sl = self._volume_to_canvas_center("axial", x, y, z)
                roi_ax = SphericalROI(ax_pt.x(), ax_pt.y(), int(getattr(roi_src, "radius", 18)), tag=name, slice_index=int(ax_sl))
                roi_ax.locked_to_slice = True
                roi_ax.follow_on_scroll = False
            if getattr(roi_ax, "slice_index", None) is not None:
                self.slider_axial.setValue(int(roi_ax.slice_index))
            res = self._analyze_roi(self.canvas_axial, roi_ax, site=name)
            if res:
                self.vertebral[name] = res
                self._display_results(res, name)
                measured += 1

        summary = self._validate_auto_roi_summary()
        failed = [k for k, v in summary.items() if not v.get("ok")]

        msg = f"Placed {placed} vertebra ROIs"
        if hip_placed:
            msg += f" + {hip_placed} femoral neck ROIs"
        if psoas_placed:
            msg += f" + {psoas_placed} psoas ROI(s)"
        if fat_placed:
            msg += " + FAT ROI"
        msg += f"\nAuto-measured {measured} sites."
        if failed:
            msg += "\nMissing/failed sites: " + ", ".join(failed)
        if placement_notes:
            msg += "\n\nPlacement details:\n- " + "\n- ".join(placement_notes[:12])

        if placed > 0 or hip_placed > 0:
            QMessageBox.information(
                self,
                "Auto-ROI",
                msg + "\n\nReview the placements and manually drag/resize any ROI that needs correction.",
            )
            if self._open_dxa_after_autoplace:
                self._open_dxa_after_autoplace = False
                QTimer.singleShot(0, self.open_dxa_projection_dialog)
        else:
            self._open_dxa_after_autoplace = False
            QMessageBox.warning(
                self,
                "Auto-ROI",
                f"No ROIs could be placed.\nMasks found: {sorted(list(files.keys())) if isinstance(files, dict) else []}\n\nPlacement details:\n- " + "\n- ".join(placement_notes[:12]),
            )


    # ----------------------------------------------------------------
    # Registration
    # ----------------------------------------------------------------
    def _kickoff_registration(self):
        if self.pet_data is None or self.ct_data is None:
            QMessageBox.warning(self, "Registration", "Load CT and PET first.")
            return
        if hasattr(self, "worker") and self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait()
        self.worker = RegistrationWorker(self.ct_img, self.pet_img, self.ct_data, self.pet_data)
        self.worker.stage.connect(lambda s: self._set_busy(True, s))
        self.worker.done.connect(self._on_reg_done)
        self.worker.error.connect(self._on_reg_error)
        self._set_busy(True, "Starting…")
        self.worker.start()

    def _on_reg_done(self, payload):
        try:
            arr = payload["arr"]
            info = payload.get("info", "Done")
            tx = payload.get("tx", None)
        except Exception:
            arr = payload
            info = "Done"
            tx = None
        self.registered_pet = arr
        self.reg_tx = tx
        self._set_busy(False, info)
        self._update_pet_radio_state()
        self._update_all()
        QMessageBox.information(self, "Registration", f"Registration ready.\n{info}")

    def _on_reg_error(self, msg):
        self._set_busy(False, "Failed")
        QMessageBox.critical(self, "Registration failed", msg)

    def _set_busy(self, busy: bool, stage_text: str):
        self.progress.setVisible(busy)
        self.lbl_stage.setText(stage_text)
        try:
            self.btn_totalseg.setEnabled((not busy) and TOTALSEG_AVAILABLE)
            self.btn_auto_workflow.setEnabled((not busy) and TOTALSEG_AVAILABLE)
        except Exception:
            pass
        try:
            if hasattr(self, "results") and stage_text:
                self.results.append(f"[Auto-Seg] {stage_text}")
        except Exception:
            pass
        QApplication.processEvents()

    # ----------------------------------------------------------------
    # QC  — FIX 4: All _find_roi_by_tag callers now unpack tuples
    # ----------------------------------------------------------------
    def run_qc(self):
        if self.ct_data is None:
            QMessageBox.warning(self, "QC", "Load CT and run Auto-place first.")
            return
        qc = self._compute_qc_all()
        self.qc_results = qc
        self._populate_qc_table(qc)
        QMessageBox.information(self, "QC", f"QC updated for {len(qc)} sites.")

    def export_qc_csv(self):
        if not self.qc_results:
            QMessageBox.warning(self, "QC Export", "Run QC first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save QC CSV", "", "CSV Files (*.csv)")
        if not path:
            return
        rows = []
        for site in self.site_names:
            r = self.qc_results.get(site, {})
            rows.append(
                {
                    "Site": site,
                    "Status": r.get("status", ""),
                    "Reasons": ";".join(r.get("reasons", [])),
                    "Overlap": f"{r.get('overlap', float('nan')):.3f}"
                    if r.get("overlap") is not None
                    else "",
                    "MetalFrac": f"{r.get('metal_frac', float('nan')):.4f}"
                    if r.get("metal_frac") is not None
                    else "",
                    "HasMask": int(bool(r.get("has_mask"))),
                    "HasROI": int(bool(r.get("has_roi"))),
                    "Timestamp": datetime.now().isoformat(timespec="seconds"),
                }
            )
        with open(path, "w", newline="") as f:
            wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wri.writeheader()
            wri.writerows(rows)
        QMessageBox.information(self, "QC Export", f"Saved {len(rows)} rows.")

    def _populate_qc_table(self, qc: Dict[str, dict]):
        self.qc_table.setRowCount(0)
        for site in self.site_names:
            r = qc.get(site, {})
            row = self.qc_table.rowCount()
            self.qc_table.insertRow(row)
            self.qc_table.setItem(row, 0, QTableWidgetItem(site))
            self.qc_table.setItem(row, 1, QTableWidgetItem(r.get("status", "")))
            self.qc_table.setItem(row, 2, QTableWidgetItem(";".join(r.get("reasons", []))))
            self.qc_table.setItem(
                row, 3,
                QTableWidgetItem("" if r.get("overlap") is None else f"{r.get('overlap'):.3f}"),
            )
            self.qc_table.setItem(
                row, 4,
                QTableWidgetItem(
                    "" if r.get("metal_frac") is None else f"{r.get('metal_frac'):.4f}"
                ),
            )
            self.qc_table.setItem(row, 5, QTableWidgetItem("Y" if r.get("has_mask") else "N"))
            self.qc_table.setItem(row, 6, QTableWidgetItem("Y" if r.get("has_roi") else "N"))

    def _compute_qc_all(self) -> Dict[str, dict]:
        qc: Dict[str, dict] = {}
        for site in self.site_names:
            qc[site] = self._compute_qc_for_site(site)
        return qc

    def _compute_qc_for_site(self, site: str) -> dict:
        has_mask, has_roi = False, False
        reasons: List[str] = []
        overlap = None
        metal_frac = None

        # FIX 4: unpack tuple from _find_roi_by_tag
        _canvas_for_roi, roi = self._find_roi_by_tag(site)
        if roi is None:
            reasons.append("MissingROI")
        else:
            has_roi = True

        mask3d = self._get_site_mask_3d(site)
        if mask3d is None:
            reasons.append("MissingSegMask")
        else:
            has_mask = True

        if has_roi and has_mask and self.ct_data is not None:
            z_disp = int(
                np.clip(
                    roi.slice_index if roi.slice_index is not None else self.slider_axial.value(),
                    0,
                    self.ct_data.shape[2] - 1,
                )
            )
            z_idx = (self.ct_data.shape[2] - 1 - z_disp) if self.axial_invert_z else z_disp
            ct_slice = self.ct_data[:, :, z_idx]
            mask_slice = mask3d[:, :, z_idx].astype(bool)
            roi_mask = self._roi_mask_on_slice(ct_slice.shape, roi, view="axial")

            if roi_mask.sum() > 0:
                overlap = float((roi_mask & mask_slice).sum()) / float(roi_mask.sum())
                if overlap < float(self.qc_overlap_min):
                    reasons.append("LowROIMaskOverlap")
                vals = ct_slice[roi_mask]
                if vals.size:
                    metal_frac = float((vals >= float(self.qc_metal_hu_thr)).sum()) / float(
                        vals.size
                    )
                    if metal_frac > float(self.qc_metal_frac_thr):
                        reasons.append("MetalArtifact")

        if site in ("FN_L", "FN_R"):
            key = "FEMUR_L" if site == "FN_L" else "FEMUR_R"
            if self._femur_cropped.get(key, False):
                reasons.append("FemurCropped")

        status = "PASS" if not reasons else "FAIL"
        return {
            "site": site,
            "status": status,
            "reasons": reasons,
            "overlap": overlap,
            "metal_frac": metal_frac,
            "has_mask": has_mask,
            "has_roi": has_roi,
        }

    # ---- QC overlay ----
    def _on_qc_overlay_toggle(self, checked: bool):
        self.qc_overlay_enabled = bool(checked)
        self._qc_overlay_cache = None
        for c in self._canvases():
            c.update()

    def _on_qc_overlay_site_changed(self, txt_site: str):
        s = (txt_site or "").strip()
        self.qc_overlay_site = None if s.lower().startswith("selected") else s
        self._qc_overlay_cache = None
        for c in self._canvases():
            c.update()

    def _focus_overlay_site_roi(self):
        site = self.qc_overlay_site
        if not site:
            QMessageBox.information(self, "Overlay", "Select a specific site to focus.")
            return
        # FIX 4: unpack tuple
        _canvas_for_roi, roi = self._find_roi_by_tag(site)
        if roi is None:
            QMessageBox.warning(self, "Overlay", f"No ROI tagged '{site}' found.")
            return
        if getattr(roi, "slice_index", None) is not None:
            try:
                self.slider_axial.setValue(int(roi.slice_index))
            except Exception:
                pass
        self.canvas_axial.selected_roi = roi
        self.canvas_axial.update()
        self._qc_overlay_cache = None

    def _get_qc_overlay_for_canvas(self, canvas):
        if not self.qc_overlay_enabled:
            return None
        if self.ct_data is None or canvas is None:
            return None

        site = self.qc_overlay_site
        roi = None
        if site:
            # FIX 4: unpack tuple
            _canvas_for_roi, roi = self._find_roi_by_tag(site)
        if roi is None:
            roi = canvas.selected_roi
            site = getattr(roi, "tag", None) if roi is not None else None

        if site not in self.site_names:
            return None
        if roi is None:
            return None

        key = (
            canvas.view,
            int(canvas.current_slice),
            str(site),
            float(roi.center.x()),
            float(roi.center.y()),
            int(roi.radius),
            bool(canvas.flip_h),
            bool(canvas.flip_v),
            int(canvas.rotation_angle),
            bool(self.axial_flip_ud),
            bool(self.mask_enable),
        )
        if isinstance(self._qc_overlay_cache, dict) and self._qc_overlay_cache.get("key") == key:
            return self._qc_overlay_cache

        # Compute raw overlay masks
        raw = self._compute_overlay_masks_raw(canvas, roi, site)
        if not isinstance(raw, dict):
            return None

        roi_disp = self._raw2d_to_display(canvas, raw["roi_mask"])
        site_disp = self._raw2d_to_display(canvas, raw["site_mask"])
        meas_disp = self._raw2d_to_display(canvas, raw["meas_mask"])
        if roi_disp is None or site_disp is None or meas_disp is None:
            return None

        h, w = site_disp.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[site_disp] = np.array([0, 0, 255, 50], dtype=np.uint8)
        rgba[meas_disp] = np.array([0, 255, 0, 90], dtype=np.uint8)
        inter = roi_disp & meas_disp
        rgba[inter] = np.array([255, 255, 0, 140], dtype=np.uint8)

        buf = rgba.tobytes()
        qimg = QImage(buf, w, h, 4 * w, QImage.Format.Format_RGBA8888)
        info = f"QC Overlay {site} | overlap={raw['overlap']:.2f}"
        self._qc_overlay_cache = {"key": key, "rgba": rgba, "buf": buf, "qimg": qimg, "info": info}
        return self._qc_overlay_cache

    def _raw2d_to_display(self, canvas, raw2d: np.ndarray):
        if raw2d is None:
            return None
        s = raw2d.T
        if canvas.view == "axial" and bool(self.axial_flip_ud):
            s = np.flipud(s)
        if canvas.flip_h:
            s = np.fliplr(s)
        if canvas.flip_v:
            s = np.flipud(s)
        if canvas.rotation_angle != 0 and ndi_rotate is not None:
            s = ndi_rotate(
                s.astype(np.float32), canvas.rotation_angle,
                reshape=False, order=0, mode="constant", cval=0.0,
            )
            s = s > 0.5
        return s.astype(bool)

    def _compute_overlay_masks_raw(self, canvas, roi, site: str):
        if self.ct_data is None or canvas is None or roi is None or not site:
            return None
        site_mask3d = self._get_site_mask_3d(site)
        if not isinstance(site_mask3d, np.ndarray) or site_mask3d.shape != self.ct_data.shape:
            return None

        if canvas.view == "axial":
            z_disp = int(np.clip(canvas.current_slice, 0, self.ct_data.shape[2] - 1))
            slice_idx = (self.ct_data.shape[2] - 1 - z_disp) if self.axial_invert_z else z_disp
            ct_slice = self.ct_data[:, :, slice_idx]
            site2d = site_mask3d[:, :, slice_idx].astype(bool)
        elif canvas.view == "coronal":
            slice_idx = int(np.clip(canvas.current_slice, 0, self.ct_data.shape[1] - 1))
            ct_slice = self.ct_data[:, slice_idx, :]
            site2d = site_mask3d[:, slice_idx, :].astype(bool)
        else:
            slice_idx = int(np.clip(canvas.current_slice, 0, self.ct_data.shape[0] - 1))
            ct_slice = self.ct_data[slice_idx, :, :]
            site2d = site_mask3d[slice_idx, :, :].astype(bool)

        untransformed_center = canvas._inverse_map(roi.center.x(), roi.center.y())
        i0 = int(round(untransformed_center.x()))
        i1 = int(round(untransformed_center.y()))
        h, w = ct_slice.shape
        if not (0 <= i0 < h and 0 <= i1 < w):
            return None

        rows, cols = np.ogrid[:h, :w]
        sx, sy, sz = self.ct_spacing if self.ct_spacing else (1, 1, 1)
        pixel_ar = 1.0
        if canvas.view == "axial":
            pixel_ar = sy / sx
        elif canvas.view == "coronal":
            pixel_ar = sz / sx
        else:
            pixel_ar = sz / sy

        r = int(max(5, roi.radius))
        roi_mask = ((rows - i0) ** 2 + ((cols - i1) / pixel_ar) ** 2) <= (r ** 2)
        if roi_mask.sum() == 0:
            return None

        overlap = float((roi_mask & site2d).sum()) / float(max(1, roi_mask.sum()))
        mask = roi_mask & site2d

        if self.mask_enable:
            within = (ct_slice >= self.mask_hu_min) & (ct_slice <= self.mask_hu_max)
            cand = mask & within
            if cand.sum() >= max(10, int(0.05 * mask.sum())):
                mask = cand

        if self.mask_erode_px > 0 and binary_erosion is not None:
            cand = binary_erosion(mask, iterations=int(self.mask_erode_px))
            if cand.sum() >= max(10, int(0.05 * mask.sum())):
                mask = cand

        return {
            "roi_mask": roi_mask.astype(bool),
            "site_mask": site2d.astype(bool),
            "meas_mask": mask.astype(bool),
            "overlap": float(overlap),
        }

    # ----------------------------------------------------------------
    # Composite / export
    # ----------------------------------------------------------------
    def compute_composite(self):
        names = ["L1", "L2", "L3", "L4"]
        vals = [(n, self.vertebral.get(n)) for n in names if self.vertebral.get(n) is not None]
        if len(vals) < 2:
            QMessageBox.warning(self, "Composite", "Measure at least two of L1–L4 first.")
            return None
        bmds = np.array([v["bmd"] for _, v in vals], float)
        ts = np.array([v["t_score"] for _, v in vals], float)
        zs = np.array([v.get("z_score", np.nan) for _, v in vals], float)
        bmd_comp = float(np.mean(bmds))
        t_comp = float(np.mean(ts))
        z_comp = float(np.nanmean(zs))
        who = "Normal" if t_comp >= -1.0 else ("Osteopenia" if t_comp >= -2.5 else "Osteoporosis")
        lines = [
            "DXA-Style L1–L4 Composite",
            "-" * 34,
            f"Composite BMD: {bmd_comp:.1f} mg/cm³",
            f"Composite T-score: {t_comp:.2f}",
            f"Composite Z-score: {z_comp:.2f}",
            f"Classification: {who}",
        ]
        self.results.append("\n" + "\n".join(lines))
        return {"bmd": bmd_comp, "t_score": t_comp, "z_score": z_comp, "class": who}

    def export_csv(self):
        rows = []
        for n in self.site_names:
            r = self.vertebral.get(n)
            if r:
                rows.append(
                    {
                        "PatientAge": self.patient_age,
                        "PatientSex": self.patient_sex,
                        "Site": n,
                        "MeanHU": f"{r['mean_hu']:.1f}",
                        "BMD_mg_cm3": f"{r['bmd']:.1f}",
                        "TScore": f"{r['t_score']:.2f}",
                        "ZScore": f"{r.get('z_score', float('nan')):.2f}",
                        "Class": r["who_classification"],
                        "Voxels": r.get("nvox", 0),
                        "Mask": r.get("mask_status", ""),
                        "QC_Status": r.get("qc_status", ""),
                        "QC_Reasons": ";".join(r.get("qc_reasons", [])),
                        "CalibSlope": f"{self.cal_slope_eff:.5f}",
                        "CalibIntercept": f"{self.cal_intercept_eff:.2f}",
                        "Timestamp": datetime.now().isoformat(timespec="seconds"),
                    }
                )

        comp = None
        if any(self.vertebral.get(n) for n in ["L1", "L2", "L3", "L4"]):
            comp = self.compute_composite()
        if comp:
            rows.append(
                {
                    "PatientAge": self.patient_age,
                    "PatientSex": self.patient_sex,
                    "Site": "L1-L4 Composite",
                    "MeanHU": "",
                    "BMD_mg_cm3": f"{comp['bmd']:.1f}",
                    "TScore": f"{comp['t_score']:.2f}",
                    "ZScore": f"{comp['z_score']:.2f}",
                    "Class": comp["class"],
                    "Voxels": "",
                    "Mask": "",
                    "QC_Status": "",
                    "QC_Reasons": "",
                    "CalibSlope": f"{self.cal_slope_eff:.5f}",
                    "CalibIntercept": f"{self.cal_intercept_eff:.2f}",
                    "Timestamp": datetime.now().isoformat(timespec="seconds"),
                }
            )

        if not rows:
            QMessageBox.warning(self, "Export CSV", "No measurements to export.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
        if not path:
            return

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        QMessageBox.information(self, "Export CSV", f"Saved {len(rows)} rows.")

    def show_all(self):
        d = QDialog(self)
        d.setWindowTitle("Site Measurements")
        l = QVBoxLayout()
        l.setContentsMargins(12, 12, 12, 12)
        l.setSpacing(6)

        cols = [
            "Site", "Mean HU", "BMD", "T-score", "Z-score", "Class",
            "Voxels", "Mask", "QC", "QC Reasons",
        ]
        t = QTableWidget(len(self.site_names), len(cols))
        t.setHorizontalHeaderLabels(cols)

        for i, s in enumerate(self.site_names):
            t.setItem(i, 0, QTableWidgetItem(s))
            m = self.vertebral.get(s)
            if m:
                t.setItem(i, 1, QTableWidgetItem(f"{m.get('mean_hu', 0):.1f}"))
                t.setItem(i, 2, QTableWidgetItem(f"{m.get('bmd', 0):.1f}"))
                t.setItem(i, 3, QTableWidgetItem(f"{m.get('t_score', 0):.2f}"))
                t.setItem(i, 4, QTableWidgetItem(f"{m.get('z_score', 0):.2f}"))
                t.setItem(i, 5, QTableWidgetItem(str(m.get("who_classification", ""))))
                t.setItem(i, 6, QTableWidgetItem(str(m.get("nvox", 0))))
                t.setItem(i, 7, QTableWidgetItem(str(m.get("mask_status", ""))))
                t.setItem(i, 8, QTableWidgetItem(str(m.get("qc_status", ""))))
                t.setItem(i, 9, QTableWidgetItem(";".join(m.get("qc_reasons", []))))

        t.resizeColumnsToContents()
        t.horizontalHeader().setStretchLastSection(True)
        l.addWidget(t)
        btn = QPushButton("Close")
        btn.clicked.connect(d.accept)
        l.addWidget(btn)
        d.setLayout(l)
        d.exec()

    # ----------------------------------------------------------------
    # Report
    # ----------------------------------------------------------------

    def report(self):
        valid = [v for v in self.vertebral.values() if v is not None]
        if not valid:
            QMessageBox.warning(self, "No Data", "Analyze at least one ROI first.")
            return
        report = f"""QCT BMD ANALYSIS REPORT
=================================
Sex: {'Female' if self.patient_sex=='F' else 'Male'}
Age: {self.patient_age} years
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Calibration:
  Effective: {self._eq_eff_text()}
  Blend λ: {self.cal_blend_lambda:.2f}   Preset: {self.cal_preset}
  Cancellous mask: {'on' if self.mask_enable else 'off'} HU[{self.mask_hu_min:.0f},{self.mask_hu_max:.0f}]

Results:
------------------------------------
"""
        for name in self.site_names:
            res = self.vertebral.get(name)
            if res:
                report += (
                    f"{name}:\n"
                    f"  Mean HU (cancellous): {res['mean_hu']:.1f} ± {res['std_hu']:.1f}\n"
                    f"  BMD: {res.get('bmd',0):.1f} mg/cm³\n"
                    f"  T-score: {res.get('t_score',0):.2f}\n"
                    f"  Z-score: {res.get('z_score',0):.2f}\n"

                    f"  Voxels: {res.get('nvox',0)}\n"
                    f"  QC: {res.get('qc_status','')}\n\n"
                )
        if self.dxa_proj_last:
            d = self.dxa_proj_last
            if "sites" in d:
                report += "DXA-style AP projection:\n"
                for s in d.get("sites", []):
                    report += (
                        f"  {s['site']}: mean HU={s['mean_hu']:.1f}, "
                        f"pseudo-aBMD={s['bmd']:.1f}, "
                        f"T={s['t_score']:.2f}, Z={s['z_score']:.2f}, "
                        f"Class={s['who_classification']}\n"
                    )
                comp = d.get("composite", {})
                if "spine" in comp:
                    s = comp["spine"]
                    report += (
                        f"  Composite L1-L4: pseudo-aBMD={s['bmd']:.1f}, "
                        f"T={s['t_score']:.2f}, Z={s['z_score']:.2f}, "
                        f"Class={s['who_classification']}\n"
                    )
                if "hips_mean" in comp:
                    s = comp["hips_mean"]
                    report += (
                        f"  Bilateral femoral neck mean: pseudo-aBMD={s['bmd']:.1f}, "
                        f"T={s['t_score']:.2f}, Z={s['z_score']:.2f}, "
                        f"Class={s['who_classification']}\n"
                    )
                report += "\n"
            else:
                report += (
                    f"DXA-style AP projection:\n"
                    f"  Projected mean HU: {d['mean_hu']:.1f}\n"
                    f"  Pseudo-aBMD: {d['bmd']:.1f}\n"
                    f"  T-score: {d['t_score']:.2f}\n"
                    f"  Z-score: {d['z_score']:.2f}\n"
                    f"  Class: {d['who_classification']}\n\n"
                )
        self.results.setPlainText(report)
        QMessageBox.information(self, "Report Generated", "Report in results panel.")


    # ----------------------------------------------------------------
    # Calibration helpers
    # ----------------------------------------------------------------
    def _blend_text(self):
        return f"λ = {self.cal_blend_lambda:.2f} → blends preset & ROI"

    def _eq_text(self, slope, intercept, r2=1.0):
        return f"BMD = {slope:.5f} × HU + {intercept:.2f}  (R²≈{r2:.3f})"

    def _eq_eff_text(self):
        return self._eq_text(self.cal_slope_eff, self.cal_intercept_eff, self.cal_r2_eff)

    def _set_preset(self, name: str):
        self.cal_preset = name
        self._recompute_effective_calibration()

    def _on_blend(self, val: float):
        self.cal_blend_lambda = float(np.clip(val, 0.0, 1.0))
        if hasattr(self, "lbl_blend"):
            self.lbl_blend.setText(self._blend_text())
        self._recompute_effective_calibration()

    def _recompute_effective_calibration(self):
        a_p, b_p = self.preset_params[
            "contrast" if self.cal_preset == "contrast" else "native"
        ]
        a_r, b_r = self.cal_roi_slope, self.cal_roi_intercept
        lam = float(np.clip(self.cal_blend_lambda, 0.0, 1.0))
        self.cal_slope_eff = (1.0 - lam) * a_p + lam * a_r
        self.cal_intercept_eff = (1.0 - lam) * b_p + lam * b_r
        self.cal_r2_eff = 1.0
        if hasattr(self, "lbl_eq_eff"):
            self.lbl_eq_eff.setText(self._eq_eff_text())

    def cal_check(self):
        HUs = [120, 150, 180]
        a, b = self.cal_slope_eff, self.cal_intercept_eff
        rows = [
            f"CAL CHECK (a={a:.4f}, b={b:.2f})",
            "HU    →  BMD (mg/cm³)",
        ]
        for hu in HUs:
            rows.append(f"{hu:<5}→  {a*hu + b:6.1f}")
        self.results.append("\n" + "\n".join(rows))

    def recompute_all_with_calibration(self):
        for name in list(self.vertebral.keys()):
            res = self.vertebral.get(name)
            if not res:
                continue
            res = res.copy()
            bmd = max(0.0, self.cal_slope_eff * res["mean_hu"] + self.cal_intercept_eff)
            ya_mu, ya_sd = young_adult_norms(self.patient_sex)
            t = (bmd - ya_mu) / ya_sd
            z_mu, z_sd = age_matched_norms(self.patient_age, self.patient_sex)
            z_score = (bmd - z_mu) / z_sd
            who = "Normal" if t >= -1.0 else ("Osteopenia" if t >= -2.5 else "Osteoporosis")
            res.update({"bmd": bmd, "t_score": t, "z_score": z_score, "who_classification": who})
            self.vertebral[name] = res
            self._display_results(res, name)
        QMessageBox.information(self, "Calibration", "Recomputed all with current calibration.")

    def auto_pick_lambda(self):
        tgt = 140.0
        lam_grid = np.linspace(0, 1, 21)
        best = (self.cal_blend_lambda, float("inf"))
        for lam in lam_grid:
            a_p, b_p = self.preset_params[
                "contrast" if self.cal_preset == "contrast" else "native"
            ]
            a = (1 - lam) * a_p + lam * self.cal_roi_slope
            b = (1 - lam) * b_p + lam * self.cal_roi_intercept
            r = self.vertebral.get("L1")
            if not r:
                continue
            bmd = a * r["mean_hu"] + b
            err = abs(bmd - tgt)
            if err < best[1]:
                best = (lam, err)
        self.cal_blend_lambda = best[0]
        self.blend.setValue(int(self.cal_blend_lambda * 100))
        self._recompute_effective_calibration()
        QMessageBox.information(self, "λ updated", f"Auto-picked λ={self.cal_blend_lambda:.2f}")

    def _open_phantomless(self):
        PhantomlessDialog(self).exec()
        self._recompute_effective_calibration()

    def _retag_selected_roi(self, tag: str):
        c = self._active_canvas()
        if not c or c.selected_roi is None:
            QMessageBox.warning(self, "Calibration ROI", "Select or draw an ROI first.")
            return
        self._lock_roi_to_current_slice(c.selected_roi, c, tag=str(tag))
        c.update()
        self.results.append(f"Tagged and locked ROI as {tag} on slice {c.current_slice}.")


    def _compute_manual_calibration(self):
        c_fat, roi_fat = self._find_roi_by_tag("FAT")
        if roi_fat is None:
            QMessageBox.warning(self, "Calibration", "Place and tag a FAT ROI first.")
            return
        muscle_rois = []
        for key in ("MUSCLE", "PSOAS_L", "PSOAS_R"):
            c_p, roi_p = self._find_roi_by_tag(key)
            if roi_p is not None:
                muscle_rois.append((c_p, roi_p, key))
        if not muscle_rois:
            QMessageBox.warning(self, "Calibration", "Place and tag at least one MUSCLE or PSOAS ROI first.")
            return
        fat_res = self._analyze_roi(c_fat, roi_fat, site="FAT")
        if not fat_res:
            QMessageBox.warning(self, "Calibration", "Could not analyze FAT ROI.")
            return
        m_vals = []
        m_tags = []
        for c_p, roi_p, key in muscle_rois:
            res = self._analyze_roi(c_p, roi_p, site=key)
            if res:
                m_vals.append(res["mean_hu"])
                m_tags.append(f"{key}={res['mean_hu']:.1f}")
        if not m_vals:
            QMessageBox.warning(self, "Calibration", "Could not analyze MUSCLE / PSOAS ROI(s).")
            return
        hu_fat = float(fat_res["mean_hu"])
        hu_muscle = float(np.mean(m_vals))
        if abs(hu_muscle - hu_fat) < 1e-3:
            QMessageBox.warning(self, "Calibration", "FAT and MUSCLE HU are too similar to fit a calibration line.")
            return
        a = (float(self.cal_psoas_bmd_anchor) - float(self.cal_fat_bmd_anchor)) / (hu_muscle - hu_fat)
        b = float(self.cal_fat_bmd_anchor) - a * hu_fat
        if hu_fat >= hu_muscle:
            QMessageBox.warning(
                self,
                "Calibration",
                (
                    f"Calibration failed: FAT HU ({hu_fat:.1f}) is not lower than MUSCLE/PSOAS HU ({hu_muscle:.1f}).\n"
                    "This usually means the crosshair / ROI mapping is off or the ROIs are misplaced."
                )
            )
            return

        self.cal_roi_slope = float(a)
        self.cal_roi_intercept = float(b)
        self.cal_blend_lambda = 1.0
        try:
            self.blend.setValue(100)
        except Exception:
            pass
        self._recompute_effective_calibration()
        self.results.append("\n" + "\n".join([
            "MANUAL PHANTOMLESS CALIBRATION",
            f"FAT HU: {hu_fat:.1f}",
            f"MUSCLE HU: {hu_muscle:.1f} ({', '.join(m_tags)})",
            f"Anchors: fat={self.cal_fat_bmd_anchor:.1f}, muscle={self.cal_psoas_bmd_anchor:.1f} mg/cm³",
            f"ROI equation: {self._eq_text(self.cal_roi_slope, self.cal_roi_intercept)}",
        ]))
        QMessageBox.information(self, "Calibration", "Updated ROI-based calibration from manual FAT + MUSCLE / PSOAS ROIs.")

    # ----------------------------------------------------------------
    # FIX 3: DXA-like AP Projection — bone emphasis + correct orientation
    # ----------------------------------------------------------------
    def _make_ap_projection(self, z_min=None, z_max=None):
        """Create an AP projection image from CT data.

        Display image:
            full-body coronal-style MIP for publication-friendly visibility.
        Quantification image:
            bone-only mean projection used for DXA-style pseudo-aBMD calculations.
        """
        if self.ct_data is None:
            return None, None

        data = self.ct_data.astype(np.float32)
        nz = data.shape[2]
        if z_min is None or z_max is None:
            z_min, z_max = 0, nz - 1
        z_min = int(max(0, min(int(z_min), nz - 1)))
        z_max = int(max(z_min, min(int(z_max), nz - 1)))
        sub = data[:, :, z_min:z_max + 1]

        hu_min = float(self.ap_hu_min) if self.ap_bone_only_zero_clip else -300.0
        hu_max = float(self.ap_hu_max)
        if self.ap_auto_hu_min:
            hu_min = max(hu_min, 100.0)

        bone = (sub >= hu_min) & (sub <= hu_max)
        masked = np.where(bone, sub, 0.0)
        thickness = bone.sum(axis=1).astype(np.float32)
        thickness_safe = np.maximum(thickness, 1.0)
        proj_bone_mean = masked.sum(axis=1) / thickness_safe
        proj_bone_mean[thickness == 0] = 0.0

        # Display image should remain anatomically readable and publication-friendly.
        # Use a full-body MIP-like projection rather than the bone-only mean image.
        body_clip = np.clip(sub, -250.0, 1800.0)
        proj_body_mip = np.max(body_clip, axis=1)

        bone_coverage = float(np.count_nonzero(proj_bone_mean > 0.0)) / float(max(1, proj_bone_mean.size))
        bone_dynamic = float(np.nanmax(proj_bone_mean) - np.nanmin(proj_bone_mean)) if proj_bone_mean.size else 0.0

        proj_quant = proj_bone_mean if self.ap_bone_only_zero_clip else proj_body_mip
        proj_display = proj_body_mip

        proj_display = np.flipud(proj_display.T)
        proj_quant = np.flipud(proj_quant.T)

        meta = {
            "z_min": z_min,
            "z_max": z_max,
            "display_proj": proj_display,
            "quant_proj": proj_quant,
            "used_body_fallback": False,
            "bone_coverage": bone_coverage,
            "bone_dynamic": bone_dynamic,
        }
        return proj_display, meta

    def _ensure_dxa_projection_results(self, force: bool = False) -> bool:
        dxa = self.dxa_proj_last or {}
        has_dxa = bool(dxa.get("sites")) or bool((dxa.get("composite", {}) or {}).get("spine")) or bool((dxa.get("composite", {}) or {}).get("hips_mean"))
        if has_dxa and not force:
            return True
        if self.ct_data is None:
            return False
        if not any(self._get_target_spec(tag) for tag in self._required_target_labels()):
            return False

        proj, meta = self._build_dxa_quant_payload()
        if proj is None:
            return False

        dlg = DXAProjectionDialog(self, proj, meta)
        try:
            dlg.measure_all()
        finally:
            dlg.deleteLater()
        dxa = self.dxa_proj_last or {}
        return bool(dxa.get("sites")) or bool((dxa.get("composite", {}) or {}).get("spine")) or bool((dxa.get("composite", {}) or {}).get("hips_mean"))

    def _roi_based_projection_range(self):
        if self.ct_data is None:
            return None, None
        z_values = []
        for tag in ["L1", "L2", "L3", "L4", "FN_L", "FN_R"]:
            _canvas, roi = self._find_roi_by_tag(tag)
            if roi is None:
                continue
            sl = getattr(roi, "slice_index", None)
            if sl is not None:
                z_values.append(int(sl))
        if not z_values:
            return None, None
        pad = 8
        zmin = max(0, min(z_values) - pad)
        zmax = min(self.ct_data.shape[2] - 1, max(z_values) + pad)
        return int(zmin), int(zmax)

    def _build_dxa_projection_payload(self):
        if self.ct_data is None:
            return None, None
        display_proj, display_meta = self._build_dxa_display_payload()
        quant_proj, quant_meta = self._build_dxa_quant_payload()
        if display_proj is None:
            return None, None
        meta = dict(display_meta or {})
        meta["display_proj"] = display_proj
        if quant_proj is not None:
            meta["quant_proj"] = quant_proj
            meta["quant_meta"] = quant_meta or {}
        return display_proj, meta

    def _collect_projected_dxa_preview_rois(self, meta: dict) -> List[dict]:
        out = []
        if self.ct_data is None:
            return out
        quant_meta = (meta or {}).get("quant_meta", {}) or {}
        z_max = int(quant_meta.get("z_max", (meta or {}).get("z_max", self.ct_data.shape[2] - 1)))
        px_x, px_y, px_z = self.ct_spacing if self.ct_spacing is not None else (1.0, 1.0, 1.0)
        for tag in ("L1", "L2", "L3", "L4", "FN_L", "FN_R"):
            spec = self._get_target_spec(tag)
            if not spec:
                continue
            x = float(np.clip(int(spec.get("x", 0)), 0, self.ct_data.shape[0] - 1))
            z = float(np.clip(int(spec.get("z", 0)), 0, self.ct_data.shape[2] - 1))
            r = int(max(5, spec.get("radius", 18)))
            proj_x = x
            proj_y = float(z_max - z)
            rx_mm = float(r) * float(px_x)
            rz_px = max(5, int(round(rx_mm / max(px_z, 1e-6))))
            out.append({"tag": tag, "x": proj_x, "y": proj_y, "r": rz_px})
        return out

    def _measure_projected_dxa_preview_roi(self, proj: np.ndarray, roi_info: dict) -> Optional[dict]:
        try:
            rows, cols = np.ogrid[:proj.shape[0], :proj.shape[1]]
            x = float(roi_info["x"])
            y = float(roi_info["y"])
            r = float(roi_info["r"])
            mask = ((cols - x) ** 2 + (rows - y) ** 2) <= (r ** 2)
            vals = proj[mask]
            if vals.size < 10:
                return None
            bone_vals = vals[vals > 0] if self.ap_bone_only_zero_clip else vals
            if bone_vals.size < 5:
                bone_vals = vals
            mean_hu = float(np.mean(bone_vals))
            tag = str(roi_info.get("tag", "ROI"))
            a, b = self._proj_calibration_for_site(tag)
            pseudo_bmd = max(0.0, a * mean_hu + b)
            if tag in ("FN_L", "FN_R"):
                ya_mu = float(getattr(self, "proj_fn_young_mean", 235.0))
                ya_sd = max(1e-6, float(getattr(self, "proj_fn_young_sd", 40.0)))
                z_mu = float(getattr(self, "proj_fn_age_mean", 205.0))
                z_sd = max(1e-6, float(getattr(self, "proj_fn_age_sd", 35.0)))
            else:
                ya_mu = float(getattr(self, "proj_spine_young_mean", 210.0))
                ya_sd = max(1e-6, float(getattr(self, "proj_spine_young_sd", 35.0)))
                z_mu = float(getattr(self, "proj_spine_age_mean", 180.0))
                z_sd = max(1e-6, float(getattr(self, "proj_spine_age_sd", 30.0)))
            t = (pseudo_bmd - ya_mu) / ya_sd
            z = (pseudo_bmd - z_mu) / z_sd
            return {"site": tag, "mean_hu": mean_hu, "bmd": pseudo_bmd, "t_score": t, "z_score": z}
        except Exception:
            return None

    def _render_dxa_preview_pixmap(self, display_proj: np.ndarray, quant_proj: np.ndarray, meta: dict) -> Optional[QPixmap]:
        try:
            arr = np.asarray(display_proj, np.float32)
            if arr.ndim != 2 or arr.size == 0:
                return None
            lo = float(np.nanpercentile(arr, 1.0))
            hi = float(np.nanpercentile(arr, 99.5))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo = float(np.nanmin(arr))
                hi = float(np.nanmax(arr))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                return None
            img8 = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
            img8 = (img8 * 255.0 + 0.5).astype(np.uint8)
            h, w = img8.shape
            qimg = QImage(img8.tobytes(), w, h, w, QImage.Format.Format_Grayscale8).copy()
            pm = QPixmap.fromImage(qimg)
            painter = QPainter(pm)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            rois = self._collect_projected_dxa_preview_rois(meta)
            placed_boxes = []
            for roi in rois:
                x = float(roi["x"]); y = float(roi["y"]); r = float(roi["r"])
                tag = str(roi["tag"])
                pen = QPen(QColor(0, 255, 0), 2)
                if tag in ("FN_L", "FN_R"):
                    pen = QPen(QColor(255, 255, 0), 2)
                painter.setPen(pen)
                painter.drawEllipse(QPointF(x, y), r, r)

                meas = self._measure_projected_dxa_preview_roi(np.asarray(quant_proj, np.float32), roi)
                label_lines = [tag]
                if meas is not None:
                    label_lines.append(f"aBMD {meas['bmd']:.1f}")
                    label_lines.append(f"T {meas['t_score']:.2f}  Z {meas['z_score']:.2f}")

                painter.setFont(QFont("Arial", 10))
                fm = painter.fontMetrics()
                box_w = max(fm.horizontalAdvance(s) for s in label_lines) + 10
                box_h = fm.height() * len(label_lines) + 8

                candidates = [
                    QRectF(x + r + 8, y - box_h / 2, box_w, box_h),
                    QRectF(x - r - box_w - 8, y - box_h / 2, box_w, box_h),
                    QRectF(x - box_w / 2, y - r - box_h - 8, box_w, box_h),
                    QRectF(x - box_w / 2, y + r + 8, box_w, box_h),
                ]
                chosen = None
                roi_rect = QRectF(x - r, y - r, 2 * r, 2 * r)
                bounds = QRectF(0, 0, w, h)
                for cand in candidates:
                    if not bounds.contains(cand):
                        continue
                    if cand.intersects(roi_rect):
                        continue
                    if any(cand.intersects(prev) for prev in placed_boxes):
                        continue
                    chosen = cand
                    break
                if chosen is None:
                    chosen = QRectF(min(max(x + r + 8, 0), max(0, w - box_w)),
                                    min(max(y - box_h / 2, 0), max(0, h - box_h)),
                                    box_w, box_h)
                placed_boxes.append(chosen)

                anchor = QPointF(chosen.left(), chosen.center().y()) if chosen.center().x() >= x else QPointF(chosen.right(), chosen.center().y())
                painter.drawLine(QPointF(x, y), anchor)
                painter.fillRect(chosen, QColor(0, 0, 0, 180))
                painter.setPen(QPen(QColor(255, 255, 255), 1))
                yy = chosen.top() + fm.ascent() + 4
                for s in label_lines:
                    painter.drawText(QPointF(chosen.left() + 5, yy), s)
                    yy += fm.height()
                painter.setPen(pen)
            painter.end()
            return pm
        except Exception:
            return None

    def refresh_dxa_tab_preview(self):
        if not hasattr(self, "lbl_dxa_preview"):
            return
        if self.ct_data is None:
            self.lbl_dxa_preview.setText("Load CT first.")
            self.lbl_dxa_preview.setPixmap(QPixmap())
            return
        proj, meta = self._build_dxa_projection_payload()
        if proj is None:
            self.lbl_dxa_preview.setText("Could not create DXA/AP projection.")
            self.lbl_dxa_preview.setPixmap(QPixmap())
            return
        display_proj = np.asarray((meta or {}).get("display_proj", proj), np.float32)
        quant_proj = np.asarray((meta or {}).get("quant_proj", proj), np.float32)
        rois = self._collect_projected_dxa_preview_rois(meta)
        if not rois:
            self.lbl_dxa_preview.setText("No projected L1-L4 / FN_L / FN_R ROIs found.")
            self.lbl_dxa_preview.setPixmap(QPixmap())
            return
        pm = self._render_dxa_preview_pixmap(display_proj, quant_proj, meta)
        if pm is None or pm.isNull():
            self.lbl_dxa_preview.setText("DXA/AP preview could not be rendered.")
            self.lbl_dxa_preview.setPixmap(QPixmap())
            return
        tgt = self.lbl_dxa_preview.size() if self.lbl_dxa_preview.size().width() > 10 else QSize(620, 760)
        scaled = pm.scaled(tgt, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.lbl_dxa_preview.setPixmap(scaled)
        self.lbl_dxa_preview.setText("")
        used_fb = bool((meta or {}).get("used_body_fallback", False))
        self.lbl_dxa_preview_info.setText(
            f"Labelled 2D areal DXA/AP preview from current CT ROIs. "
            f"Projected ROIs: {len(rois)}. "
            f"Display fallback: {'yes' if used_fb else 'no'}. "
            f"Use 'Open Editable Areal 2D DXA/AP' to move or resize ROIs."
        )

    def open_dxa_projection_dialog(self):
        if self.ct_data is None:
            QMessageBox.warning(self, "DXA Projection", "Load CT first.")
            return

        proj, meta = self._build_dxa_projection_payload()
        if proj is None:
            QMessageBox.warning(self, "DXA Projection", "Could not create projection image.")
            return

        if np.count_nonzero(np.asarray(proj) > 0) == 0:
            QMessageBox.warning(
                self,
                "DXA Projection",
                "Projection image is empty. Try lowering the AP bone threshold or review ROI placement.",
            )
            return

        dlg = DXAProjectionDialog(self, proj, meta)
        dlg.exec()
        self.refresh_dxa_tab_preview()

    # ----------------------------------------------------------------
    # Debug
    # ----------------------------------------------------------------
    def debug_roi_hu(self):
        c = self._active_canvas()
        if not c or not c.spherical_rois:
            QMessageBox.information(self, "Debug HU", "Draw/select an ROI first.")
            return
        roi = c.selected_roi or c.spherical_rois[-1]
        tag = getattr(roi, "tag", "ROI")
        res = self._analyze_roi(c, roi, site=tag)
        if not res:
            QMessageBox.warning(self, "Debug HU", "Failed to analyze ROI.")
            return
        lines = [
            "DEBUG ROI ANALYSIS",
            f"View: {c.view}, Slice: {c.current_slice}",
            f"ROI tag: {tag}",
            f"ROI Center (displayed): ({roi.center.x():.1f}, {roi.center.y():.1f})",
            f"ROI Radius: {roi.radius}",
            f"ROI slice_index: {getattr(roi, 'slice_index', 'None')}",
            f"Mean HU: {res['mean_hu']:.1f}",
            f"Std HU: {res['std_hu']:.1f}",
            f"BMD: {res['bmd']:.1f}",
            f"Voxels: {res['nvox']}",
            f"Mask Status: {res['mask_status']}",
            f"QC: {res['qc_status']}",
            f"QC Reasons: {';'.join(res.get('qc_reasons', []))}",
        ]
        self.results.append("\n" + "\n".join(lines))

    # ----------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------
    def closeEvent(self, event):
        try:
            self._clear_temp_nifti()
        except Exception:
            pass
        try:
            if self._dicom_extract_dir and os.path.isdir(self._dicom_extract_dir):
                shutil.rmtree(self._dicom_extract_dir, ignore_errors=True)
        except Exception:
            pass
        super().closeEvent(event)


# ================================================================
# DXA Projection Canvas / Dialog
# ================================================================

class ProjectionCanvas(QLabel):
    roi_changed = pyqtSignal()

    def __init__(self, arr2d: np.ndarray, parent=None):
        super().__init__(parent)
        self.arr = np.asarray(arr2d, np.float32)
        self.rois: List[SphericalROI] = []
        self.selected_roi: Optional[SphericalROI] = None
        self.dragging = False
        self.overlay_measurements: Dict[str, dict] = {}
        self.setMinimumSize(500, 350)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("border: 2px solid #333; background-color: black;")
        self.setMouseTracking(True)
        self.base_pixmap = QPixmap()
        self.update_image()

    def set_overlay_measurements(self, values: Optional[Dict[str, dict]]):
        self.overlay_measurements = dict(values or {})
        self.update()

    def update_image(self):
        a = np.nan_to_num(self.arr, nan=np.nanmin(self.arr) if self.arr.size else 0.0)
        lo = float(np.percentile(a, 1)) if a.size else 0.0
        hi = float(np.percentile(a, 99)) if a.size else 1.0
        if hi <= lo:
            hi = lo + 1.0
        norm = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
        img = (norm * 255.0 + 0.5).astype(np.uint8)
        h, w = img.shape
        qimg = QImage(img.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
        self.base_pixmap = QPixmap.fromImage(qimg)
        self.update()

    def _centered_rect(self, size):
        if size.height() == 0:
            return QRect()
        ar = size.width() / size.height()
        wr = self.width() / max(self.height(), 1)
        if wr > ar:
            h = self.height()
            w = h * ar
        else:
            w = self.width()
            h = w / ar
        r = QRect(0, 0, int(w), int(h))
        r.moveCenter(self.rect().center())
        return r

    def _widget_to_img(self, pos):
        rect = self._centered_rect(self.base_pixmap.size())
        if not rect.contains(int(pos.x()), int(pos.y())):
            return None
        x = (pos.x() - rect.left()) * self.arr.shape[1] / max(rect.width(), 1)
        y = (pos.y() - rect.top()) * self.arr.shape[0] / max(rect.height(), 1)
        return QPointF(float(x), float(y))

    def _img_to_widget(self, pt):
        rect = self._centered_rect(self.base_pixmap.size())
        x = rect.left() + pt.x() * rect.width() / max(self.arr.shape[1], 1)
        y = rect.top() + pt.y() * rect.height() / max(self.arr.shape[0], 1)
        return QPointF(float(x), float(y))

    def _roi_at_widget_pos(self, pos):
        if not self.rois:
            return None
        rect = self._centered_rect(self.base_pixmap.size())
        scale = rect.width() / max(self.arr.shape[1], 1)
        for roi in reversed(self.rois):
            c = self._img_to_widget(roi.center)
            r = max(4.0, roi.radius * scale)
            dx = pos.x() - c.x()
            dy = pos.y() - c.y()
            if (dx * dx + dy * dy) <= (r * r):
                return roi
        return None

    def _label_text_for_roi(self, roi: SphericalROI) -> str:
        return str(getattr(roi, "tag", "ROI"))

    def _best_label_point(self, p: QPainter, roi: SphericalROI, center: QPointF, radius: float, image_rect: QRect, used_rects: List[QRectF]) -> Tuple[QPointF, QRectF]:
        text = self._label_text_for_roi(roi)
        metrics = p.fontMetrics()
        text_w = metrics.horizontalAdvance(text) + 8
        text_h = metrics.height() + 4
        gap = 10

        def make_rect(pt: QPointF) -> QRectF:
            return QRectF(pt.x() - 2, pt.y() - metrics.ascent() - 2, text_w, text_h)

        margin = 4
        bounds = QRectF(
            image_rect.left() + margin,
            image_rect.top() + margin,
            max(1, image_rect.width() - 2 * margin),
            max(1, image_rect.height() - 2 * margin),
        )

        # Side-only label placement: prefer right or left of the ROI, never above or below.
        y_mid = center.y() + metrics.ascent() / 2.0
        candidates = [
            QPointF(center.x() + radius + gap, y_mid),
            QPointF(center.x() - radius - gap - text_w, y_mid),
            QPointF(center.x() + radius + gap, y_mid - text_h * 0.6),
            QPointF(center.x() + radius + gap, y_mid + text_h * 0.6),
            QPointF(center.x() - radius - gap - text_w, y_mid - text_h * 0.6),
            QPointF(center.x() - radius - gap - text_w, y_mid + text_h * 0.6),
        ]

        best = None
        best_rect = None
        best_score = None

        for i, pt0 in enumerate(candidates):
            pt = QPointF(pt0)
            rectf = make_rect(pt)

            if rectf.left() < bounds.left():
                dx = bounds.left() - rectf.left()
                pt.setX(pt.x() + dx)
                rectf.translate(dx, 0)
            if rectf.right() > bounds.right():
                dx = rectf.right() - bounds.right()
                pt.setX(pt.x() - dx)
                rectf.translate(-dx, 0)
            if rectf.top() < bounds.top():
                dy = bounds.top() - rectf.top()
                pt.setY(pt.y() + dy)
                rectf.translate(0, dy)
            if rectf.bottom() > bounds.bottom():
                dy = rectf.bottom() - bounds.bottom()
                pt.setY(pt.y() - dy)
                rectf.translate(0, -dy)

            overlaps = any(rectf.intersects(prev) for prev in used_rects)

            # Penalize any candidate whose label box vertically overlaps the ROI body too much.
            roi_band = QRectF(center.x() - radius, center.y() - radius, 2 * radius, 2 * radius)
            vertical_obscure = rectf.intersects(roi_band)

            dist = (rectf.center().x() - center.x()) ** 2 + (rectf.center().y() - center.y()) ** 2

            # Prefer pure side placements first (first two entries), then nearby side offsets.
            side_rank = 0 if i < 2 else 1
            score = (
                100000.0 * float(overlaps) +
                50000.0 * float(vertical_obscure) +
                2000.0 * side_rank +
                dist
            )

            if best_score is None or score < best_score:
                best_score = score
                best = pt
                best_rect = rectf

        fallback = QPointF(center.x() + radius + gap, y_mid)
        fallback_rect = make_rect(fallback)
        return best or fallback, best_rect or fallback_rect

    def paintEvent(self, e):
        if self.base_pixmap.isNull():
            return super().paintEvent(e)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self._centered_rect(self.base_pixmap.size())
        p.drawPixmap(rect, self.base_pixmap)
        scale = rect.width() / max(self.arr.shape[1], 1)

        p.setFont(QFont("Consolas", 10))
        used_label_rects: List[QRectF] = []

        for roi in self.rois:
            c = self._img_to_widget(roi.center)
            r = max(4.0, roi.radius * scale)
            pen = QPen(QColor(255, 255, 0), 2) if roi == self.selected_roi else QPen(QColor(0, 255, 0), 2)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(c, r, r)

            text = self._label_text_for_roi(roi)
            text_pt, text_rect = self._best_label_point(p, roi, c, r, rect, used_label_rects)
            used_label_rects.append(QRectF(text_rect))
            p.drawText(int(round(text_pt.x())), int(round(text_pt.y())), text)

        p.end()

    def mousePressEvent(self, e):
        hit = self._roi_at_widget_pos(e.position())
        if hit is None:
            return
        self.selected_roi = hit
        self.dragging = True
        self.update()
        self.roi_changed.emit()

    def mouseMoveEvent(self, e):
        if self.dragging and self.selected_roi is not None:
            img = self._widget_to_img(e.position())
            if img is not None:
                self.selected_roi.center = img
                self.update()
                self.roi_changed.emit()

    def mouseReleaseEvent(self, e):
        self.dragging = False

    def wheelEvent(self, e):
        if self.selected_roi is not None:
            self.selected_roi.resize(2 if e.angleDelta().y() > 0 else -2)
            self.update()
            self.roi_changed.emit()





    def _make_publication_dxa_projection_pixmap(self) -> Optional[QPixmap]:
        if self.ct_data is None:
            return None
        if not self._ensure_dxa_projection_results(force=False):
            return None

        dxa = self.dxa_proj_last or {}
        zmin = dxa.get("z_min")
        zmax = dxa.get("z_max")
        proj, meta = self._make_ap_projection(zmin, zmax)
        if proj is None:
            return None

        dlg = DXAProjectionDialog(self, proj, meta)
        try:
            dlg.resize(1200, 900)
            dlg.measure_all()
            dlg.canvas.resize(1100, 760)
            QApplication.processEvents()
            pm = QPixmap(dlg.canvas.size())
            pm.fill(QColor("#000000"))
            dlg.canvas.render(pm)
            return pm
        finally:
            dlg.deleteLater()



class PublicationROIFigureDialog(QDialog):
    def __init__(self, main, parent=None):
        super().__init__(parent or main)
        self.main = main
        self.setWindowTitle("Publication ROI Figures")
        self.resize(1580, 1020)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        title = QLabel("Publication-quality CT and DXA figures with ROI annotations")
        title.setStyleSheet("font-size: 13pt; font-weight: 600; color: #f5f5f5;")
        outer.addWidget(title)

        subtitle = QLabel(
            "Axial montage shows one panel per placed ROI slice. Sagittal montage shows one panel per ROI centerline. "
            "Areal DXA Projection shows the labeled AP/DXA image generated from the current ROIs."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #cfcfcf; font-size: 9.8pt;")
        outer.addWidget(subtitle)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setStyleSheet("""
            QTabBar::tab {
                min-width: 170px;
                min-height: 34px;
                padding: 5px 9px;
                font-size: 11pt;
            }
        """)

        self.lbl_axial = QLabel()
        self.lbl_axial.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_axial.setStyleSheet("background: #111; border: 1px solid #444;")

        self.lbl_sagittal = QLabel()
        self.lbl_sagittal.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_sagittal.setStyleSheet("background: #111; border: 1px solid #444;")

        self.lbl_dxa = QLabel()
        self.lbl_dxa.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_dxa.setStyleSheet("background: #111; border: 1px solid #444;")

        sa_ax = QScrollArea()
        sa_ax.setWidgetResizable(True)
        sa_ax.setWidget(self.lbl_axial)

        sa_sag = QScrollArea()
        sa_sag.setWidgetResizable(True)
        sa_sag.setWidget(self.lbl_sagittal)

        sa_dxa = QScrollArea()
        sa_dxa.setWidgetResizable(True)
        sa_dxa.setWidget(self.lbl_dxa)

        self.tabs.addTab(sa_ax, "Axial ROI Montage")
        self.tabs.addTab(sa_sag, "Sagittal ROI Montage")
        self.tabs.addTab(sa_dxa, "Areal DXA Projection")
        outer.addWidget(self.tabs, 1)

        self.axial_pixmap = self.main._make_publication_roi_montage("axial")
        self.sagittal_pixmap = self.main._make_publication_roi_montage("sagittal")
        self.dxa_pixmap = self.main._make_publication_dxa_projection_pixmap()

        if self.axial_pixmap is not None and not self.axial_pixmap.isNull():
            self.lbl_axial.setPixmap(self.axial_pixmap)
        if self.sagittal_pixmap is not None and not self.sagittal_pixmap.isNull():
            self.lbl_sagittal.setPixmap(self.sagittal_pixmap)
        if self.dxa_pixmap is not None and not self.dxa_pixmap.isNull():
            self.lbl_dxa.setPixmap(self.dxa_pixmap)
        else:
            self.lbl_dxa.setText("DXA/AP projection figure is not available. Ensure L1-L4 and femoral neck ROIs are present.")

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_export_current = QPushButton("Export Current PNG")
        btn_export_all = QPushButton("Export All PNGs")
        btn_close = QPushButton("Close")
        for b in (btn_export_current, btn_export_all, btn_close):
            b.setMinimumHeight(34)
        btn_export_current.clicked.connect(self.export_current_png)
        btn_export_all.clicked.connect(self.export_all_pngs)
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_export_current)
        btn_row.addWidget(btn_export_all)
        btn_row.addWidget(btn_close)
        outer.addLayout(btn_row)

    def _save_pixmap(self, pixmap: QPixmap, title: str):
        if pixmap is None or pixmap.isNull():
            QMessageBox.warning(self, "Export figure", "No figure is available to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            title,
            "",
            "PNG (*.png)"
        )
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        pixmap.save(path, "PNG")

    def export_current_png(self):
        idx = self.tabs.currentIndex()
        if idx == 0:
            self._save_pixmap(self.axial_pixmap, "Export Axial ROI Montage")
        elif idx == 1:
            self._save_pixmap(self.sagittal_pixmap, "Export Sagittal ROI Montage")
        else:
            self._save_pixmap(self.dxa_pixmap, "Export Areal DXA Projection")

    def export_all_pngs(self):
        directory = QFileDialog.getExistingDirectory(self, "Select export folder")
        if not directory:
            return
        saved = []
        if self.axial_pixmap is not None and not self.axial_pixmap.isNull():
            p1 = os.path.join(directory, "ct_roi_axial_montage.png")
            self.axial_pixmap.save(p1, "PNG")
            saved.append(p1)
        if self.sagittal_pixmap is not None and not self.sagittal_pixmap.isNull():
            p2 = os.path.join(directory, "ct_roi_sagittal_montage.png")
            self.sagittal_pixmap.save(p2, "PNG")
            saved.append(p2)
        if self.dxa_pixmap is not None and not self.dxa_pixmap.isNull():
            p3 = os.path.join(directory, "ct_dxa_ap_projection.png")
            self.dxa_pixmap.save(p3, "PNG")
            saved.append(p3)
        if saved:
            QMessageBox.information(self, "Export complete", "Saved:\n" + "\n".join(saved))
        else:
            QMessageBox.warning(self, "Export complete", "No figures were available to export.")


class DXAProjectionDialog(QDialog):
    SPINE_TAGS = ("L1", "L2", "L3", "L4")
    HIP_TAGS = ("FN_L", "FN_R")

    def __init__(self, main, proj2d: np.ndarray, meta: dict):
        super().__init__(main)
        self.main = main
        self.display_proj = np.asarray((meta or {}).get("display_proj", proj2d), np.float32)
        self.quant_proj = np.asarray((meta or {}).get("quant_proj", proj2d), np.float32)
        self.proj = self.quant_proj
        self.meta = meta or {}
        self.setWindowTitle("Editable Areal 2D DXA/AP Projection")
        self.resize(1120, 860)

        v = QVBoxLayout(self)
        hdr = QLabel("Editable Areal 2D DXA/AP Projection")
        hdr.setStyleSheet("font-size: 13pt; font-weight: 600; color: #f5f5f5;")
        v.addWidget(hdr)

        lbl = QLabel(
            "Existing L1-L4 and femoral neck ROIs are automatically projected onto the AP image. "
            "The labeled DXA/AP figure is editable here: left-drag to move a selected ROI, use the mouse wheel to resize, "
            "and the numeric labels update automatically."
        )
        lbl.setWordWrap(True)
        v.addWidget(lbl)

        if bool(self.meta.get("used_body_fallback", False)):
            note = QLabel("Display note: publication-style full-body AP display is shown; DXA measurements still use the bone-only quantification image.")
            note.setWordWrap(True)
            note.setStyleSheet("color: #ffd54f;")
            v.addWidget(note)

        self.canvas = ProjectionCanvas(self.display_proj, self)
        v.addWidget(self.canvas, 1)

        row = QHBoxLayout()
        self.btn_measure = QPushButton("Refresh DXA/AP ROI Labels")
        self.btn_measure.setMinimumHeight(34)
        self.btn_measure.clicked.connect(self.measure_all)
        row.addWidget(self.btn_measure)

        self.btn_export_png = QPushButton("Export Labeled DXA/AP PNG")
        self.btn_export_png.setMinimumHeight(34)
        self.btn_export_png.clicked.connect(self.export_png)
        row.addWidget(self.btn_export_png)

        self.btn_export_csv = QPushButton("Export DXA/AP CSV")
        self.btn_export_csv.setMinimumHeight(34)
        self.btn_export_csv.clicked.connect(self.export_csv)
        row.addWidget(self.btn_export_csv)
        v.addLayout(row)

        self.txt = QTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setMinimumHeight(220)
        v.addWidget(self.txt)

        self._roi_recalc_timer = QTimer(self)
        self._roi_recalc_timer.setSingleShot(True)
        self._roi_recalc_timer.timeout.connect(self.measure_all)
        self.canvas.roi_changed.connect(lambda: self._roi_recalc_timer.start(120))

        self._load_projected_rois()
        QTimer.singleShot(0, self.measure_all)

    def _collect_source_rois(self):
        source = []
        for tag in list(self.SPINE_TAGS) + list(self.HIP_TAGS):
            canvas, roi = self.main._find_roi_by_tag(tag)
            if roi is not None:
                source.append((tag, roi))
        return source

    def _load_projected_rois(self):
        self.canvas.rois.clear()
        if self.main.ct_data is None:
            self.canvas.update()
            return

        z_min = int(self.meta.get("z_min", 0))
        z_max = int(self.meta.get("z_max", self.main.ct_data.shape[2] - 1))
        sx, sy, sz = self.main.ct_data.shape
        px_x, px_y, px_z = self.main.ct_spacing if self.main.ct_spacing is not None else (1.0, 1.0, 1.0)

        for tag, roi in self._collect_source_rois():
            sl = getattr(roi, "slice_index", None)
            if sl is None:
                continue
            x = float(np.clip(roi.center.x(), 0, sx - 1))
            z = float(np.clip(sl, 0, sz - 1))
            proj_x = x
            proj_y = float(z_max - z)
            rx_mm = float(roi.radius) * float(px_x)
            rz_px = max(5, int(round(rx_mm / max(px_z, 1e-6))))
            proj_roi = SphericalROI(proj_x, proj_y, rz_px, tag=tag, slice_index=None)
            self.canvas.rois.append(proj_roi)

        if self.canvas.rois:
            self.canvas.selected_roi = self.canvas.rois[0]
        self.canvas.update()

    def _norms_for_site(self, tag: str):
        if tag in self.HIP_TAGS:
            ya_mu = float(getattr(self.main, "proj_fn_young_mean", 235.0))
            ya_sd = max(1e-6, float(getattr(self.main, "proj_fn_young_sd", 40.0)))
            z_mu = float(getattr(self.main, "proj_fn_age_mean", 205.0))
            z_sd = max(1e-6, float(getattr(self.main, "proj_fn_age_sd", 35.0)))
        else:
            ya_mu = float(getattr(self.main, "proj_spine_young_mean", 210.0))
            ya_sd = max(1e-6, float(getattr(self.main, "proj_spine_young_sd", 35.0)))
            z_mu = float(getattr(self.main, "proj_spine_age_mean", 180.0))
            z_sd = max(1e-6, float(getattr(self.main, "proj_spine_age_sd", 30.0)))
        return ya_mu, ya_sd, z_mu, z_sd

    def _measure_one_roi(self, roi: SphericalROI) -> Optional[dict]:
        rows, cols = np.ogrid[:self.proj.shape[0], :self.proj.shape[1]]
        mask = ((cols - roi.center.x()) ** 2 + (rows - roi.center.y()) ** 2) <= (roi.radius ** 2)
        vals = self.proj[mask]
        if vals.size < 10:
            return None

        bone_vals = vals[vals > 0] if self.main.ap_bone_only_zero_clip else vals
        if bone_vals.size < 5:
            bone_vals = vals

        mean_hu = float(np.mean(bone_vals))
        site_tag = str(getattr(roi, "tag", ""))
        a, b = self.main._proj_calibration_for_site(site_tag)
        pseudo_bmd = max(0.0, a * mean_hu + b)
        ya_mu, ya_sd, z_mu, z_sd = self._norms_for_site(site_tag)
        t = (pseudo_bmd - ya_mu) / ya_sd
        z = (pseudo_bmd - z_mu) / z_sd
        return {
            "site": site_tag or "ROI",
            "mean_hu": mean_hu,
            "bmd": pseudo_bmd,
            "t_score": t,
            "z_score": z,
            "nvox": int(bone_vals.size),
            "nvox_all": int(vals.size),
            "radius": int(roi.radius),
            "cx": float(roi.center.x()),
            "cy": float(roi.center.y()),
        }


    def _render_canvas_pixmap(self) -> Optional[QPixmap]:
        try:
            self.canvas.repaint()
            QApplication.processEvents()
            pm = QPixmap(self.canvas.size())
            pm.fill(QColor("#000000"))
            self.canvas.render(pm)
            return pm
        except Exception:
            return None

    def export_png(self):
        pm = self._render_canvas_pixmap()
        if pm is None or pm.isNull():
            QMessageBox.warning(self, "DXA/AP Export", "Could not render labeled DXA/AP image.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export labeled DXA/AP PNG",
            "ct_dxa_ap_projection_labeled.png",
            "PNG (*.png)",
        )
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        if not pm.save(path, "PNG"):
            QMessageBox.warning(self, "DXA/AP Export", "Failed to save PNG.")
            return
        self.main.results.append(f"DXA/AP labeled PNG exported: {path}")

    def export_csv(self):
        per_site = []
        for roi in self.canvas.rois:
            res = self._measure_one_roi(roi)
            if res is not None:
                per_site.append(res)
        if not per_site:
            QMessageBox.warning(self, "DXA/AP Export", "No DXA/AP measurements available.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export DXA/AP CSV",
            "ct_dxa_ap_projection_results.csv",
            "CSV (*.csv)",
        )
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Site", "Mean HU", "Pseudo-aBMD", "T-score", "Z-score", "Bone voxels", "All voxels", "Center X", "Center Y", "Radius"])
            for r in per_site:
                w.writerow([
                    r["site"], f'{r["mean_hu"]:.3f}', f'{r["bmd"]:.3f}', f'{r["t_score"]:.3f}',
                    f'{r["z_score"]:.3f}', r["nvox"], r["nvox_all"],
                    f'{r["cx"]:.2f}', f'{r["cy"]:.2f}', r["radius"]
                ])
        self.main.results.append(f"DXA/AP CSV exported: {path}")

    def measure_all(self):
        if not self.canvas.rois:
            QMessageBox.warning(self, "DXA Projection", "No projected ROIs found from L1-L4/FN-L/FN-R.")
            return

        per_site = []
        lines = ["DXA-STYLE AP PROJECTION FROM EXISTING ROIS", ""]
        for roi in self.canvas.rois:
            res = self._measure_one_roi(roi)
            if res is None:
                lines.append(f"{getattr(roi, 'tag', 'ROI')}: measurement failed")
                lines.append("")
                continue
            per_site.append(res)
            lines.extend([
                f"{res['site']}:",
                f"  Projected mean HU: {res['mean_hu']:.1f}",
                f"  Bone voxels: {res['nvox']} / {res['nvox_all']}",
                f"  Pseudo-aBMD: {res['bmd']:.1f}",
                f"  T-score: {res['t_score']:.2f}",
                f"  Z-score: {res['z_score']:.2f}",
                "",
            ])

        spine_sites = [r for r in per_site if r["site"] in self.SPINE_TAGS]
        hip_sites = [r for r in per_site if r["site"] in self.HIP_TAGS]

        composite = {}
        if spine_sites:
            w = np.array([max(1, int(r["nvox"])) for r in spine_sites], dtype=float)
            comp_mean_hu = float(np.average([r["mean_hu"] for r in spine_sites], weights=w))
            comp_bmd = float(np.average([r["bmd"] for r in spine_sites], weights=w))
            ya_mu = float(getattr(self.main, "proj_spine_young_mean", 210.0))
            ya_sd = max(1e-6, float(getattr(self.main, "proj_spine_young_sd", 35.0)))
            z_mu = float(getattr(self.main, "proj_spine_age_mean", 180.0))
            z_sd = max(1e-6, float(getattr(self.main, "proj_spine_age_sd", 30.0)))
            comp_t = (comp_bmd - ya_mu) / ya_sd
            comp_z = (comp_bmd - z_mu) / z_sd
            composite["spine"] = {
                "mean_hu": comp_mean_hu,
                "bmd": comp_bmd,
                "t_score": comp_t,
                "z_score": comp_z,
            }
            lines.extend([
                "L1-L4 COMPOSITE:",
                f"  Weighted mean HU: {comp_mean_hu:.1f}",
                f"  Weighted pseudo-aBMD: {comp_bmd:.1f}",
                f"  T-score: {comp_t:.2f}",
                f"  Z-score: {comp_z:.2f}",
                "",
            ])

        if hip_sites:
            w = np.array([max(1, int(r["nvox"])) for r in hip_sites], dtype=float)
            hip_mean_hu = float(np.average([r["mean_hu"] for r in hip_sites], weights=w))
            hip_mean_bmd = float(np.average([r["bmd"] for r in hip_sites], weights=w))
            ya_mu = float(getattr(self.main, "proj_fn_young_mean", 235.0))
            ya_sd = max(1e-6, float(getattr(self.main, "proj_fn_young_sd", 40.0)))
            z_mu = float(getattr(self.main, "proj_fn_age_mean", 205.0))
            z_sd = max(1e-6, float(getattr(self.main, "proj_fn_age_sd", 35.0)))
            hip_t = (hip_mean_bmd - ya_mu) / ya_sd
            hip_z = (hip_mean_bmd - z_mu) / z_sd
            composite["hips_mean"] = {
                "mean_hu": hip_mean_hu,
                "bmd": hip_mean_bmd,
                "t_score": hip_t,
                "z_score": hip_z,
            }
            lines.extend([
                "BILATERAL FEMORAL NECK SUMMARY:",
                f"  Weighted mean HU: {hip_mean_hu:.1f}",
                f"  Weighted pseudo-aBMD: {hip_mean_bmd:.1f}",
                f"  T-score: {hip_t:.2f}",
                f"  Z-score: {hip_z:.2f}",
            ])

        if bool(self.meta.get("used_body_fallback", False)):
            lines.extend(["", f"Display fallback used: yes  (bone coverage={100.0 * float(self.meta.get('bone_coverage', 0.0)):.2f}%)"])
        out = "\n".join(lines)
        self.txt.setPlainText(out)
        self.canvas.set_overlay_measurements({r["site"]: r for r in per_site})
        self.main.results.append("\n" + out)
        self.main.dxa_proj_last = {
            "sites": per_site,
            "composite": composite,
            "z_min": self.meta.get("z_min"),
            "z_max": self.meta.get("z_max"),
        }

# ================================================================
# Phantomless Calibration Dialog
# ================================================================
class PhantomlessDialog(QDialog):
    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self.setWindowTitle("Phantomless Calibration")
        self.setModal(True)

        self._orig = {
            "cal_blend_lambda": float(main.cal_blend_lambda),
            "cal_preset": str(main.cal_preset),
        }

        v = QVBoxLayout(self)
        lbl_info = QLabel("Blend preset calibration with ROI-based (phantomless) calibration.")
        lbl_info.setWordWrap(True)
        v.addWidget(lbl_info)

        row_p = QHBoxLayout()
        row_p.addWidget(QLabel("Preset:"))
        self.cb_preset = QComboBox()
        self.cb_preset.addItems(["contrast", "native"])
        self.cb_preset.setCurrentText(main.cal_preset)
        self.cb_preset.currentTextChanged.connect(self._on_preset)
        row_p.addWidget(self.cb_preset)
        v.addLayout(row_p)

        row_l = QHBoxLayout()
        self.lbl_lambda = QLabel(self._blend_text())
        row_l.addWidget(self.lbl_lambda, 1)
        self.sl_lambda = QSlider(Qt.Orientation.Horizontal)
        self.sl_lambda.setRange(0, 100)
        self.sl_lambda.setValue(int(round(main.cal_blend_lambda * 100)))
        self.sl_lambda.valueChanged.connect(self._on_lambda)
        row_l.addWidget(self.sl_lambda, 2)
        v.addLayout(row_l)

        self.lbl_eq_preset = QLabel(self._eq_preset_text())
        self.lbl_eq_roi = QLabel(self._eq_roi_text())
        self.lbl_eq_eff = QLabel(self.main._eq_eff_text())
        for w in (self.lbl_eq_preset, self.lbl_eq_roi, self.lbl_eq_eff):
            w.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            v.addWidget(w)

        btns = QHBoxLayout()
        b_ok = QPushButton("OK")
        b_cancel = QPushButton("Cancel")
        b_ok.clicked.connect(self.accept)
        b_cancel.clicked.connect(self.reject)
        btns.addStretch(1)
        btns.addWidget(b_ok)
        btns.addWidget(b_cancel)
        v.addLayout(btns)

    def _blend_text(self):
        return f"Blend λ = {self.main.cal_blend_lambda:.2f}"

    def _eq_preset_text(self):
        a_p, b_p = self.main.preset_params[
            "contrast" if self.main.cal_preset == "contrast" else "native"
        ]
        return f"Preset eq: {self.main._eq_text(a_p, b_p)}"

    def _eq_roi_text(self):
        return f"ROI eq: {self.main._eq_text(self.main.cal_roi_slope, self.main.cal_roi_intercept)}"

    def _on_lambda(self, val: int):
        lam = float(val) / 100.0
        self.main._on_blend(lam)
        self.lbl_lambda.setText(self._blend_text())
        self.lbl_eq_eff.setText(self.main._eq_eff_text())

    def _on_preset(self, name: str):
        self.main._set_preset(name)
        self.lbl_eq_preset.setText(self._eq_preset_text())
        self.lbl_eq_eff.setText(self.main._eq_eff_text())

    def reject(self):
        self.main.cal_preset = self._orig["cal_preset"]
        self.main.cal_blend_lambda = self._orig["cal_blend_lambda"]
        self.main._recompute_effective_calibration()
        super().reject()


# ================================================================
# Entry point
# ================================================================
def main():
    print("[QCT] main() enter", flush=True)
    logging.debug("main() enter")
    os.environ.setdefault("QT_DEBUG_PLUGINS", "0")
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(
        """
        QGroupBox { margin-top: 8px; padding: 8px; border: 1px solid #444; border-radius: 6px; }
        QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 8px; padding: 0 4px; }
        QPushButton, QToolButton { min-height: 27px; padding: 5px 9px; }
        QSlider::groove:horizontal { height: 6px; }
        QSlider::handle:horizontal { width: 14px; }
        QTabWidget::pane { border: 0; }
        """
    )
    w = BoneDensityAnalyzer()
    w.show()
    w.raise_()
    w.activateWindow()
    print("[QCT] window shown", flush=True)
    logging.debug("window shown")
    print("=" * 60 + "\nBone Density Analyzer – AutoSEG build\n" + "=" * 60, flush=True)
    logging.debug("Entering app.exec()")
    rc = app.exec()
    logging.debug("app.exec() returned rc=%s", rc)
    sys.exit(rc)




# ---- Patch v20: force all required target specs/paired ROIs after auto-placement ----
def _patched_ensure_all_required_target_specs(self):
    if self.ct_data is None:
        return []
    ensured = []
    base_r = 18
    try:
        base_r = int(getattr(getattr(self, "sp_cal_radius", None), "value", lambda: 18)())
    except Exception:
        base_r = 18

    for tag in self._required_target_labels():
        spec = self._get_target_spec(tag)
        if spec:
            try:
                self._sync_target_roi_pair_from_spec(tag, select_view="coronal")
            except Exception:
                pass
            ensured.append((tag, str(spec.get("status", "locked"))))
            continue

        c_exist, roi_exist = self._find_latest_roi_by_exact_tag_anywhere(tag)
        coords = None
        rad = base_r
        status = "fallback"
        if roi_exist is not None and c_exist is not None:
            try:
                coords = self._canvas_roi_to_volume(c_exist, roi_exist)
                rad = int(max(5, getattr(roi_exist, "radius", base_r)))
            except Exception:
                coords = None

        if coords is None:
            coords = self._estimate_target_volume_center(tag)
            status = "placeholder"

        if coords is None:
            sx, sy, sz = self.ct_data.shape
            coords = (sx // 2, sy // 2, sz // 2)
            status = "placeholder"

        x, y, z = [int(v) for v in coords]
        self._set_target_spec(tag, x, y, z, radius=rad, status=status)
        try:
            self._sync_target_roi_pair_from_spec(tag, select_view="coronal")
        except Exception:
            pass
        ensured.append((tag, status))

    self._update_roi_target_status()
    for c in self._canvases():
        try:
            c.update()
        except Exception:
            pass
    return ensured

_old_on_totalseg_done = BoneDensityAnalyzer._on_totalseg_done

def _patched_on_totalseg_done(self, files: dict, out_dir: str):
    _old_on_totalseg_done(self, files, out_dir)
    try:
        ensured = self._ensure_all_required_target_specs()
        names = [f"{tag}:{status}" for tag, status in ensured]
        try:
            self.results.append("[Auto-Seg] Required targets ensured: " + ", ".join(names))
        except Exception:
            pass
    except Exception as e:
        try:
            self.results.append(f"[Auto-Seg] Ensure required targets failed: {e}")
        except Exception:
            pass

BoneDensityAnalyzer._ensure_all_required_target_specs = _patched_ensure_all_required_target_specs
BoneDensityAnalyzer._on_totalseg_done = _patched_on_totalseg_done

if __name__ == "__main__":
    print("[QCT] __main__ starting", flush=True)
    try:
        main()
    except Exception as e:
        import traceback
        print("[QCT] FATAL:", e, flush=True)
        traceback.print_exc()
        logging.exception("Fatal in main: %s", e)