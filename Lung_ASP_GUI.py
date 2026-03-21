"""
Lung ASP GUI — Lung-tumour segmentation and metrics application.

Requires:
    Python 3.8+, tkinter (stdlib), Pillow, numpy, SimpleITK (optional)
"""

from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional

# ---------------------------------------------------------------------------
# Optional heavy dependencies — gracefully absent so the GUI still launches
# ---------------------------------------------------------------------------
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False

try:
    import SimpleITK as sitk
    _HAS_SITK = True
except ImportError:  # pragma: no cover
    _HAS_SITK = False

try:
    from PIL import Image, ImageTk
    _HAS_PIL = True
except ImportError:  # pragma: no cover
    _HAS_PIL = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
APP_TITLE = "Lung ASP — Segmentation & Metrics"
PAD = 6
BG_DARK = "#1e1e2e"
BG_MID  = "#2a2a3e"
BG_LIGHT = "#3a3a5e"
FG_TEXT = "#cdd6f4"
ACCENT  = "#89b4fa"
BTN_BG  = "#45475a"

# ---------------------------------------------------------------------------
# Segmentation / metrics helpers
# ---------------------------------------------------------------------------

def _hu_threshold_mask(volume: "np.ndarray", lo: float = -1000,
                        hi: float = -200) -> "np.ndarray":
    """Return a binary mask of voxels whose HU value lies within [lo, hi]."""
    return (volume >= lo) & (volume <= hi)


def _erode_dilate(mask: "np.ndarray", iterations: int = 1,
                  mode: str = "dilate") -> "np.ndarray":
    """Simple binary morphology using scipy if available, else identity."""
    try:
        from scipy.ndimage import binary_dilation, binary_erosion
        fn = binary_dilation if mode == "dilate" else binary_erosion
        return fn(mask, iterations=iterations)
    except ImportError:
        return mask


def _largest_connected_component(mask: "np.ndarray") -> "np.ndarray":
    """Return mask containing only the largest connected component."""
    try:
        from scipy.ndimage import label
        labelled, n = label(mask)
        if n == 0:
            return mask
        sizes = [(labelled == i).sum() for i in range(1, n + 1)]
        largest = sizes.index(max(sizes)) + 1
        return labelled == largest
    except ImportError:
        return mask


def compute_metrics(mask: "np.ndarray",
                    spacing: tuple = (1.0, 1.0, 1.0)) -> dict:
    """Compute basic volumetric metrics from a binary segmentation mask."""
    voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
    n_voxels = int(mask.sum()) if _HAS_NUMPY else 0
    volume_mm3 = n_voxels * voxel_vol_mm3
    volume_cc   = volume_mm3 / 1000.0
    return {
        "voxels":    n_voxels,
        "volume_mm3": volume_mm3,
        "volume_cc":  volume_cc,
    }


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class LungASPApp(tk.Tk):
    """Main GUI window for the Lung ASP application."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self) -> None:
        super().__init__()

        self.title(APP_TITLE)
        self.resizable(True, True)
        self.configure(bg=BG_DARK)

        # --- File paths ---
        self._input_path:  tk.StringVar = tk.StringVar()
        self._output_dir:  tk.StringVar = tk.StringVar()

        # --- Segmentation parameters ---
        self._hu_low:       tk.DoubleVar = tk.DoubleVar(value=-950.0)
        self._hu_high:      tk.DoubleVar = tk.DoubleVar(value=-400.0)

        self._lung_dilation:  tk.IntVar  = tk.IntVar(value=3)
        self._excl_dilation:  tk.IntVar  = tk.IntVar(value=2)
        self._min_lesion_sz:  tk.IntVar  = tk.IntVar(value=10)
        self._max_lesion_sz:  tk.IntVar  = tk.IntVar(value=10000)

        self._keep_largest:   tk.BooleanVar = tk.BooleanVar(value=True)
        self._fill_holes:     tk.BooleanVar = tk.BooleanVar(value=True)
        self._save_mask:      tk.BooleanVar = tk.BooleanVar(value=True)
        self._save_metrics:   tk.BooleanVar = tk.BooleanVar(value=True)

        # --- Display state ---
        self._current_slice:  tk.IntVar  = tk.IntVar(value=0)
        self._volume:   Optional[object] = None   # numpy array when loaded
        self._mask:     Optional[object] = None
        self._photo:    Optional[object] = None   # ImageTk.PhotoImage

        self._build_ui()
        self._center_window(960, 640)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        """Build all widgets."""
        # ---- Top bar: file selection ----------------------------------
        top = tk.Frame(self, bg=BG_DARK, pady=PAD)
        top.pack(side=tk.TOP, fill=tk.X, padx=PAD)

        tk.Label(top, text="Input file:", bg=BG_DARK, fg=FG_TEXT).pack(
            side=tk.LEFT)
        tk.Entry(top, textvariable=self._input_path, width=45,
                 bg=BG_MID, fg=FG_TEXT, insertbackground=FG_TEXT).pack(
            side=tk.LEFT, padx=(4, 2))
        tk.Button(top, text="Browse…", command=self._browse_input,
                  bg=BTN_BG, fg=FG_TEXT, relief=tk.FLAT).pack(
            side=tk.LEFT, padx=(0, 12))

        tk.Label(top, text="Output dir:", bg=BG_DARK, fg=FG_TEXT).pack(
            side=tk.LEFT)
        tk.Entry(top, textvariable=self._output_dir, width=30,
                 bg=BG_MID, fg=FG_TEXT, insertbackground=FG_TEXT).pack(
            side=tk.LEFT, padx=(4, 2))
        tk.Button(top, text="Browse…", command=self._browse_output,
                  bg=BTN_BG, fg=FG_TEXT, relief=tk.FLAT).pack(
            side=tk.LEFT)

        # ---- Main area: left options + right canvas -------------------
        main = tk.Frame(self, bg=BG_DARK)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=PAD, pady=PAD)

        # ---- Left panel: options --------------------------------------
        opt_frame = tk.LabelFrame(main, text=" Options ", bg=BG_DARK,
                                  fg=ACCENT, bd=1, relief=tk.GROOVE)
        opt_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, PAD))

        opt_inner = tk.Frame(opt_frame, bg=BG_DARK)
        opt_inner.pack(padx=PAD, pady=PAD)

        # HU range
        tk.Label(opt_inner, text="─── HU Range ───",
                 bg=BG_DARK, fg=ACCENT).grid(
            row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 4))

        self._make_spinbox(opt_inner, "HU low:", self._hu_low,
                           lo=-1500, hi=0, inc=10, fmt="%.0f", row=1)
        self._make_spinbox(opt_inner, "HU high:", self._hu_high,
                           lo=-1500, hi=0, inc=10, fmt="%.0f", row=2)

        # Morphology
        tk.Label(opt_inner, text="─── Morphology ───",
                 bg=BG_DARK, fg=ACCENT).grid(
            row=3, column=0, columnspan=2, sticky=tk.W, pady=(8, 4))

        self._make_spinbox(opt_inner, "Excl. dilation (mm):", self._excl_dilation,
                           lo=0, hi=20, inc=1, fmt="%.0f", row=4)
        self._make_spinbox(opt_inner, "Lung dilation (mm):", self._lung_dilation,
                           lo=0, hi=20, inc=1, fmt="%.0f", row=5)

        # Lesion size
        tk.Label(opt_inner, text="─── Lesion size (vox) ───",
                 bg=BG_DARK, fg=ACCENT).grid(
            row=6, column=0, columnspan=2, sticky=tk.W, pady=(8, 4))

        self._make_spinbox(opt_inner, "Min lesion:", self._min_lesion_sz,
                           lo=1, hi=100000, inc=1, fmt="%.0f", row=7)
        self._make_spinbox(opt_inner, "Max lesion:", self._max_lesion_sz,
                           lo=1, hi=10000000, inc=100, fmt="%.0f", row=8)

        # Toggles
        tk.Label(opt_inner, text="─── Flags ───",
                 bg=BG_DARK, fg=ACCENT).grid(
            row=9, column=0, columnspan=2, sticky=tk.W, pady=(8, 4))

        for r, (text, var) in enumerate([
            ("Keep largest component", self._keep_largest),
            ("Fill holes",             self._fill_holes),
            ("Save mask",              self._save_mask),
            ("Save metrics CSV",       self._save_metrics),
        ], start=10):
            tk.Checkbutton(opt_inner, text=text, variable=var,
                           bg=BG_DARK, fg=FG_TEXT,
                           selectcolor=BG_LIGHT,
                           activebackground=BG_DARK).grid(
                row=r, column=0, columnspan=2, sticky=tk.W)

        # ---- Right panel: image canvas + slice slider ----------------
        right = tk.Frame(main, bg=BG_DARK)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._canvas = tk.Canvas(right, bg="#000000",
                                 highlightthickness=0)
        self._canvas.pack(fill=tk.BOTH, expand=True)

        slice_row = tk.Frame(right, bg=BG_DARK)
        slice_row.pack(fill=tk.X)
        tk.Label(slice_row, text="Slice:", bg=BG_DARK, fg=FG_TEXT).pack(
            side=tk.LEFT)
        self._slice_slider = tk.Scale(
            slice_row, variable=self._current_slice,
            orient=tk.HORIZONTAL, from_=0, to=0,
            bg=BG_DARK, fg=FG_TEXT, troughcolor=BG_LIGHT,
            highlightthickness=0,
            command=lambda _: self._redraw_slice())
        self._slice_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # ---- Bottom bar: run + status --------------------------------
        bot = tk.Frame(self, bg=BG_DARK, pady=PAD)
        bot.pack(side=tk.BOTTOM, fill=tk.X, padx=PAD)

        self._run_btn = tk.Button(bot, text="▶  Run Segmentation",
                                  command=self._run_segmentation,
                                  bg=ACCENT, fg=BG_DARK,
                                  font=("TkDefaultFont", 10, "bold"),
                                  relief=tk.FLAT, padx=12, pady=4)
        self._run_btn.pack(side=tk.LEFT, padx=(0, 12))

        self._status_var = tk.StringVar(value="Ready.")
        tk.Label(bot, textvariable=self._status_var,
                 bg=BG_DARK, fg=FG_TEXT, anchor=tk.W).pack(
            side=tk.LEFT, fill=tk.X, expand=True)

        self._progress = ttk.Progressbar(bot, mode="indeterminate",
                                         length=160)
        self._progress.pack(side=tk.RIGHT, padx=(8, 0))

    # ------------------------------------------------------------------
    # Widget helpers
    # ------------------------------------------------------------------
    def _make_spinbox(self, parent: tk.Widget, label: str,
                      var: tk.Variable, *, lo: float, hi: float,
                      inc: float, fmt: str = "%.0f",
                      row: int = 0) -> tk.Spinbox:
        """Create a labelled Spinbox and grid it into *parent*.

        Parameters
        ----------
        parent : tk.Widget
            Container widget.
        label  : str
            Text for the accompanying label.
        var    : tk.Variable
            Tkinter variable linked to the spinbox.
        lo, hi : float
            Minimum and maximum values.
        inc    : float
            Increment step.
        fmt    : str
            Tcl/Tk format string.  Must be a ``%f``-style specifier such as
            ``"%.0f"`` or ``"%.2f"``.  The integer ``"%d"`` specifier is
            **not** accepted by Tcl/Tk's Spinbox widget.
        row    : int
            Grid row index.
        """
        tk.Label(parent, text=label, bg=BG_DARK, fg=FG_TEXT,
                 anchor=tk.W).grid(row=row, column=0, sticky=tk.W,
                                   padx=(0, 8), pady=2)
        sb = tk.Spinbox(parent, textvariable=var, from_=lo, to=hi, increment=inc,
                        format=fmt,
                        width=8, bg=BG_MID, fg=FG_TEXT,
                        buttonbackground=BTN_BG,
                        insertbackground=FG_TEXT,
                        relief=tk.FLAT)
        sb.grid(row=row, column=1, sticky=tk.EW, pady=2)
        return sb

    # ------------------------------------------------------------------
    # File dialogs
    # ------------------------------------------------------------------
    def _browse_input(self) -> None:
        path = filedialog.askopenfilename(
            title="Select input volume",
            filetypes=[
                ("Medical images", "*.nii *.nii.gz *.mha *.mhd *.nrrd *.dcm"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._input_path.set(path)

    def _browse_output(self) -> None:
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            self._output_dir.set(path)

    # ------------------------------------------------------------------
    # Segmentation pipeline
    # ------------------------------------------------------------------
    def _run_segmentation(self) -> None:
        """Validate inputs and launch segmentation in a background thread."""
        input_path = self._input_path.get().strip()
        if not input_path:
            messagebox.showwarning("No input", "Please select an input file.")
            return
        if not os.path.isfile(input_path):
            messagebox.showerror("File not found",
                                 f"Cannot find:\n{input_path}")
            return

        self._run_btn.configure(state=tk.DISABLED)
        self._progress.start(10)
        self._status_var.set("Loading…")

        thread = threading.Thread(target=self._segmentation_worker,
                                  args=(input_path,), daemon=True)
        thread.start()

    def _segmentation_worker(self, input_path: str) -> None:
        """Background worker: load → segment → compute metrics → save."""
        try:
            volume, spacing = self._load_volume(input_path)
            self._set_status("Segmenting…")
            mask = self._segment(volume)
            self._set_status("Computing metrics…")
            metrics = compute_metrics(mask, spacing)
            self._set_status("Saving results…")
            self._save_results(mask, metrics, input_path)

            # Hand results back to the main thread
            self.after(0, self._on_segmentation_done, volume, mask, metrics)
        except Exception as exc:  # noqa: BLE001
            self.after(0, self._on_segmentation_error, str(exc))

    def _load_volume(self, path: str):
        """Load a medical image volume; returns (ndarray, spacing)."""
        if not _HAS_SITK:
            raise RuntimeError(
                "SimpleITK is required to load medical images.\n"
                "Install it with:  pip install SimpleITK"
            )
        img = sitk.ReadImage(path)
        volume = sitk.GetArrayFromImage(img)   # Z, Y, X
        spacing = img.GetSpacing()              # X, Y, Z -> reorder
        spacing = (spacing[2], spacing[1], spacing[0])
        return volume, spacing

    def _segment(self, volume) -> "np.ndarray":
        """Apply the configured segmentation pipeline."""
        hu_lo  = self._hu_low.get()
        hu_hi  = self._hu_high.get()
        e_dil  = int(self._excl_dilation.get())
        l_dil  = int(self._lung_dilation.get())
        sz_min = int(self._min_lesion_sz.get())
        sz_max = int(self._max_lesion_sz.get())

        mask = _hu_threshold_mask(volume, hu_lo, hu_hi)

        if l_dil > 0:
            mask = _erode_dilate(mask, iterations=l_dil, mode="dilate")
        if e_dil > 0:
            mask = _erode_dilate(mask, iterations=e_dil, mode="erode")

        if self._keep_largest.get():
            mask = _largest_connected_component(mask)

        # Size filter
        if _HAS_NUMPY:
            try:
                from scipy.ndimage import label as sp_label
                labelled, n = sp_label(mask)
                filtered = np.zeros_like(mask)
                for i in range(1, n + 1):
                    comp = labelled == i
                    sz = int(comp.sum())
                    if sz_min <= sz <= sz_max:
                        filtered |= comp
                mask = filtered
            except ImportError:
                pass

        if self._fill_holes.get():
            try:
                from scipy.ndimage import binary_fill_holes
                mask = binary_fill_holes(mask)
            except ImportError:
                pass

        return mask

    def _save_results(self, mask, metrics: dict, input_path: str) -> None:
        """Save mask and/or metrics CSV to the output directory."""
        out_dir = self._output_dir.get().strip() or os.path.dirname(input_path)
        base = os.path.splitext(os.path.basename(input_path))[0]
        base = base.replace(".nii", "")  # handle .nii.gz

        if self._save_mask.get() and _HAS_SITK:
            import SimpleITK as sitk  # noqa: PLC0415
            mask_img = sitk.GetImageFromArray(mask.astype("uint8"))
            sitk.WriteImage(mask_img,
                            os.path.join(out_dir, f"{base}_mask.nii.gz"))

        if self._save_metrics.get():
            csv_path = os.path.join(out_dir, f"{base}_metrics.csv")
            with open(csv_path, "w") as fh:
                fh.write("metric,value\n")
                for k, v in metrics.items():
                    fh.write(f"{k},{v}\n")

    # ------------------------------------------------------------------
    # Callbacks on main thread
    # ------------------------------------------------------------------
    def _on_segmentation_done(self, volume, mask, metrics: dict) -> None:
        self._volume = volume
        self._mask   = mask
        self._progress.stop()
        self._run_btn.configure(state=tk.NORMAL)

        if _HAS_NUMPY:
            n_slices = int(volume.shape[0]) - 1
        else:
            n_slices = 0
        self._slice_slider.configure(to=n_slices)
        self._current_slice.set(n_slices // 2)
        self._redraw_slice()

        vol_cc = metrics.get("volume_cc", 0.0)
        self._status_var.set(
            f"Done — volume {vol_cc:.1f} cc  "
            f"({metrics.get('voxels', 0):,} voxels)"
        )

    def _on_segmentation_error(self, msg: str) -> None:
        self._progress.stop()
        self._run_btn.configure(state=tk.NORMAL)
        self._status_var.set(f"Error: {msg}")
        messagebox.showerror("Segmentation error", msg)

    def _set_status(self, msg: str) -> None:
        self.after(0, self._status_var.set, msg)

    # ------------------------------------------------------------------
    # Slice display
    # ------------------------------------------------------------------
    def _redraw_slice(self) -> None:
        """Render the current slice (with optional mask overlay) on canvas."""
        if self._volume is None or not _HAS_NUMPY or not _HAS_PIL:
            return
        z = int(self._current_slice.get())
        z = max(0, min(z, self._volume.shape[0] - 1))

        sl = self._volume[z].astype(float)
        # Window/level: lung window
        vmin, vmax = -1500.0, 500.0
        sl = (sl - vmin) / (vmax - vmin)
        sl = np.clip(sl, 0.0, 1.0)
        sl_u8 = (sl * 255).astype("uint8")

        img = Image.fromarray(sl_u8, mode="L").convert("RGB")

        if self._mask is not None:
            overlay = np.zeros((*sl_u8.shape, 3), dtype="uint8")
            overlay[self._mask[z] > 0] = (255, 80, 80)
            overlay_img = Image.fromarray(overlay, mode="RGB")
            img = Image.blend(img, overlay_img, alpha=0.35)

        # Fit to canvas
        cw = self._canvas.winfo_width() or 512
        ch = self._canvas.winfo_height() or 512
        img = img.resize((cw, ch), Image.LANCZOS)

        self._photo = ImageTk.PhotoImage(img)
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def _center_window(self, w: int, h: int) -> None:
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x  = (sw - w) // 2
        y  = (sh - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = LungASPApp()
    app.mainloop()


if __name__ == "__main__":
    main()
