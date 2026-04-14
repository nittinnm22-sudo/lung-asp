"""
bone_density_qct_merged_cleaned_workflow1_v48.py
Bone Density QCT Analysis Workflow v48 — GUI Version

Tkinter-based GUI front-end for the BoneDensityQCT analysis engine defined in
bone_density_qct_merged_cleaned_workflow_v48.py.

Features
--------
- Load DICOM directory or NIfTI file via file-dialog
- Add / remove spherical ROIs interactively
- Run full analysis (axial QCT + pseudo-DXA projection)
- Display results in a text panel
- Export results as JSON
- Embedded matplotlib figure showing CT mid-slice with ROI overlays
"""

import os
import sys
import json
import logging
import threading
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Import core library.  Fall back to duplicating constants if not importable.
# ---------------------------------------------------------------------------
try:
    from bone_density_qct_merged_cleaned_workflow_v48 import (
        BoneDensityQCT,
        ROI,
        SPINE_TAGS,
        FN_TAGS,
        QCT_SPINE_SLOPE,
        QCT_SPINE_INTERCEPT,
        QCT_FN_SLOPE,
        QCT_FN_INTERCEPT,
        QCT_SPINE_NORM_MEAN,
        QCT_SPINE_NORM_SD,
        QCT_FN_NORM_MEAN,
        QCT_FN_NORM_SD,
        DXA_SPINE_NORM_MEAN,
        DXA_SPINE_NORM_SD,
        DXA_FN_NORM_MEAN,
        DXA_FN_NORM_SD,
        PROJ_SPINE_SLOPE,
        PROJ_SPINE_INTERCEPT,
        PROJ_FN_SLOPE,
        PROJ_FN_INTERCEPT,
        BONE_THRESHOLD,
        _build_synthetic_demo,
        _json_default,
    )
    _CORE_IMPORTED = True
except ImportError:
    _CORE_IMPORTED = False
    # Duplicate constants so the GUI can still start if the file is run alone
    SPINE_TAGS = {"L1", "L2", "L3", "L4", "L5"}
    FN_TAGS = {"FN_R", "FN_L", "FEMORAL_NECK_R", "FEMORAL_NECK_L"}
    BONE_THRESHOLD = 150.0

# ---------------------------------------------------------------------------
# Tkinter — optional; GUI is disabled gracefully if unavailable
# ---------------------------------------------------------------------------
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext
    _TK_AVAILABLE = True
except ImportError:
    _TK_AVAILABLE = False

# ---------------------------------------------------------------------------
# Matplotlib — optional; plot panel is disabled gracefully if unavailable
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    import matplotlib.patches as mpatches
    _MPL_AVAILABLE = True
except Exception:
    _MPL_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# ROI entry dialog
# ─────────────────────────────────────────────────────────────────────────────

class _ROIDialog:
    """Modal dialog for adding a new ROI."""

    def __init__(self, parent):
        self.result = None
        self._top = tk.Toplevel(parent)
        self._top.title("Add ROI")
        self._top.resizable(False, False)
        self._top.grab_set()

        pad = {"padx": 6, "pady": 3}

        ttk.Label(self._top, text="Tag (e.g. L1, FN_R):").grid(
            row=0, column=0, sticky="e", **pad
        )
        self._tag = ttk.Entry(self._top, width=14)
        self._tag.grid(row=0, column=1, **pad)
        self._tag.insert(0, "L1")

        for i, (lbl, default) in enumerate(
            [("Center X (vox):", "20"), ("Center Y (vox):", "30"), ("Center Z (vox):", "10")],
            start=1,
        ):
            ttk.Label(self._top, text=lbl).grid(row=i, column=0, sticky="e", **pad)
            entry = ttk.Entry(self._top, width=10)
            entry.insert(0, default)
            entry.grid(row=i, column=1, **pad)
            setattr(self, f"_c{['x','y','z'][i-1]}", entry)

        ttk.Label(self._top, text="Radius (mm):").grid(
            row=4, column=0, sticky="e", **pad
        )
        self._radius = ttk.Entry(self._top, width=10)
        self._radius.insert(0, "15.0")
        self._radius.grid(row=4, column=1, **pad)

        btn_frame = ttk.Frame(self._top)
        btn_frame.grid(row=5, column=0, columnspan=2, pady=8)
        ttk.Button(btn_frame, text="Add", command=self._ok).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Cancel", command=self._top.destroy).pack(
            side="left", padx=4
        )

        parent.wait_window(self._top)

    def _ok(self):
        try:
            tag = self._tag.get().strip().upper()
            cx = float(self._cx.get())
            cy = float(self._cy.get())
            cz = float(self._cz.get())
            radius = float(self._radius.get())
            if not tag:
                raise ValueError("Tag cannot be empty")
            self.result = {"tag": tag, "center": (cx, cy, cz), "radius_mm": radius}
            self._top.destroy()
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc), parent=self._top)


# ─────────────────────────────────────────────────────────────────────────────
# Main application window
# ─────────────────────────────────────────────────────────────────────────────

class BoneDensityApp:
    """Tkinter GUI for the Bone Density QCT analysis engine."""

    # Window title
    TITLE = "Bone Density QCT v48"

    def __init__(self, root: "tk.Tk"):
        self.root = root
        self.root.title(self.TITLE)
        self.root.geometry("1100x760")

        self._qct: BoneDensityQCT = BoneDensityQCT() if _CORE_IMPORTED else None
        self._current_slice = 0
        self._canvas_fig = None  # matplotlib FigureCanvasTkAgg

        self._build_menu()
        self._build_layout()
        self._refresh_roi_list()
        self._update_status("Ready. Load a DICOM directory or NIfTI file to begin.")

    # ── Menu ──────────────────────────────────────────────────────────

    def _build_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load DICOM Directory…", command=self._load_dicom)
        file_menu.add_command(label="Load NIfTI File…", command=self._load_nifti)
        file_menu.add_separator()
        file_menu.add_command(label="Load ROIs from JSON…", command=self._load_roi_json)
        file_menu.add_command(label="Export Results as JSON…", command=self._export_json)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Run Analysis", command=self._run_analysis)
        tools_menu.add_command(label="Load Demo Volume", command=self._load_demo)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)

    # ── Layout ────────────────────────────────────────────────────────

    def _build_layout(self):
        # Top toolbar
        toolbar = ttk.Frame(self.root, relief="ridge", padding=4)
        toolbar.pack(side="top", fill="x")

        ttk.Button(toolbar, text="📂 DICOM", command=self._load_dicom).pack(
            side="left", padx=2
        )
        ttk.Button(toolbar, text="📂 NIfTI", command=self._load_nifti).pack(
            side="left", padx=2
        )
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=4)
        ttk.Button(toolbar, text="▶ Run Analysis", command=self._run_analysis).pack(
            side="left", padx=2
        )
        ttk.Button(toolbar, text="🔬 Demo", command=self._load_demo).pack(
            side="left", padx=2
        )
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=4)
        ttk.Button(toolbar, text="💾 Export JSON", command=self._export_json).pack(
            side="left", padx=2
        )

        # Status bar (bottom)
        self._status_var = tk.StringVar(value="")
        status_bar = ttk.Label(
            self.root, textvariable=self._status_var, relief="sunken", anchor="w"
        )
        status_bar.pack(side="bottom", fill="x")

        # Main paned window
        paned = ttk.PanedWindow(self.root, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=4, pady=4)

        # Left panel: ROI list + controls
        left_frame = ttk.LabelFrame(paned, text="ROIs", width=220)
        paned.add(left_frame, weight=0)
        self._build_roi_panel(left_frame)

        # Centre panel: CT image viewer
        centre_frame = ttk.LabelFrame(paned, text="CT Viewer")
        paned.add(centre_frame, weight=2)
        self._build_image_panel(centre_frame)

        # Right panel: Results
        right_frame = ttk.LabelFrame(paned, text="Results")
        paned.add(right_frame, weight=1)
        self._build_results_panel(right_frame)

    def _build_roi_panel(self, parent):
        self._roi_listvar = tk.StringVar()
        listbox_frame = ttk.Frame(parent)
        listbox_frame.pack(fill="both", expand=True, padx=4, pady=4)

        self._roi_listbox = tk.Listbox(
            listbox_frame, listvariable=self._roi_listvar, height=14, selectmode="single"
        )
        scrollbar = ttk.Scrollbar(
            listbox_frame, orient="vertical", command=self._roi_listbox.yview
        )
        self._roi_listbox.configure(yscrollcommand=scrollbar.set)
        self._roi_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill="x", padx=4, pady=2)
        ttk.Button(btn_frame, text="+ Add", command=self._add_roi).pack(
            side="left", padx=2
        )
        ttk.Button(btn_frame, text="− Remove", command=self._remove_roi).pack(
            side="left", padx=2
        )
        ttk.Button(btn_frame, text="⟳ Clear All", command=self._clear_rois).pack(
            side="left", padx=2
        )

        # Slice slider
        slider_frame = ttk.LabelFrame(parent, text="CT Slice")
        slider_frame.pack(fill="x", padx=4, pady=4)
        self._slice_var = tk.IntVar(value=0)
        self._slice_slider = ttk.Scale(
            slider_frame,
            from_=0,
            to=0,
            variable=self._slice_var,
            orient="horizontal",
            command=self._on_slice_change,
        )
        self._slice_slider.pack(fill="x", padx=4, pady=2)
        self._slice_label = ttk.Label(slider_frame, text="Slice: 0 / 0")
        self._slice_label.pack()

    def _build_image_panel(self, parent):
        if _MPL_AVAILABLE:
            self._fig = Figure(figsize=(5, 5), dpi=96)
            self._ax = self._fig.add_subplot(111)
            self._ax.set_title("CT slice")
            self._ax.axis("off")

            self._canvas_fig = FigureCanvasTkAgg(self._fig, master=parent)
            self._canvas_fig.get_tk_widget().pack(fill="both", expand=True)
            NavigationToolbar2Tk(self._canvas_fig, parent)
        else:
            ttk.Label(
                parent,
                text="matplotlib (TkAgg) not available.\nInstall matplotlib to see CT images.",
                anchor="center",
            ).pack(expand=True)

    def _build_results_panel(self, parent):
        self._results_text = scrolledtext.ScrolledText(
            parent, wrap="word", state="disabled", font=("Courier", 9)
        )
        self._results_text.pack(fill="both", expand=True, padx=4, pady=4)

    # ── File / load actions ───────────────────────────────────────────

    def _load_dicom(self):
        if not _CORE_IMPORTED:
            messagebox.showerror("Import Error", "Core library not importable.")
            return
        d = filedialog.askdirectory(title="Select DICOM Directory")
        if not d:
            return
        self._update_status(f"Loading DICOM: {d} …")
        self.root.update_idletasks()
        try:
            self._qct.load_dicom(d)
            self._on_volume_loaded()
        except Exception as exc:
            messagebox.showerror("DICOM Load Error", str(exc))
            self._update_status("DICOM load failed.")

    def _load_nifti(self):
        if not _CORE_IMPORTED:
            messagebox.showerror("Import Error", "Core library not importable.")
            return
        f = filedialog.askopenfilename(
            title="Select NIfTI File",
            filetypes=[("NIfTI files", "*.nii *.nii.gz"), ("All files", "*.*")],
        )
        if not f:
            return
        self._update_status(f"Loading NIfTI: {f} …")
        self.root.update_idletasks()
        try:
            self._qct.load_nifti(f)
            self._on_volume_loaded()
        except Exception as exc:
            messagebox.showerror("NIfTI Load Error", str(exc))
            self._update_status("NIfTI load failed.")

    def _load_roi_json(self):
        if not _CORE_IMPORTED:
            messagebox.showerror("Import Error", "Core library not importable.")
            return
        f = filedialog.askopenfilename(
            title="Select ROI JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not f:
            return
        try:
            self._qct.load_rois_from_json(f)
            self._refresh_roi_list()
            self._draw_slice()
            self._update_status(f"Loaded ROIs from {f}")
        except Exception as exc:
            messagebox.showerror("ROI Load Error", str(exc))

    def _load_demo(self):
        if not _CORE_IMPORTED:
            messagebox.showerror("Import Error", "Core library not importable.")
            return
        try:
            self._qct = _build_synthetic_demo()
            self._on_volume_loaded()
            self._refresh_roi_list()
            self._draw_slice()
            self._update_status("Demo volume loaded with synthetic spine ROIs.")
        except Exception as exc:
            messagebox.showerror("Demo Error", str(exc))

    def _on_volume_loaded(self):
        vol = self._qct.ct_volume
        if vol is None:
            return
        nz = vol.shape[0]
        self._slice_var.set(nz // 2)
        self._slice_slider.configure(to=max(0, nz - 1))
        self._draw_slice()
        self._update_status(
            f"Volume loaded: shape={vol.shape}, spacing={self._qct.voxel_spacing}"
        )

    # ── ROI controls ──────────────────────────────────────────────────

    def _add_roi(self):
        if not _CORE_IMPORTED:
            messagebox.showerror("Import Error", "Core library not importable.")
            return
        dlg = _ROIDialog(self.root)
        if dlg.result is None:
            return
        r = dlg.result
        self._qct.add_roi(r["tag"], r["center"], r["radius_mm"])
        self._refresh_roi_list()
        self._draw_slice()

    def _remove_roi(self):
        if not _CORE_IMPORTED:
            return
        sel = self._roi_listbox.curselection()
        if not sel:
            return
        tag = self._roi_listbox.get(sel[0]).split()[0]
        self._qct.remove_roi(tag)
        self._refresh_roi_list()
        self._draw_slice()

    def _clear_rois(self):
        if not _CORE_IMPORTED:
            return
        if not messagebox.askyesno("Clear ROIs", "Remove all registered ROIs?"):
            return
        self._qct.rois.clear()
        self._refresh_roi_list()
        self._draw_slice()

    def _refresh_roi_list(self):
        if not _CORE_IMPORTED:
            return
        items = []
        for tag, roi in self._qct.rois.items():
            cx, cy, cz = roi.center_ijk
            items.append(f"{tag}  ({cx:.0f},{cy:.0f},{cz:.0f})  r={roi.radius_mm:.1f}mm")
        self._roi_listvar.set(items)

    # ── Analysis ──────────────────────────────────────────────────────

    def _run_analysis(self):
        if not _CORE_IMPORTED:
            messagebox.showerror("Import Error", "Core library not importable.")
            return
        if self._qct.ct_volume is None:
            messagebox.showwarning("No Volume", "Please load a CT volume first.")
            return
        if not self._qct.rois:
            messagebox.showwarning("No ROIs", "Please add at least one ROI first.")
            return

        self._update_status("Running analysis … please wait")
        self.root.update_idletasks()

        def _worker():
            try:
                self._qct.run()
                summary = self._qct.get_results_summary()
                self.root.after(0, self._show_results, summary)
            except Exception as exc:
                self.root.after(
                    0,
                    messagebox.showerror,
                    "Analysis Error",
                    str(exc),
                )
                self.root.after(0, self._update_status, "Analysis failed.")

        threading.Thread(target=_worker, daemon=True).start()

    def _show_results(self, summary: str):
        self._results_text.configure(state="normal")
        self._results_text.delete("1.0", "end")
        self._results_text.insert("end", summary)
        self._results_text.configure(state="disabled")
        self._update_status("Analysis complete.")

    # ── Export ────────────────────────────────────────────────────────

    def _export_json(self):
        if not _CORE_IMPORTED:
            messagebox.showerror("Import Error", "Core library not importable.")
            return
        if not self._qct.axial_results and not self._qct.dxa_results:
            messagebox.showwarning("No Results", "Run the analysis first.")
            return
        f = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not f:
            return
        try:
            self._qct.export_results_json(f)
            self._update_status(f"Results exported to {f}")
        except Exception as exc:
            messagebox.showerror("Export Error", str(exc))

    # ── CT viewer ─────────────────────────────────────────────────────

    def _on_slice_change(self, _event=None):
        idx = int(self._slice_var.get())
        self._current_slice = idx
        self._draw_slice()

    def _draw_slice(self):
        if not _MPL_AVAILABLE or self._canvas_fig is None:
            return
        if not _CORE_IMPORTED or self._qct.ct_volume is None:
            return

        vol = self._qct.ct_volume
        nz = vol.shape[0]
        idx = min(self._current_slice, nz - 1)
        self._slice_label.configure(text=f"Slice: {idx} / {nz - 1}")

        self._ax.clear()
        self._ax.imshow(
            vol[idx],
            cmap="gray",
            vmin=-200,
            vmax=1000,
            interpolation="bilinear",
            aspect="equal",
        )
        self._ax.set_title(f"Axial CT — slice {idx}")
        self._ax.axis("off")

        # Overlay ROI circles (projected onto axial slice)
        if _CORE_IMPORTED:
            for tag, roi in self._qct.rois.items():
                cx, cy, cz = roi.center_ijk
                r_x = roi.radius_mm / self._qct.voxel_spacing[0]
                r_y = roi.radius_mm / self._qct.voxel_spacing[1]
                # Show ROI if it spans this slice
                dz = abs(cz - idx) * self._qct.voxel_spacing[2]
                if dz <= roi.radius_mm:
                    color = "lime" if tag.upper() in SPINE_TAGS else "cyan"
                    ellipse = mpatches.Ellipse(
                        (cx, cy), 2 * r_x, 2 * r_y,
                        fill=False, edgecolor=color, linewidth=1.5,
                    )
                    self._ax.add_patch(ellipse)
                    self._ax.text(
                        cx, cy - r_y - 2, tag,
                        color=color, fontsize=7, ha="center",
                    )

        self._canvas_fig.draw()

    # ── Helpers ───────────────────────────────────────────────────────

    def _update_status(self, msg: str):
        self._status_var.set(msg)
        logger.info(msg)

    def _show_about(self):
        messagebox.showinfo(
            "About",
            "Bone Density QCT Analysis v48\n\n"
            "Computes volumetric BMD (vBMD) via axial CT ROIs\n"
            "and pseudo-DXA areal BMD via AP projection.\n\n"
            "Core library: bone_density_qct_merged_cleaned_workflow_v48.py",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not _TK_AVAILABLE:
        print(
            "tkinter is not available in this Python installation.\n"
            "Install tkinter (e.g. `sudo apt-get install python3-tk`) and retry.\n"
            "For command-line usage run bone_density_qct_merged_cleaned_workflow_v48.py directly.",
            file=sys.stderr,
        )
        return 1

    if not _CORE_IMPORTED:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Import Error",
            "Could not import bone_density_qct_merged_cleaned_workflow_v48.\n"
            "Make sure both files are in the same directory.",
        )
        root.destroy()
        return 1

    root = tk.Tk()
    app = BoneDensityApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
