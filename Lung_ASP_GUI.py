"""Lung_ASP_GUI.py  v6.3
Tkinter GUI for the Lung Tumor Segmentation & Radiomics Pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Dict, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DICOM conversion helpers
# ---------------------------------------------------------------------------

def _dicom_to_nifti_ct(dicom_dir: str, out_nifti: str) -> None:
    """Convert a CT DICOM series to a NIfTI file (HU values preserved)."""
    try:
        import dicom2nifti
        dicom2nifti.dicom_series_to_nifti(dicom_dir, out_nifti,
                                           reorient_nifti=True)
    except Exception as exc:
        logger.error("CT DICOM→NIfTI conversion failed: %s", exc)
        raise


def _dicom_to_nifti_pet_suv(
    dicom_dir: str,
    out_nifti: str,
    weight_kg: float = 70.0,
    dose_mbq: float = 370.0,
) -> None:
    """Convert a PET DICOM series to SUV-normalised NIfTI.

    SUV = (pixel_value / (dose_MBq × 1e6 / weight_kg))
    """
    try:
        import pydicom
        import nibabel as nib
        import numpy as np
        from glob import glob

        dcm_files = sorted(glob(os.path.join(dicom_dir, "*.dcm")))
        if not dcm_files:
            dcm_files = sorted(glob(os.path.join(dicom_dir, "**", "*.dcm"),
                                    recursive=True))
        if not dcm_files:
            raise FileNotFoundError(f"No DICOM files found in {dicom_dir}")

        slices = []
        for f in dcm_files:
            ds = pydicom.dcmread(f)
            slices.append(ds)

        # Sort by ImagePositionPatient z
        slices.sort(key=lambda s: float(getattr(s, "ImagePositionPatient", [0, 0, 0])[2]))

        # Read rescale slope/intercept from first slice
        slope = float(getattr(slices[0], "RescaleSlope", 1.0))
        intercept = float(getattr(slices[0], "RescaleIntercept", 0.0))

        pixel_spacing = slices[0].PixelSpacing
        slice_thickness = float(getattr(slices[0], "SliceThickness", 1.0))

        volume = np.stack(
            [s.pixel_array.astype(np.float32) * slope + intercept for s in slices],
            axis=-1,
        )

        # SUV normalisation
        suv_factor = weight_kg * 1000.0 / (dose_mbq * 1e6)
        volume = volume * suv_factor

        dx = float(pixel_spacing[1])
        dy = float(pixel_spacing[0])
        dz = slice_thickness

        affine = np.diag([dx, dy, dz, 1.0])
        img = nib.Nifti1Image(volume, affine)
        nib.save(img, out_nifti)
        logger.info("Saved PET SUV NIfTI → %s", out_nifti)

    except Exception as exc:
        logger.error("PET DICOM→NIfTI conversion failed: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Lobe map
# ---------------------------------------------------------------------------

LOBE_MAP: Dict[str, str] = {
    "right_upper_lobe": "right_upper_lobe",
    "right_middle_lobe": "right_middle_lobe",
    "right_lower_lobe": "right_lower_lobe",
    "left_upper_lobe": "left_upper_lobe",
    "left_lower_lobe": "left_lower_lobe",
}

# ---------------------------------------------------------------------------
# GUI Application
# ---------------------------------------------------------------------------

class LungASPApp(tk.Tk):
    """Main application window for the Lung Tumor Segmentation Pipeline."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Lung-ASP  v6.3 — Lung Tumor Segmentation & Radiomics")
        self.geometry("900x820")
        self.resizable(True, True)
        self.configure(bg="#1e1e2e")

        self._nifti_mode = tk.BooleanVar(value=True)
        self._pet_path = tk.StringVar()
        self._ct_path = tk.StringVar()
        self._out_dir = tk.StringVar()
        self._dicom_pet_dir = tk.StringVar()
        self._dicom_ct_dir = tk.StringVar()

        # Options
        self._side = tk.StringVar(value="auto")
        self._device = tk.StringVar(value="cpu")
        self._fast = tk.BooleanVar(value=False)
        self._excl_dilation = tk.IntVar(value=5)
        self._hilar_radius = tk.IntVar(value=20)
        self._node_sep = tk.BooleanVar(value=True)
        self._fg_frac = tk.DoubleVar(value=0.7)
        self._bg_frac = tk.DoubleVar(value=0.1)
        self._beta = tk.DoubleVar(value=130.0)
        self._tolerance = tk.DoubleVar(value=1e-3)

        # Lobe selection
        self._lobe_vars: Dict[str, tk.BooleanVar] = {
            k: tk.BooleanVar(value=True) for k in LOBE_MAP
        }

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")

        main = tk.Frame(self, bg="#1e1e2e")
        main.pack(fill="both", expand=True, padx=10, pady=10)

        # --- Input section ---
        inp = tk.LabelFrame(main, text=" Input ", bg="#1e1e2e", fg="#cdd6f4",
                            font=("Helvetica", 10, "bold"))
        inp.pack(fill="x", pady=(0, 6))

        mode_frame = tk.Frame(inp, bg="#1e1e2e")
        mode_frame.grid(row=0, column=0, columnspan=3, sticky="w", padx=6, pady=4)
        tk.Radiobutton(mode_frame, text="NIfTI mode", variable=self._nifti_mode,
                       value=True, bg="#1e1e2e", fg="#cdd6f4",
                       selectcolor="#313244",
                       command=self._toggle_mode).pack(side="left")
        tk.Radiobutton(mode_frame, text="DICOM mode", variable=self._nifti_mode,
                       value=False, bg="#1e1e2e", fg="#cdd6f4",
                       selectcolor="#313244",
                       command=self._toggle_mode).pack(side="left", padx=10)

        # NIfTI entries
        self._nifti_frame = tk.Frame(inp, bg="#1e1e2e")
        self._nifti_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=6)
        self._make_file_row(self._nifti_frame, "PET NIfTI:", self._pet_path, 0)
        self._make_file_row(self._nifti_frame, "CT NIfTI: ", self._ct_path, 1)

        # DICOM entries (hidden by default)
        self._dicom_frame = tk.Frame(inp, bg="#1e1e2e")
        self._dicom_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=6)
        self._dicom_frame.grid_remove()
        self._make_dir_row(self._dicom_frame, "PET DICOM:", self._dicom_pet_dir, 0)
        self._make_dir_row(self._dicom_frame, "CT DICOM: ", self._dicom_ct_dir, 1)

        self._make_dir_row(inp, "Output Dir:", self._out_dir, 3, is_frame=False)

        # --- Options section ---
        opt = tk.LabelFrame(main, text=" Options ", bg="#1e1e2e", fg="#cdd6f4",
                            font=("Helvetica", 10, "bold"))
        opt.pack(fill="x", pady=(0, 6))

        opt_inner = tk.Frame(opt, bg="#1e1e2e")
        opt_inner.pack(fill="x", padx=6, pady=4)

        self._make_combo(opt_inner, "Side:", self._side,
                         ["auto", "left", "right"], 0, 0)
        self._make_combo(opt_inner, "Device:", self._device,
                         ["cpu", "cuda", "mps"], 0, 2)
        self._make_check(opt_inner, "Fast mode", self._fast, 0, 4)

        self._make_spinbox(opt_inner, "Excl. dilation (mm):", self._excl_dilation,
                           1, 0, lo=0, hi=30)
        self._make_spinbox(opt_inner, "Hilar radius (mm):", self._hilar_radius,
                           1, 2, lo=5, hi=60)
        self._make_check(opt_inner, "Node separation", self._node_sep, 1, 4)

        self._make_spinbox(opt_inner, "FG fraction:", self._fg_frac,
                           2, 0, lo=0.0, hi=1.0, inc=0.05, fmt="%.2f")
        self._make_spinbox(opt_inner, "BG fraction:", self._bg_frac,
                           2, 2, lo=0.0, hi=1.0, inc=0.05, fmt="%.2f")
        self._make_spinbox(opt_inner, "Beta:", self._beta,
                           2, 4, lo=1.0, hi=1000.0, inc=10.0, fmt="%.1f")
        self._make_spinbox(opt_inner, "Tolerance:", self._tolerance,
                           3, 0, lo=1e-6, hi=1e-1, inc=1e-4, fmt="%.6f")

        # --- Lobe selection ---
        lobe_frame = tk.LabelFrame(main, text=" Lobe Selection ", bg="#1e1e2e",
                                    fg="#cdd6f4", font=("Helvetica", 10, "bold"))
        lobe_frame.pack(fill="x", pady=(0, 6))
        lobe_inner = tk.Frame(lobe_frame, bg="#1e1e2e")
        lobe_inner.pack(fill="x", padx=6, pady=4)
        for col, (key, _) in enumerate(LOBE_MAP.items()):
            label = key.replace("_", " ").title()
            tk.Checkbutton(lobe_inner, text=label, variable=self._lobe_vars[key],
                           bg="#1e1e2e", fg="#cdd6f4",
                           selectcolor="#313244").grid(row=0, column=col, padx=4)

        # --- Run button + progress ---
        run_frame = tk.Frame(main, bg="#1e1e2e")
        run_frame.pack(fill="x", pady=(0, 6))
        self._run_btn = tk.Button(run_frame, text="▶  Run Pipeline",
                                   bg="#89b4fa", fg="#1e1e2e",
                                   font=("Helvetica", 11, "bold"),
                                   relief="flat", padx=20, pady=6,
                                   command=self._start_pipeline)
        self._run_btn.pack(side="left")
        self._progress = ttk.Progressbar(run_frame, mode="indeterminate", length=300)
        self._progress.pack(side="left", padx=12)

        # --- Log box ---
        log_frame = tk.LabelFrame(main, text=" Log ", bg="#1e1e2e", fg="#cdd6f4",
                                   font=("Helvetica", 10, "bold"))
        log_frame.pack(fill="both", expand=True, pady=(0, 6))
        self._log_box = scrolledtext.ScrolledText(
            log_frame, height=8, bg="#181825", fg="#cdd6f4",
            font=("Courier", 9), state="disabled",
        )
        self._log_box.pack(fill="both", expand=True, padx=4, pady=4)

        # --- Results display ---
        res_frame = tk.LabelFrame(main, text=" Results ", bg="#1e1e2e", fg="#cdd6f4",
                                   font=("Helvetica", 10, "bold"))
        res_frame.pack(fill="x", pady=(0, 4))
        self._results_text = scrolledtext.ScrolledText(
            res_frame, height=10, bg="#181825", fg="#a6e3a1",
            font=("Courier", 9), state="disabled",
        )
        self._results_text.pack(fill="both", expand=True, padx=4, pady=4)

        self._setup_logging()

    # ------------------------------------------------------------------
    # Widget helpers
    # ------------------------------------------------------------------

    def _make_file_row(self, parent, label, var, row):
        tk.Label(parent, text=label, bg="#1e1e2e", fg="#cdd6f4",
                 width=12, anchor="w").grid(row=row, column=0, sticky="w", pady=2)
        tk.Entry(parent, textvariable=var, bg="#313244", fg="#cdd6f4",
                 insertbackground="white", width=55).grid(row=row, column=1,
                                                           sticky="ew", padx=4)
        tk.Button(parent, text="Browse…", bg="#585b70", fg="#cdd6f4",
                  relief="flat",
                  command=lambda v=var: self._browse_file(v)).grid(row=row, column=2)
        parent.columnconfigure(1, weight=1)

    def _make_dir_row(self, parent, label, var, row, is_frame=True):
        target = parent if is_frame else parent
        tk.Label(target, text=label, bg="#1e1e2e", fg="#cdd6f4",
                 width=12, anchor="w").grid(row=row, column=0, sticky="w", pady=2,
                                             padx=6)
        tk.Entry(target, textvariable=var, bg="#313244", fg="#cdd6f4",
                 insertbackground="white", width=55).grid(row=row, column=1,
                                                           sticky="ew", padx=4)
        tk.Button(target, text="Browse…", bg="#585b70", fg="#cdd6f4",
                  relief="flat",
                  command=lambda v=var: self._browse_dir(v)).grid(row=row, column=2,
                                                                    padx=6)
        target.columnconfigure(1, weight=1)

    def _make_combo(self, parent, label, var, values, row, col):
        tk.Label(parent, text=label, bg="#1e1e2e", fg="#cdd6f4").grid(
            row=row, column=col, sticky="w", padx=(6, 2))
        cb = ttk.Combobox(parent, textvariable=var, values=values, width=8,
                          state="readonly")
        cb.grid(row=row, column=col + 1, sticky="w", padx=(0, 10))

    def _make_check(self, parent, label, var, row, col):
        tk.Checkbutton(parent, text=label, variable=var, bg="#1e1e2e", fg="#cdd6f4",
                       selectcolor="#313244").grid(row=row, column=col,
                                                    columnspan=2, sticky="w", padx=6)

    def _make_spinbox(self, parent, label, var, row, col, lo=0, hi=100,
                      inc=1, fmt="%d"):
        tk.Label(parent, text=label, bg="#1e1e2e", fg="#cdd6f4").grid(
            row=row, column=col, sticky="w", padx=(6, 2))
        sb = tk.Spinbox(parent, textvariable=var, from_=lo, to=hi, increment=inc,
                        format=fmt, width=8, bg="#313244", fg="#cdd6f4",
                        insertbackground="white", buttonbackground="#585b70")
        sb.grid(row=row, column=col + 1, sticky="w", padx=(0, 10))

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _toggle_mode(self) -> None:
        if self._nifti_mode.get():
            self._nifti_frame.grid()
            self._dicom_frame.grid_remove()
        else:
            self._nifti_frame.grid_remove()
            self._dicom_frame.grid()

    def _browse_file(self, var: tk.StringVar) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("NIfTI files", "*.nii *.nii.gz"), ("All files", "*.*")]
        )
        if path:
            var.set(path)

    def _browse_dir(self, var: tk.StringVar) -> None:
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def _log(self, msg: str) -> None:
        self._log_box.configure(state="normal")
        self._log_box.insert("end", msg + "\n")
        self._log_box.see("end")
        self._log_box.configure(state="disabled")

    def _setup_logging(self) -> None:
        """Redirect root logger to the GUI log box."""
        class _GUIHandler(logging.Handler):
            def __init__(self_, app):
                super().__init__()
                self_._app = app

            def emit(self_, record):
                msg = self_.format(record)
                self_._app.after(0, self_._app._log, msg)

        handler = _GUIHandler(self)
        handler.setFormatter(logging.Formatter("%(levelname)s  %(message)s"))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

    def _start_pipeline(self) -> None:
        """Validate inputs and launch pipeline in a background thread."""
        if self._nifti_mode.get():
            pet_path = self._pet_path.get().strip()
            ct_path = self._ct_path.get().strip()
            if not pet_path or not ct_path:
                messagebox.showerror("Error", "Please select PET and CT NIfTI files.")
                return
        else:
            pet_dir = self._dicom_pet_dir.get().strip()
            ct_dir = self._dicom_ct_dir.get().strip()
            if not pet_dir or not ct_dir:
                messagebox.showerror("Error", "Please select PET and CT DICOM directories.")
                return

        out_dir = self._out_dir.get().strip()
        if not out_dir:
            messagebox.showerror("Error", "Please select an output directory.")
            return

        self._run_btn.configure(state="disabled")
        self._progress.start(10)
        thread = threading.Thread(target=self._run_pipeline, daemon=True)
        thread.start()

    def _run_pipeline(self) -> None:
        """Run the full pipeline in a background thread."""
        try:
            from Lung_ASP import process_case

            out_dir = self._out_dir.get().strip()

            if self._nifti_mode.get():
                pet_path = self._pet_path.get().strip()
                ct_path = self._ct_path.get().strip()
            else:
                # Convert DICOM → NIfTI
                import tempfile
                tmpdir = tempfile.mkdtemp(prefix="lung_asp_")
                pet_path = os.path.join(tmpdir, "pet_suv.nii.gz")
                ct_path = os.path.join(tmpdir, "ct_hu.nii.gz")
                self.after(0, self._log, "Converting DICOM → NIfTI…")
                _dicom_to_nifti_ct(self._dicom_ct_dir.get().strip(), ct_path)
                _dicom_to_nifti_pet_suv(self._dicom_pet_dir.get().strip(), pet_path)

            selected_lobes = [k for k, v in self._lobe_vars.items() if v.get()]

            self.after(0, self._log, "Starting pipeline…")
            metrics = process_case(
                pet_path=pet_path,
                ct_path=ct_path,
                out_dir=out_dir,
                side=self._side.get(),
                device=self._device.get(),
                fast=self._fast.get(),
                exclusion_dilation_mm=float(self._excl_dilation.get()),
                hilar_radius_mm=float(self._hilar_radius.get()),
                separate_nodes=self._node_sep.get(),
                fg_frac=float(self._fg_frac.get()),
                bg_frac=float(self._bg_frac.get()),
                beta=float(self._beta.get()),
                tolerance=float(self._tolerance.get()),
                selected_lobes=selected_lobes,
            )
            self.after(0, self._display_results, metrics)
            self.after(0, self._log, "Pipeline completed successfully.")
        except Exception as exc:
            logger.exception("Pipeline error: %s", exc)
            self.after(0, messagebox.showerror, "Pipeline Error", str(exc))
        finally:
            self.after(0, self._run_btn.configure, {"state": "normal"})
            self.after(0, self._progress.stop)

    def _display_results(self, metrics: Optional[Dict]) -> None:
        """Display metrics in the results text box."""
        self._results_text.configure(state="normal")
        self._results_text.delete("1.0", "end")

        if not metrics:
            self._results_text.insert("end", "No metrics returned.\n")
            self._results_text.configure(state="disabled")
            return

        lines = [
            "═" * 55,
            "  LUNG TUMOR RADIOMICS RESULTS",
            "═" * 55,
            "",
            "── Shape / Volume ──────────────────────────────",
            f"  Volume          : {metrics.get('volume_cm3', 0):.3f} cm³",
            f"  Dmax            : {metrics.get('Dmax_mm', 0):.2f} mm",
            f"  Sphericity      : {metrics.get('sphericity', 0):.4f}",
            f"  Asphericity     : {metrics.get('asphericity', 0):.4f}",
            "",
            "── PET Metrics ─────────────────────────────────",
            f"  SUVmax          : {metrics.get('SUVmax', 0):.3f}",
            f"  SUVmean         : {metrics.get('SUVmean', 0):.3f}",
            f"  SUVpeak (1cc)   : {metrics.get('SUVpeak', 0):.3f}",
            f"  TLG             : {metrics.get('TLG', 0):.3f}",
            f"  MTV             : {metrics.get('MTV_cm3', 0):.3f} cm³",
            "",
            "── Heterogeneity ───────────────────────────────",
            f"  NHOCmax         : {metrics.get('NHOCmax', 0):.4f}",
            f"  NHOPmax         : {metrics.get('NHOPmax', 0):.4f}",
            "",
            "── gETU ────────────────────────────────────────",
            f"  gETU            : {metrics.get('gETU', 0):.3f}",
            f"  gETU MTV        : {metrics.get('gETU_MTV_cm3', 0):.3f} cm³",
            f"  gETU SUVmean    : {metrics.get('gETU_SUVmean', 0):.3f}",
            "═" * 55,
        ]
        self._results_text.insert("end", "\n".join(lines))
        self._results_text.configure(state="disabled")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Launch the Lung-ASP GUI."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")
    app = LungASPApp()
    app.mainloop()


if __name__ == "__main__":
    main()
