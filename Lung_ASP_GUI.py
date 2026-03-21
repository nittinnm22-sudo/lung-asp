"""
Lung_ASP_GUI.py  v6.3
Tkinter GUI for FDG PET/CT Lung Tumor Segmentation (Lung ASP Pipeline)
"""

import os
import sys
import logging
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path

logger = logging.getLogger(__name__)
VERSION = "6.3"


class TextHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)

        def append():
            self.text_widget.configure(state='normal')
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.configure(state='disabled')
            self.text_widget.yview(tk.END)

        self.text_widget.after(0, append)


class LungASPApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Lung ASP v{VERSION}")
        self.root.minsize(800, 600)
        self._build_ui()
        self._setup_logging()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1)  # log frame expands

        # ── Input files ────────────────────────────────────────────────
        input_frame = ttk.LabelFrame(self.root, text="Input Files", padding=8)
        input_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        input_frame.columnconfigure(1, weight=1)

        self.pet_path = tk.StringVar()
        self.ct_path = tk.StringVar()
        self.out_dir = tk.StringVar()

        ttk.Label(input_frame, text="PET NIfTI:").grid(
            row=0, column=0, sticky="w", pady=2)
        ttk.Entry(input_frame, textvariable=self.pet_path).grid(
            row=0, column=1, sticky="ew", padx=(4, 2))
        ttk.Button(input_frame, text="Browse",
                   command=self._browse_pet).grid(row=0, column=2, padx=(2, 0))

        ttk.Label(input_frame, text="CT NIfTI:").grid(
            row=1, column=0, sticky="w", pady=2)
        ttk.Entry(input_frame, textvariable=self.ct_path).grid(
            row=1, column=1, sticky="ew", padx=(4, 2))
        ttk.Button(input_frame, text="Browse",
                   command=self._browse_ct).grid(row=1, column=2, padx=(2, 0))

        ttk.Label(input_frame, text="Output Dir:").grid(
            row=2, column=0, sticky="w", pady=2)
        ttk.Entry(input_frame, textvariable=self.out_dir).grid(
            row=2, column=1, sticky="ew", padx=(4, 2))
        ttk.Button(input_frame, text="Browse",
                   command=self._browse_output).grid(row=2, column=2, padx=(2, 0))

        # ── Options ────────────────────────────────────────────────────
        opt_frame = ttk.LabelFrame(self.root, text="Options", padding=8)
        opt_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=4)
        opt_frame.columnconfigure(3, weight=1)

        self.use_totalseg = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_frame, text="Use TotalSegmentator",
                        variable=self.use_totalseg).grid(
            row=0, column=0, sticky="w", padx=(0, 20))

        self.qc_dpi = tk.IntVar(value=150)
        ttk.Label(opt_frame, text="QC DPI:").grid(
            row=0, column=1, sticky="w")
        dpi_spin = ttk.Spinbox(opt_frame, textvariable=self.qc_dpi,
                               values=(72, 100, 150, 200, 300), width=6)
        dpi_spin.grid(row=0, column=2, sticky="w", padx=(4, 0))

        # ── Progress ───────────────────────────────────────────────────
        prog_frame = ttk.LabelFrame(self.root, text="Progress", padding=8)
        prog_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=4)
        prog_frame.columnconfigure(0, weight=1)

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            prog_frame, variable=self.progress_var,
            mode='determinate', maximum=100)
        self.progress_bar.grid(row=0, column=0, sticky="ew", pady=(0, 4))

        self.status_label = ttk.Label(prog_frame, text="Ready")
        self.status_label.grid(row=1, column=0, sticky="w")

        # ── Log output ─────────────────────────────────────────────────
        log_frame = ttk.LabelFrame(self.root, text="Log", padding=8)
        log_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=4)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self.log_text = scrolledtext.ScrolledText(
            log_frame, state='disabled', height=12,
            font=("Courier", 9), wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky="nsew")

        # ── Buttons ────────────────────────────────────────────────────
        btn_frame = ttk.Frame(self.root, padding=(10, 4, 10, 10))
        btn_frame.grid(row=4, column=0, sticky="ew")

        self.run_btn = ttk.Button(
            btn_frame, text="Run Segmentation",
            command=self._run_segmentation)
        self.run_btn.pack(side=tk.LEFT, padx=(0, 6))

        ttk.Button(btn_frame, text="Clear Log",
                   command=self._clear_log).pack(side=tk.LEFT, padx=(0, 6))

        ttk.Button(btn_frame, text="Open Output Dir",
                   command=self._open_output_dir).pack(side=tk.LEFT)

    # ------------------------------------------------------------------
    # Browse helpers
    # ------------------------------------------------------------------

    def _browse_pet(self):
        path = filedialog.askopenfilename(
            title="Select PET NIfTI file",
            filetypes=[("NIfTI files", "*.nii *.nii.gz"), ("All files", "*.*")])
        if path:
            self.pet_path.set(path)

    def _browse_ct(self):
        path = filedialog.askopenfilename(
            title="Select CT NIfTI file",
            filetypes=[("NIfTI files", "*.nii *.nii.gz"), ("All files", "*.*")])
        if path:
            self.ct_path.set(path)

    def _browse_output(self):
        path = filedialog.askdirectory(title="Select Output Directory")
        if path:
            self.out_dir.set(path)

    def _open_output_dir(self):
        out = self.out_dir.get().strip()
        if not out or not os.path.isdir(out):
            messagebox.showwarning("Warning", "Output directory is not set or does not exist.")
            return
        if sys.platform.startswith("win"):
            os.startfile(out)
        elif sys.platform == "darwin":
            os.system(f'open "{out}"')
        else:
            os.system(f'xdg-open "{out}"')

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_inputs(self):
        pet = self.pet_path.get().strip()
        ct = self.ct_path.get().strip()
        out = self.out_dir.get().strip()

        if not pet:
            messagebox.showerror("Validation Error", "PET NIfTI file is required.")
            return False
        if not os.path.isfile(pet):
            messagebox.showerror("Validation Error", f"PET file not found:\n{pet}")
            return False
        if not ct:
            messagebox.showerror("Validation Error", "CT NIfTI file is required.")
            return False
        if not os.path.isfile(ct):
            messagebox.showerror("Validation Error", f"CT file not found:\n{ct}")
            return False
        if not out:
            messagebox.showerror("Validation Error", "Output directory is required.")
            return False
        return True

    # ------------------------------------------------------------------
    # Run segmentation
    # ------------------------------------------------------------------

    def _run_segmentation(self):
        if not self._validate_inputs():
            return

        self.run_btn.configure(state='disabled')
        self.progress_var.set(0.0)
        self.status_label.configure(text="Starting…")

        pet_path = self.pet_path.get().strip()
        ct_path = self.ct_path.get().strip()
        out_dir = self.out_dir.get().strip()
        use_totalseg = self.use_totalseg.get()
        dpi = self.qc_dpi.get()

        os.makedirs(out_dir, exist_ok=True)

        t = threading.Thread(
            target=self._worker,
            args=(pet_path, ct_path, out_dir, use_totalseg, dpi),
            daemon=True,
        )
        t.start()

    def _update_progress(self, step, frac):
        def _update():
            self.progress_var.set(frac * 100)
            self.status_label.configure(text=str(step))

        self.root.after(0, _update)

    def _worker(self, pet_path, ct_path, out_dir, use_totalseg, dpi):
        try:
            import Lung_ASP
            result = Lung_ASP.segment_lung_tumor(
                pet_nifti_path=pet_path,
                ct_nifti_path=ct_path,
                output_dir=out_dir,
                use_totalseg=use_totalseg,
                dpi=dpi,
                progress_callback=self._update_progress,
            )
            self.root.after(0, lambda: self._on_complete(result))
        except Exception as e:
            logger.error(f"Segmentation failed: {e}", exc_info=True)
            self.root.after(0, lambda: self._on_error(str(e)))

    def _on_complete(self, result):
        self.progress_var.set(100.0)
        self.status_label.configure(text="Complete")
        self.run_btn.configure(state='normal')

        metrics = result if isinstance(result, dict) else {}
        suv_max = metrics.get("SUVmax", "N/A")
        mtv = metrics.get("MTV", "N/A")
        dmax = metrics.get("Dmax_mm", "N/A")
        getu = metrics.get("gETU_index", "N/A")

        msg = (
            "Segmentation completed successfully.\n\n"
            f"SUVmax:     {suv_max}\n"
            f"MTV:        {mtv}\n"
            f"Dmax (mm):  {dmax}\n"
            f"gETU index: {getu}"
        )
        messagebox.showinfo("Segmentation Complete", msg)

    def _on_error(self, msg):
        self.progress_var.set(0.0)
        self.status_label.configure(text="Error")
        self.run_btn.configure(state='normal')
        messagebox.showerror("Segmentation Failed", f"An error occurred:\n\n{msg}")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _clear_log(self):
        self.log_text.configure(state='normal')
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state='disabled')

    def _setup_logging(self):
        handler = TextHandler(self.log_text)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S"
        ))
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)


# ----------------------------------------------------------------------

def main():
    root = tk.Tk()
    app = LungASPApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
