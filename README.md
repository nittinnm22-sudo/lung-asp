# Lung-ASP — Lung Tumor Segmentation & Radiomics

FDG PET/CT pipeline for primary lung-tumour segmentation, shape analysis, and
metabolic radiomics (SUVpeak, Dmax, NHOCmax, NHOPmax, gETU, sphericity).

---

## Repository files

| File | Purpose |
|------|---------|
| `Lung_ASP.py` | Main pipeline — segmentation → metrics → QC images |
| `Lung_ASP_GUI.py` | Tkinter desktop GUI (wraps `Lung_ASP.py`) |
| `advanced_metrics.py` | Radiomics engine (SUV, Dmax, NHOCmax, NHOPmax, gETU, sphericity) |
| `Mask_QC.py` | QC overlay images (3×3 grid + per-metric figures) |
| `requirements.txt` | Python package dependencies |

---

## Quick start — copy files to your desktop folder

### 1. Prerequisites

You need **Python 3.10 or later**.  
Check with: `python --version` (or `python3 --version` on macOS/Linux).

If Python is not installed, download it from <https://www.python.org/downloads/>.

---

### 2. Download / copy the scripts

**Option A — clone with Git** (recommended)

```bash
# Open a terminal / command prompt, then:
git clone https://github.com/nittinnm22-sudo/lung-asp.git
cd lung-asp
```

**Option B — download a ZIP**

1. Click the green **Code** button on the GitHub page.
2. Choose **Download ZIP**.
3. Unzip the downloaded file to any folder (e.g. `C:\Users\YourName\Desktop\lung-asp`
   or `~/Desktop/lung-asp`).

After either option you should have these files together in **one folder**:

```
lung-asp/
├── Lung_ASP.py
├── Lung_ASP_GUI.py
├── advanced_metrics.py
├── Mask_QC.py
└── requirements.txt
```

> **Important:** all four `.py` files must be in the **same folder**.  
> They import each other by name, so they cannot be separated.

---

### 3. Install dependencies

Open a terminal / command prompt **inside the `lung-asp` folder**, then run:

```bash
pip install -r requirements.txt
```

If `pip` is not found, try:

```bash
python -m pip install -r requirements.txt
# or on macOS/Linux:
python3 -m pip install -r requirements.txt
```

> **TotalSegmentator** (anatomical structures) is included in `requirements.txt`.  
> After installation you also need to download its model weights once:
> ```bash
> totalseg_download_weights -t total
> ```
> If you skip this step the pipeline will still run — it falls back to a
> CT-intensity based lung region instead.

---

### 4. Run the desktop GUI

From inside the `lung-asp` folder:

```bash
python Lung_ASP_GUI.py
# or on macOS/Linux:
python3 Lung_ASP_GUI.py
```

The GUI window will open. Steps:

1. Choose **NIfTI mode** (default) or **DICOM mode**.
2. Browse to your **PET NIfTI** (SUV-normalised) and **CT NIfTI** (HU) files.
3. Select an **Output directory** where results will be saved.
4. Adjust options if needed (side, device, lobes, …).
5. Click **▶ Run Pipeline**.

Results written to the output directory:

```
<output_dir>/
├── tumor_mask.nii.gz      # Segmented tumour mask
├── constraint_mask.nii.gz # Lung constraint region
├── Mask_QC.png            # 3×3 PET/CT/Fusion QC grid
├── Dmax_QC.png            # Maximum diameter QC
├── NHOCmax_QC.png         # Hotspot-to-centroid QC
├── NHOPmax_QC.png         # Hotspot-to-periphery QC
├── gETU_QC.png            # gETU / MTV QC
├── metrics.json           # All radiomics metrics (JSON)
└── metrics.csv            # All radiomics metrics (CSV)
```

---

### 5. Run from the command line (no GUI)

```bash
python Lung_ASP.py  path/to/pet.nii.gz  path/to/ct.nii.gz  path/to/output_dir
```

Additional options:

```
--side          left | right | auto (default: auto)
--device        cpu | cuda | mps   (default: cpu)
--fast          Use fast TotalSegmentator mode
--excl-dilation Dilation (mm) applied to exclusion structures (default: 5)
--hilar-radius  Hilar exclusion radius in mm (default: 20)
--no-node-sep   Disable lymph-node / primary separation
--fg-frac       Foreground seed fraction for random walker (default: 0.7)
--bg-frac       Background seed fraction (default: 0.1)
--beta          Random walker beta (default: 130)
--tol           Solver tolerance (default: 0.001)
```

Example:

```bash
python Lung_ASP.py pet_suv.nii.gz ct_hu.nii.gz ./results --side right --device cpu
```

---

### 6. Use the modules in your own Python script

Because all files are in the **same folder**, you can import them directly once
your working directory is set to that folder:

```python
import sys
sys.path.insert(0, r"C:\Users\YourName\Desktop\lung-asp")  # ← your folder path

import advanced_metrics
import Mask_QC
import Lung_ASP

# Run the full pipeline
metrics = Lung_ASP.process_case(
    pet_path="pet_suv.nii.gz",
    ct_path="ct_hu.nii.gz",
    out_dir="./results",
)
print(metrics)

# Or compute metrics only (no segmentation)
metrics, qc_coords = advanced_metrics.compute_all_metrics(
    pet_nifti_path="pet_suv.nii.gz",
    tumor_mask_path="tumor_mask.nii.gz",
    ct_nifti_path="ct_hu.nii.gz",
)
print(metrics)
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'nibabel'` | Run `pip install -r requirements.txt` inside the lung-asp folder |
| `ModuleNotFoundError: No module named 'Mask_QC'` | Make sure all `.py` files are in the **same** folder and you run Python from that folder |
| GUI does not open (Linux) | Install Tk: `sudo apt install python3-tk` |
| TotalSegmentator model missing | Run `totalseg_download_weights -t total` once after installation |
| CUDA / GPU errors | Set **Device** to `cpu` in the GUI, or pass `--device cpu` on the command line |

---

## License

MIT — see [LICENSE](LICENSE).
