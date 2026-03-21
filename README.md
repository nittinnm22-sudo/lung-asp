# Lung ASP — FDG PET/CT Primary Lung Tumor Segmentation Pipeline

Lung ASP is an automated pipeline for primary lung tumor segmentation and radiomics metric extraction from FDG PET/CT images. It leverages TotalSegmentator for anatomical priors, Random Walker segmentation, and a 6-strategy primary tumor isolation algorithm to deliver robust, reproducible results with full QC overlay generation.

---

## Features

- **TotalSegmentator integration** — automatic organ/structure segmentation for exclusion masks
- **Random Walker segmentation** — probabilistic tumor delineation on PET
- **Exclusion masks** — removes non-tumor uptake (heart, liver, bladder, etc.)
- **Tumor protection zone** — prevents erosion of true tumor voxels
- **Hilar zone handling** — separates hilar lymph node uptake from primary tumor
- **Node separation** — disconnects adjacent nodal structures
- **6-strategy primary tumor isolation** — robust selection of the dominant lesion
- **Full radiomics metrics** — SUVmax, SUVmean, SUVpeak, MTV, TLG, Dmax, NHOPmax, NHOCmax, Sphericity, Asphericity, gETU
- **QC overlay generation** — 5 publication-ready QC images (Mask_QC.png, Dmax_QC.png, NHOPmax_QC.png, NHOCmax_QC.png, gETU_QC.png)
- **Tkinter GUI** — user-friendly interface with threaded execution, progress bar, and live log

---

## Installation

```bash
git clone https://github.com/nittinnm22-sudo/lung-asp.git
cd lung-asp
pip install -r requirements.txt
```

---

## Usage

### Command-Line Interface (CLI)

```bash
python Lung_ASP.py --pet pet.nii.gz --ct ct.nii.gz --out output_dir
```

**Optional flags:**

| Flag | Description | Default |
|------|-------------|---------|
| `--no-totalseg` | Skip TotalSegmentator (use PET-only pipeline) | TotalSegmentator enabled |
| `--dpi 300` | DPI for QC overlay images | 150 |
| `--totalseg-timeout 900` | Timeout (seconds) for TotalSegmentator | 600 |

### Graphical User Interface (GUI)

```bash
python Lung_ASP_GUI.py
```

The GUI lets you browse for PET/CT files and output directory, configure optional flags, and monitor progress via a real-time log and progress bar.

---

## Outputs

| File | Description |
|------|-------------|
| `tumor_mask.nii.gz` | Binary mask of the primary lung tumor |
| `constraint_mask.nii.gz` | Combined exclusion/constraint mask |
| `Mask_QC.png` | PET/CT overlay with tumor contour |
| `Dmax_QC.png` | Maximum diameter visualization |
| `NHOPmax_QC.png` | Nearest hot-object-to-primary-max distance overlay |
| `NHOCmax_QC.png` | Nearest hot-object-to-centroid-max distance overlay |
| `gETU_QC.png` | Gross Extended Tumour Uptake overlay |

---

## Metrics

| Metric | Description |
|--------|-------------|
| SUVmax | Maximum standardized uptake value in the tumor |
| SUVmean | Mean SUV across the tumor volume |
| SUVpeak | Peak SUV (1 cm³ sphere centered on hottest voxel) |
| MTV | Metabolic tumor volume (mL) |
| TLG | Total lesion glycolysis (MTV × SUVmean) |
| Dmax | Maximum 3-D diameter of the tumor (mm) |
| NHOPmax | Nearest-hot-object distance from the tumor periphery (mm) |
| NHOCmax | Nearest-hot-object distance from the tumor centroid (mm) |
| Sphericity | Shape descriptor: ratio of sphere surface area to tumor surface area |
| Asphericity | 1 − Sphericity |
| gETU | Gross Extended Tumor Uptake — total PET-positive volume including satellites |

---

## Pipeline Overview

1. **Input loading** — PET and CT NIfTI volumes are loaded and resampled to a common grid
2. **TotalSegmentator** — anatomical structures are segmented to build exclusion masks
3. **PET thresholding** — candidate voxels selected above an adaptive SUV threshold
4. **Exclusion mask application** — non-tumour uptake regions are suppressed
5. **Random Walker** — probabilistic label propagation refines the tumour boundary
6. **Tumor protection zone** — anchors the segmentation around the hottest voxel
7. **Hilar exclusion** — hilar zone uptake is separated from the primary mass
8. **Node separation** — connected-component analysis isolates individual lesions
9. **6-strategy primary isolation** — largest/hottest/closest-to-centre candidate selected
10. **Metrics computation** — all radiomics indices calculated via `advanced_metrics.py`
11. **QC overlay generation** — 5 annotated PNG images written by `Mask_QC.py`

---

## License

See [LICENSE](LICENSE) for details.
