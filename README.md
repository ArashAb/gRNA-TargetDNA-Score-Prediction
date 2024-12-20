# CRISPR-based gRNA Performance Prediction with ESM2 Embeddings

## Overview

This repository provides a pipeline for processing CRISPR perturbation datasets and training a deep learning model to predict guide RNA (gRNA) performance using embeddings generated by the ESM2 (Evolutionary Scale Modeling) language model.

### Data Source

We use the STable_09 and STable_15 Excel files from [ORCS BioGRID Dataset 19](https://orcs.thebiogrid.org/Dataset/19). These datasets contain CRISPR perturbations, their corresponding guide RNAs (gRNAs), target DNA sequences (where applicable), and experimental scores.

### What the Code Does

1. **Data Processing (`data_processing.py`)**:
   - Reads the Excel files from `data/` directory.
   - For STable_09:
     - Skips the first row as it's a header row not needed for analysis.
     - Extracts "Perturbations" and "Average Score".
     - Derives `gRNA` from the first part of the `Perturbations` field.
     - Renames the `Perturbations` column to `Target DNA`.
   - For STable_15:
     - Extracts `sgRNA` and `Average log fold change IFNgamma - mock`.
     - Renames them to `gRNA` and `Average Score`.
     - Sets `Target DNA` to `None` since this dataset does not provide a target DNA sequence.
   - Combines both datasets into one DataFrame.
   - Saves the combined dataset as `gRNA_TargetDNA_AverageScore_Combined.xlsx` in `output/`.

2. **Training (`train.py`)**:
   - Loads the combined dataset.
   - Uses the ESM2 model to generate embeddings for gRNA and Target DNA sequences.
     - If `Target DNA` is `None`, a zero-vector placeholder is used.
   - Concatenates gRNA and target DNA embeddings to form the input feature vector.
   - Trains a regression model defined in `model.py` to predict the experimental scores.
   - Logs training progress and saves the trained model to `output/`.

### Dependencies

- Python 3.8+
- `pandas`
- `torch`
- `esm` (for ESM2 embeddings)
- `openpyxl` (for reading Excel files)

You can install these via:
```bash
pip install pandas torch esm openpyxl


crispr-regression-model/
│
├── data/
│   ├── STable_09_Sel_STARSOutput.xlsx
│   ├── STable_15_IFNg_data_STARS.xlsx
│
├── output/   # Will be created if not present
│
├── src/
│   ├── data_processing.py
│   ├── model.py
│   ├── train.py
│   ├── utils.py
│
└── README.md

