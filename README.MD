# 🌀 Deep Learning-Based Analysis of Basins of Attraction 🌀

Welcome to the official repository for the paper:  
**David Valle, Alexandre Wagemakers, Miguel A. F. Sanjuán; *Deep learning-based analysis of basins of attraction*.**  
Published in [Chaos, March 2024 (Vol. 34, Issue 3)](https://doi.org/10.1063/5.0159656).  
DOI: [10.1063/5.0159656](https://doi.org/10.1063/5.0159656)

This repository provides the code and environment setup to reproduce the results of our paper. Whether you're interested in training new models from scratch or working with external datasets, we've got you covered! 🚀

---

## 🔧 Setting Up the Environment

To ensure smooth execution of the code, we recommend setting up a dedicated Python environment using Anaconda. Here's how to get started:

1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo-name/your-repo.git
   cd your-repo

2. Create the Conda environment from the provided environment.yml file:
   ```bash
   conda env create -f environment.yml

3. Activate the environment:
   ```bash
    conda activate <your-env-name>

## 📚 Available Scripts

### 1️⃣ **`train_transformer_no_files.py`**
Train a new model from scratch without using external datasets.  
This script focuses on generating models based purely on internal computations.

**Run it with:**
```bash
python train_transformer_no_files.py


### 2️⃣ **`Safety_trainer.py`**
Train models using external datasets, specifically matrices shaped `(50, 1000)`.  
Ensure your datasets are pre-processed and properly formatted before running this script.

**Run it with:**
```bash
python Safety_trainer.py

## ✨ Features

- **Reproducibility**: Follow our simple steps to replicate the results from the paper. 🔄
- **Flexibility**: Train models with or without external datasets. 🧠
- **Customizable**: Easily tweak parameters to explore new ideas. ⚙️

---

## 📜 Reference

If you use this code in your work, please cite our paper:

```bibtex
@article{valle2024deep,
  title={Deep learning-based analysis of basins of attraction},
  author={David Valle, Alexandre Wagemakers, Miguel A. F. Sanju{\'a}n},
  journal={Chaos},
  volume={34},
  number={3},
  pages={033105},
  year={2024},
  publisher={AIP Publishing},
  doi={10.1063/5.0159656}
}
