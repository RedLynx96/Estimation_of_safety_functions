# 🌀 AI-Driven Control of Chaos: A Transformer-Based Approach for Dynamical Systems 🌀

Welcome to the official repository for the paper:  
**David Valle, Alexandre Wagemakers, Miguel A. F. Sanjuán; *AI-Driven Control of Chaos: A Transformer-Based Approach for Dynamical Systems*.**  
<!-- Published in [Chaos, January 2025 (Vol. X, Issue X)](https://doi.org-).--> 
<!-- DOI: [10.1063/-](https://doi.org/10.1063/-).--> 

This repository provides the code and environment setup to reproduce the results of our paper. Whether you're interested in training new models from scratch, working with external datasets, or leveraging cloud-based resources, we've got you covered! 🚀

---

## 🔧 Setting Up the Environment

To ensure smooth execution of the code, we recommend setting up a dedicated Python environment using Anaconda. Here's how to get started:

1. Clone this repository:
   ```bash
   git clone https://github.com/RedLynx96/Estimation_of_safety_functions.git
   cd Estimation_of_safety_functions
   ```

2. Create the Conda environment from the provided environment.yml file:
   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:
   ```bash
    conda activate SafetyFunctionAI
   ```

## 📚 Available Scripts

### 1️⃣ **`train_transformer_no_files.py`**
Train a new model from scratch without using external datasets.  
This script focuses on generating models based purely on internal computations.

**Run it with:**
   ```bash
   python train_transformer_no_files.py
   ```


### 2️⃣ **`Safety_trainer.py`**
Train models using external datasets, specifically matrices shaped `(50, 1000)`.  
Ensure your datasets are pre-processed and properly formatted before running this script.
**Ensure that cuDNN is property installed to reduce computation times!**

**Run it with:**
   ```bash
   python Safety_trainer.py
   ```

### 3️⃣ **`predictor_colab.ipynb`**
Run the predictor in Google Colab, bypassing the need for a local graphics card.

**To use:**

1. **Open the notebook in [Google Colab](https://colab.research.google.com/github/RedLynx96/Estimation_of_safety_functions/blob/main/Predictor_Colab.ipynb).**

2. **Enable a GPU or TPU in the Colab runtime settings:**
   - Go to `Runtime` > `Change runtime type`.
   - Select `GPU` or `TPU` from the Hardware accelerator dropdown.

3. **Execute the notebook cells** to run the predictor seamlessly on Google's infrastructure.

### 🎥 Demo Videos

#### GUI and Colab Predictor Demonstration

[![Demo](https://www.youtube.com/watch?v=HB7j7R2oMJs/0.jpg)](https://www.youtube.com/watch?v=HB7j7R2oMJs)

## ✨ Features

- **Reproducibility**: Follow our simple steps to replicate the results from the paper. 🔄
- **Flexibility**: Train models with or without external datasets. 🧠
- **Customizable**: Easily tweak parameters to explore new ideas. ⚙️

---

## 📜 Reference

If you use this code in your work, please cite our paper:

```bibtex
@misc{valle2024aidrivencontrolchaostransformerbased,
      title={AI-Driven Control of Chaos: A Transformer-Based Approach for Dynamical Systems}, 
      author={David Valle and Rubén Capeáns and Alexandre Wagemakers and Miguel A. F. Sanjuán},
      year={2024},
      eprint={2412.17357},
      archivePrefix={arXiv},
      primaryClass={nlin.CD},
      url={https://arxiv.org/abs/2412.17357}, 
}
