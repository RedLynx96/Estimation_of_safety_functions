# 🌀 AI-Driven Control of Chaos: A Transformer-Based Approach for Dynamical Systems 🌀

Welcome to the official repository for the following papers:  

**David Valle, Rubén Capeans, Alexandre Wagemakers, Miguel A. F. Sanjuán; *AI-Driven Control of Chaos: A Transformer-Based Approach for Dynamical Systems*.**  
Available on [arXiv:2412.17357](https://arxiv.org/abs/2412.17357).  

**David Valle, Rubén Capeans, Alexandre Wagemakers, Miguel A. F. Sanjuán; *Controlling Transient Chaos in the Lorenz System with Machine Learning*.**  
Available on [arXiv:2501.17588](https://arxiv.org/abs/2501.17588).  

This repository provides the code and environment setup to reproduce the results of our papers. Whether you're interested in training new models from scratch, working with external datasets, or leveraging cloud-based resources, we've got you covered! 🚀  

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

### 3️⃣ **`Predictor_GUI.py`**

A standalone GUI application designed for predicting safety functions.

#### **Key Features**
- **Graphical User Interface**: Intuitive and user-friendly for seamless interaction.
- **Real-Time Predictions**: Provides visualizations of safety function predictions interactively.

#### **System Requirements**
To ensure the application runs smoothly, ensure the following:
- A **graphics card** with appropriate drivers and CUDA support is installed.

**Run it with:**
   ```bash
   python Predictor_GUI.py
   ```
> **Note**: This application is designed for **local use** due to its graphical nature and reliance on GPU resources. Ensure your system meets the hardware requirements (including a compatible graphics card) before running the GUI.

### 4️⃣ **`predictor_colab.ipynb`**
Run the predictor in Google Colab, bypassing the need for a local graphics card.

**To use:**

1. **Open the notebook in [Google Colab](https://colab.research.google.com/github/RedLynx96/Estimation_of_safety_functions/blob/main/Predictor_Colab.ipynb).**

2. **Enable a GPU or TPU in the Colab runtime settings:**
   - Go to `Runtime` > `Change runtime type`.
   - Select `GPU` or `TPU` from the Hardware accelerator dropdown.

3. **Execute the notebook cells** to run the predictor seamlessly on Google's infrastructure.

### 5️⃣ **Lorenz_Classical_Safety_Estimator_Matlab.m**

This MATLAB script is **mandatory** for computing the classical safety function of the Lorenz system, as it accounts for variable noise in data. The computed safety function is crucial for comparisons with the transformer-based model.

#### **Key Features**
- **Generates Sampled Data (`xVectors`)**: Creates a dataset of sampled orbits from the Lorenz system under partial control.
- **Computes Classical Safety Function (`Us`)**: Estimates the classical safety function with noise-adaptive modeling.
- **Visualizes the Lorenz Attractor**: Provides 3D trajectory plots and Poincaré section analysis.
- **Supports Controlled Trajectory Computation**: Enables testing of controlled orbits using classical or machine-learning-based safety functions.

#### **Usage Instructions**
1. **Run the script in MATLAB**:
   ```matlab
   Lorenz_Classical_Safety_Estimator_Matlab

### 🎥 Demo Videos

#### GUI and Colab Predictor Demonstration

<div align="center">
  <a href="https://www.youtube.com/watch?v=HB7j7R2oMJs">
    <img src="https://img.youtube.com/vi/HB7j7R2oMJs/0.jpg" alt="Demo" width="600">
  </a>
</div>

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
      author={David Valle and Rubén Capeans and Alexandre Wagemakers and Miguel A. F. Sanjuán},
      year={2024},
      eprint={2412.17357},
      archivePrefix={arXiv},
      primaryClass={nlin.CD},
      url={https://arxiv.org/abs/2412.17357}, 
}

@misc{valle2025controllingtransientchaoslorenz,
      title={Controlling Transient Chaos in the Lorenz System with Machine Learning}, 
      author={David Valle and Rubén Capeans and Alexandre Wagemakers and Miguel A. F. Sanjuán},
      year={2025},
      eprint={2501.17588},
      archivePrefix={arXiv},
      primaryClass={nlin.CD},
      url={https://arxiv.org/abs/2501.17588}, 
}