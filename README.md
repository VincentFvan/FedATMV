# FedATMV

This repository contains the official implementation and detailed mathematical proof for the paper:

**《FedATMV: Towards Heterogeneous Hybrid Federated Learning via Adaptive Server-Side Training and Model Variation》**

## Repository Overview

The repository is organized into two main directories:

.<br>
├── Code/ # Contains all source code and instructions for running experiments.<br>
└── Proof/ # Contains the detailed mathematical proof for the convergence analysis mentioned in the paper (Section Ⅲ-D).

## Code Directory Structure

The source code within the `Code/` directory is organized as follows:

Code/<br>
├── atmv.py # Main script to run experiments<br>
├── config.py # All hyperparameters and settings<br>
├── multi_run.py # Script to run multiple experiments and aggregate results<br>
├── README.md # This file<br>
├── requirements.txt # Python package dependencies<br>
├── data/<br>
│ └── shakespeare/ # Directory for the Shakespeare dataset<br>
│ ├── train/<br>
│ └── test/<br>
├── models/<br>
│ ├── lstm.py # LSTM model for Shakespeare<br>
│ └── vgg.py # VGG-16 model for CIFAR-100<br>
└── utils/<br>
├── language_utils.py # Helper functions for text data<br>
├── ShakeSpeare.py # Dataset loader for Shakespeare<br>
└── ShakeSpeare_reduce.py # Dataset loader with client data reduction

## Setup

### 1. Clone Repository

```bash
git clone https://github.com/VincentFvan/FedATMV.git
```

### 2. Navigate to the Code Directory

All subsequent commands should be run from within the `Code/` directory.

```bash
cd FedATMV/Code
```

### 3. Create Environment and Install Dependencies

We recommend using a virtual environment (e.g., conda or venv).

```bash
# Using conda
conda create -n fedatmv python=3.9
conda activate fedatmv

# Install dependencies
pip install -r requirements.txt
```

### 4. Dataset Preparation

- **CIFAR-10/100**: The datasets will be automatically downloaded by torchvision the first time you run an experiment. They will be saved in a `data/` directory.

- **Shakespeare**: The Shakespeare dataset is already included in the `data/shakespeare` directory. No further action is needed.

## Running Experiments

### 1. Configure the Experiment

All experimental settings and hyperparameters can be modified in `config.py`. Key parameters include:

- `DATASET`: Choose from `'cifar10'`, `'cifar100'`, or `'shake'`.
- `ORIGIN_MODEL`: Select the corresponding model (`'resnet'`, `'vgg'`, `'lstm'`).
- `IS_IID`, `ALPHA`: Control client-side data heterogeneity.
- `SERVER_IID`, `BETA`: Control server-side data heterogeneity.
- `T_ROUNDS`, `M_CLIENTS`, `K_EPOCHS`, `E_EPOCHS`: Federated learning process parameters.
- `MU`, `RHO`, `THETA`: Hyperparameters for the FedATMV algorithm.

### 2. Run a Single Experiment

To run a single experiment with the settings from `src/config.py`:

```bash
python main.py
```

The script will train all baseline models and FedATMV, print the final results to the console.

### 3. Run Multiple Experiments

To run an experiment multiple times and get the mean and standard deviation of the results (as done in the paper), you can use the `multi_run.py` script.

First, configure the number of repetitions in `multi_run.py`:

```python
# In multi_run.py
n_repeat = 5 # Set the desired number of repetitions
```

Then, run the script:

```bash
python multi_run.py
```

This will create a timestamped directory inside `output/` containing the raw results for each run and a `summary.csv` file with the aggregated statistics (mean and standard deviation).
