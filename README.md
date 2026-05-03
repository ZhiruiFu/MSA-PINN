# MSA-PINN Maxwell Experiments

This repository provides the experimental code, configuration files, and running scripts for Maxwell-equation-based electromagnetic scattering and wave propagation experiments using the MSA-PINN framework.

## 1. Repository Structure

The repository contains three experiment folders, each corresponding to a different electromagnetic scattering or wave propagation scenario:

```text
MSA-PINN/
├── environment.yml
├── README.md
├── 1_ComplexMultimodalScatteringScenarios/
│   ├── Config/
│   ├── Module/
│   ├── run_maxwell.py
│   └── ...
├── 2_Single-WavePlanePECScattering/
│   ├── Config/
│   ├── Module/
│   ├── run_maxwell.py
│   └── ...
└── 3_TravelingWavePropagatio/
    ├── Config/
    ├── Module/
    ├── run_maxwell.py
    └── ...
```

The three main experiment folders are:

```text
1_ComplexMultimodalScatteringScenarios/
2_Single-WavePlanePECScattering/
3_TravelingWavePropagatio/
```

Each experiment folder contains its own configuration files, functional modules, and execution script.

## 2. Experimental Hardware and Software Environment

All experiments were conducted on the following workstation.

| Component | Specification |
|---|---|
| CPU | Intel(R) Core(TM) i9-14900KF |
| GPU | NVIDIA GeForce RTX 5060 Ti, 16 GB VRAM |
| CUDA Toolkit | 11.8 |

## 3. Environment Setup

### 3.1 Create and Activate the Conda Environment

Create the Conda environment from the provided `environment.yml` file and activate it:

```bash
conda env create -f environment.yml
conda activate MSA-PINN
```

### 3.2 Install GPU-Supported PyTorch with CUDA 11.8

Install PyTorch, TorchVision, TorchAudio, and the CUDA 11.8 runtime package:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

This step enables GPU acceleration for CUDA-compatible NVIDIA GPUs.

## 4. Running Experiments

### 4.1 Select an Experiment Scenario

Before running the code, select one of the three experiment folders according to the target scenario.

For the complex multimodal scattering scenario:

```bash
cd 1_ComplexMultimodalScatteringScenarios
```

For the single plane-wave PEC scattering scenario:

```bash
cd 2_Single-WavePlanePECScattering
```

For the traveling wave propagation scenario:

```bash
cd 3_TravelingWavePropagatio
```

### 4.2 Select an Experimental Configuration

Before executing an experiment, the configuration index can be modified in `run_maxwell.py`.

The experimental task is specified by:

```python
task = Training.model("Maxwell", 3)
```

In this statement, the first argument `"Maxwell"` specifies the physical problem type, while the second argument `3` corresponds to `ini_num`. This index is used to select a configuration file from the `Config/` directory.

For example:

```python
task = Training.model("Maxwell", 3)
```

uses the following configuration file:

```text
Config/Maxwell_3.csv
```

To run another configuration, change the second argument. For example:

```python
task = Training.model("Maxwell", 1)
```

uses:

```text
Config/Maxwell_1.csv
```

Therefore, the general correspondence is:

```text
task = Training.model("Maxwell", n)
```

corresponds to:

```text
Config/Maxwell_n.csv
```

### 4.3 Run the Selected Experiment

After entering the selected experiment folder and confirming the configuration index, run:

```bash
python run_maxwell.py
```

For example, to run the first experiment scenario:

```bash
cd 1_ComplexMultimodalScatteringScenarios
python run_maxwell.py
```

## 5. Output Files

After execution, the program automatically generates a results folder for storing numerical and visualization outputs. A typical results folder contains:

```text
Error/
Figure/
Loss/
Models/
MSE/
Clock_time.csv
Maxwell_1.csv
```

The generated files and folders have the following functions:

| Folder/File | Description |
|---|---|
| `Error/` | Stores error-related numerical results. |
| `Figure/` | Stores generated figures and visualization results. |
| `Loss/` | Stores training loss records and loss-curve data. |
| `Models/` | Stores trained model checkpoints. |
| `MSE/` | Stores mean squared error evaluation results. |
| `Clock_time.csv` | Records runtime information. |
| `Maxwell_*.csv` | Stores experiment-specific output data. |

## 6. Notes

- The Conda environment should be reproduced using the provided `environment.yml` file.
- The experiments are designed for GPU execution with CUDA-enabled PyTorch.
- CUDA 11.8 is used as the target CUDA runtime in the provided installation command.
- If the code is executed on a different GPU or CUDA setup, PyTorch/CUDA compatibility should be checked before running the experiments.
- If a PyTorch GPU compatibility warning appears on newer NVIDIA GPUs, verify whether the installed PyTorch version supports the corresponding CUDA compute capability.
- When using relative paths in the scripts, each experiment should be executed from inside its corresponding experiment directory.
