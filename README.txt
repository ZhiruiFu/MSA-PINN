# MSA-PINN Maxwell Experiments

This repository contains the experimental code and configuration files for Maxwell-equation-based electromagnetic scattering and wave propagation experiments using the MSA-PINN framework.

## 1. Experimental Hardware and Software Environment

All experiments were conducted on the following workstation:

| Component | Specification |
|---|---|
| Operating System | Windows 11 |
| CPU | Intel(R) Core(TM) i9-14900KF |
| GPU | NVIDIA GeForce RTX 5060 Ti, 16 GB VRAM |
| NVIDIA Driver | 581.80 |
| Driver-supported CUDA Runtime | 13.0 |
| CUDA Toolkit | 11.8 |

## 2. Environment Setup

This project uses a Conda environment. The recommended environment name is:

```bash
MSA-PINN
```

Create and activate the environment with:

```bash
conda env create -f environment.yml
conda activate MSA-PINN
```

The environment uses the GPU-enabled PyTorch build compiled with CUDA 11.8 support:

| Package | Version |
|---|---|
| PyTorch | 2.7.1+cu118 |
| TorchVision | 0.22.1+cu118 |
| TorchAudio | 2.7.1+cu118 |

The `+cu118` suffix indicates that these packages are built for CUDA 11.8. Therefore, this environment is intended for NVIDIA GPU acceleration rather than CPU-only execution.

### GPU Compatibility Note

The experiments were tested on an NVIDIA GeForce RTX 5060 Ti. This GPU corresponds to the newer `sm_120` CUDA compute capability. During execution, PyTorch may display a warning similar to:

```text
NVIDIA GeForce RTX 5060 Ti with CUDA capability sm_120 is not compatible with the current PyTorch installation.
```

This warning indicates that the installed PyTorch version may not fully support the GPU architecture. Although the environment uses the CUDA 11.8 GPU-enabled PyTorch build, compatibility warnings may appear on RTX 50-series GPUs. The experiments in this repository were tested under the configuration described above.

## 3. Repository Structure

This repository contains three experiment folders, each corresponding to a different electromagnetic scattering or propagation scenario:

```text
1_ComplexMultimodalScatteringScenarios/
2_Single-WavePlanePECScattering/
3_TravelingWavePropagatio/
```

Each experiment folder contains the main execution script:

```text
run_maxwell.py
```

## 4. Running Experiments

To run an experiment, enter one of the experiment folders and execute `run_maxwell.py`:

```bash
cd 1_ComplexMultimodalScatteringScenarios
python run_maxwell.py
```

The same procedure applies to the other experiment folders:

```bash
cd 2_Single-WavePlanePECScattering
python run_maxwell.py
```

```bash
cd 3_TravelingWavePropagatio
python run_maxwell.py
```

## 5. Selecting Experimental Configurations

Before running an experiment, the configuration index can be modified in `run_maxwell.py`.

The experimental task is specified by:

```python
task = Training.model("Maxwell", 3)
```

Here, the second argument `3` corresponds to `ini_num`, which is used to select a specific configuration file. To run a different experimental configuration, change this value to the desired configuration index.

For example:

```python
task = Training.model("Maxwell", 1)
```

## 6. Output Files

After execution, the program automatically generates a results folder. A typical results folder contains:

```text
Error/
Figure/
Loss/
Models/
MSE/
Clock_time.csv
Maxwell_1.csv
```

The folders and files generally store error data, generated figures, loss curves, trained model checkpoints, mean squared error results, runtime records, and experiment-specific output data.

## 7. Notes

- Use `environment.yml` to reproduce the Conda environment.
- The environment is designed for GPU execution with CUDA-enabled PyTorch.
- If running on a different GPU or CUDA setup, PyTorch/CUDA compatibility should be checked before execution.
- If a PyTorch GPU compatibility warning appears on newer NVIDIA GPUs, verify whether the installed PyTorch version supports the corresponding CUDA compute capability.
