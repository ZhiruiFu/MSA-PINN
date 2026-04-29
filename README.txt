# README

## 1. Experimental Hardware Information

All experiments were conducted on a workstation with the following hardware and software environment:

- Operating System: Windows 11
- CPU: Intel(R) Core(TM) i9-14900KF
- GPU: NVIDIA GeForce RTX 5060 Ti, 16 GB VRAM
- RAM: 64 GB
- NVIDIA Driver: 581.80
- Driver-supported CUDA Runtime: 13.0
- CUDA Toolkit: 11.8


## 2. Experimental Environment Configuration

This project uses a Conda environment for experiments. The environment name is:

```bash
MSA-PINN
```

The environment can be created and the required dependencies can be installed as follows:

```bash
conda env create -f environment.yml
conda activate MSA-PINN
pip install -r requirements.txt
```

The main software and library versions are listed below:

- Conda Environment: MSA-PINN
- Python Version: 3.11.3
- PyTorch: 2.7.1+cu118
- TorchVision: 0.22.1+cu118
- TorchAudio: 2.7.1+cu118
- NumPy: 2.4.1
- pandas: 2.3.3
- Matplotlib: 3.10.8
- SymPy: 1.14.0
- NetworkX: 3.6.1
- Streamlit: 1.55.0


## 3. Environment Configuration Notes

This project uses the GPU-enabled PyTorch build. The specific versions are as follows:

- PyTorch: 2.7.1+cu118
- TorchVision: 0.22.1+cu118
- TorchAudio: 2.7.1+cu118

The suffix +cu118 indicates that the installed PyTorch, TorchVision, and TorchAudio packages are compiled with CUDA 11.8 support. Therefore, this experimental environment is intended for NVIDIA GPU acceleration rather than CPU-only execution.

The GPU used in this project is the NVIDIA GeForce RTX 5060 Ti. Its CUDA compute capability is sm_120. During execution, PyTorch may display a compatibility warning similar to the following:

```text
NVIDIA GeForce RTX 5060 Ti with CUDA capability sm_120 is not compatible with the current PyTorch installation.
```

This warning indicates that the currently installed PyTorch version may not fully support the compute architecture of this GPU. Although the environment uses the CUDA 11.8 GPU-enabled PyTorch build, the RTX 5060 Ti corresponds to the newer sm_120 architecture, so a compatibility warning may appear during execution. The experiments in this project were still tested using the configuration described above.


## 4. Experimental Procedure

This project contains three experiment folders, each corresponding to a different electromagnetic scattering or propagation scenario:

```text
1_ComplexMultimodalScatteringScenarios
2_Single-WavePlanePECScattering
3_TravelingWavePropagatio
```

Each experiment folder contains the corresponding execution script:

```text
run_maxwell.py
```

To run the experiments, enter each of the three experiment folders and execute the corresponding run_maxwell.py file. Before running the script, the configuration index can be modified as needed to execute different experimental configurations.

In run_maxwell.py, the experimental task can be specified using the following statement:

```python
task = Training.model("Maxwell", 3)
```

Here, the second parameter, 3, corresponds to ini_num and is used to select a specific configuration file. By changing this value, different config settings can be executed. For example, it can be changed to another index to run the corresponding Maxwell experiment configuration.

An example command for running the experiment is shown below:

```bash
python run_maxwell.py
```

After execution, the program automatically generates a results folder to store the output results of the experiment. A typical results folder contains the following contents:

```text
Error
Figure
Loss
Models
MSE
Clock_time.csv
Maxwell_1.csv
```

The contents are described as follows:

- Error: stores error-related results;
- Figure: stores visualization figures generated during the experiment;
- Loss: stores the loss curves or loss-related results during training;
- Models: stores the trained model files;
- MSE: stores mean squared error-related results;
- Clock_time.csv: records the experiment running time;
- Maxwell_1.csv: records numerical results related to the Maxwell experiment.

The generated results folder can be compared with the Example results provided in the project. By comparing the errors, loss curves, MSE metrics, and visualization results, users can verify whether the experimental outputs are consistent with the example results, thereby confirming the correctness of the environment configuration and experimental procedure.
