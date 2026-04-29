Experimental Environment

All experiments were conducted on a workstation configured as follows:

Component: Operating System
Specification: Windows 11

Component: CPU
Specification: Intel(R) Core(TM) i9-14900KF

Component: GPU
Specification: NVIDIA GeForce RTX 5060 Ti, 16 GB VRAM

Component: RAM
Specification: 64 GB

Component: NVIDIA Driver
Specification: 581.80

Component: Driver-supported CUDA Runtime
Specification: 13.0

Component: CUDA Toolkit
Specification: 11.8

Component: Conda Environment
Specification: MSA-PINN

Component: Python
Specification: 3.11.3

Component: PyTorch
Specification: 2.7.1+cu118

Component: TorchVision
Specification: 0.22.1+cu118

Component: TorchAudio
Specification: 2.7.1+cu118

The main Python dependencies used in this project include:

NumPy 2.4.1
pandas 2.3.3
Matplotlib 3.10.8
SymPy 1.14.0
NetworkX 3.6.1
Streamlit 1.55.0
PyTorch 2.7.1+cu118


Environment Setup

The experimental environment can be created and activated using the following commands:

conda env create -f environment.yml
conda activate MSA-PINN
pip install -r requirements.txt


PyTorch and CUDA Configuration

This project uses the GPU-enabled PyTorch distribution compiled with CUDA 11.8 support:

PyTorch      2.7.1+cu118
TorchVision  0.22.1+cu118
TorchAudio   2.7.1+cu118

The suffix +cu118 indicates that the installed PyTorch packages are built against CUDA 11.8. Therefore, this environment is intended for NVIDIA GPU-accelerated execution rather than CPU-only computation.

It should be noted that the CUDA Toolkit version installed on the system is 11.8, while the NVIDIA driver supports CUDA Runtime 13.0. This configuration is valid because NVIDIA drivers are generally backward compatible with applications built against earlier CUDA toolkit versions.


GPU Compatibility Note

This project was tested on an NVIDIA GeForce RTX 5060 Ti GPU. During execution, PyTorch may report a compatibility warning similar to:

NVIDIA GeForce RTX 5060 Ti with CUDA capability sm_120 is not compatible with the current PyTorch installation.

This warning indicates that the installed PyTorch CUDA 11.8 build may not include native binary support for the GPU architecture sm_120. Although the environment is GPU-enabled, users running the project on newer NVIDIA GPUs may need to install a PyTorch build with newer CUDA support when full compatibility is required.