# A Structure-Adaptive Physics-Informed Neural Network for Complex Scattering Problems

This repository contains PyTorch implementations and experiment configurations for physics-informed neural networks applied to Maxwell-equation scattering and wave-propagation cases.

## Repository structure

```text
.
├── 1_ComplexMultimodalScatteringScenarios/
│   ├── Config/              # CSV experiment configurations
│   ├── Module/              # Network, training, and visualization modules
│   ├── Results/             # Generated after training; ignored by Git
│   └── run_maxwell.py       # Entry point for this experiment
├── 2_Single-WavePlanePECScattering/
│   ├── Config/
│   ├── Module/
│   ├── Results/
│   └── run_maxwell.py
├── 3_TravelingWavePropagatio/
│   ├── Config/
│   ├── Module/
│   ├── Results/
│   └── run_maxwell.py
├── environment.yml          # Conda environment specification
├── requirements.txt         # Pip dependencies
└── README_original.txt      # Original environment notes
```

## Environment

The experiments were originally tested with:

- Windows 11
- Python 3.11.3
- PyTorch 2.7.1 with CUDA 11.8
- NVIDIA GPU acceleration

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate MSA-PINN
pip install -r requirements.txt
```

The `requirements.txt` file uses the PyTorch CUDA 11.8 wheel index. If your GPU or CUDA stack differs, install the matching PyTorch build from the official PyTorch selector before running the experiments.

## Running experiments

Run each experiment from inside its own directory, because the scripts use relative paths for `Config/` and `Results/`.

```bash
cd 1_ComplexMultimodalScatteringScenarios
python run_maxwell.py
```

```bash
cd ../2_Single-WavePlanePECScattering
python run_maxwell.py
```

```bash
cd ../3_TravelingWavePropagatio
python run_maxwell.py
```

Training outputs, figures, losses, error tables, and model checkpoints are written under each experiment's `Results/` directory. These generated files are ignored by Git to keep the repository lightweight.

## Configuration

Experiment parameters are stored in the corresponding `Config/Maxwell_*.csv` files. Key options include the model type, network width, coordinate/output dimensions, training steps, grid size, and recording interval.

## Notes for GitHub upload

This GitHub-ready copy removes local IDE metadata, Python bytecode caches, and previously generated training outputs. The source code, configuration files, environment files, and original environment notes are retained.
