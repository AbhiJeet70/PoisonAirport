# PoisonAirport

This repository contains Python scripts for training and evaluating Graph Convolutional Networks (GCNs) on the Airports dataset using PyTorch Geometric. The scripts include functionalities for data loading, model definition, training, evaluation, and perturbation analysis.

## Contents

1. `main.py`: Main script to load data, train models, and analyze perturbation effects.
2. `gcn_model.py`: Definition of the GCN model architecture with increased complexity and batch normalization.
3. `utils.py`: Utility functions including data splitting, dataset statistics, degree analysis, and perturbation functions.
4. `plot.py`: Functions for plotting accuracy against perturbation percentages.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/PoisonAirport.git
   cd PoisonAirport
   
2. Install dependencies:

     ```bash
    pip install torch torch_geometric matplotlib

3. Requirements
   ```bash
    Python 3.x
    PyTorch
    PyTorch Geometric
    Matplotlib

4. Run main.py to process multiple countries and analyze perturbations.
    
     ```bash
    python main.py
