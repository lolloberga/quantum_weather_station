# Quantum Weather Station

## Description
In a world burdened by air pollution, the inte-
gration of state-of-the-art sensor calibration techniques utilizing
Quantum Computing (QC) and Machine Learning (ML) holds
promise for enhancing the accuracy and efficiency of air quality
monitoring systems in smart cities. This article investigates
the process of calibrating inexpensive optical fine-dust sensors
through advanced methodologies such as Deep Learning (DL)
and Quantum Machine Learning (QML). The objective of the
project is to compare four sophisticated algorithms from both
classical and quantum realms to discern their disparities and ex-
plore potential alternative approaches for improving the precision
and dependability of particulate matter measurements in urban
air quality surveillance. Classical Feed-Forward Neural Net-
works (FFNN) and Long Short-Term Memory (LSTM) models
are evaluated against their quantistic counterparts: Variational
Quantum Regressors (VQR) and Quantum LSTM (QLSTM)
circuits. Through meticulous testing, including hyperparameter
optimization and cross-validation, the study assesses the potential
of quantum models for refining calibration performance. Our
analysis shows that: the FFNN model achieved superior cali-
bration accuracy on the test-set compared to the VQR model
in terms of lower L1 loss function (2.92 vs 4.81); the QLSTM
slightly outperformed the LSTM model (loss on the test-set: 2.70
vs 2.77), despite using fewer trainable weights (66 vs 482)

------------

## Table of content
- [Features](#Features)
- [Cloning the repo](#cloning-the-repo)
- [Installation](#Installation)
- [Usage](#Usage)
- [Technologies Used](#technologies-used)
- [Directory Structure](#directory-structure)

-----------

## Features
- Traditional machine learning models like Feed-Forward Neural Network (FFNN) and Long Short-Term Memory (LSTM).
- Quantum machine learning models including Quantum LSTM (QLSTM) and Variational Quantum Regression (VQR).
- Hyperparameter tuning and model improvement scripts.
- Cross-validation modules for robust testing of models.

## Cloning the repo
To clone the repo through HTTPS or SSH, you must have installed Git on your operating system.<br>
Then you can open a new terminal and type the following command (this is the cloning through HTTPS):
```bash
    git clone https://github.com/lolloberga/quantum_weather_station.git
```

If you don't have installed Git, you can simply download the repository by pressing <i>"Download ZIP"</i>.

## Installation
To run the Quantum Weather Station, you need Python 3.8 (or above) and the following packages:

```bash
pip install -r requirements.txt
```
or via conda:
```bash
conda create --name <env_name> --file requirements.txt
```

## Usage
To use the Quantum Weather Station, navigate to the project directory and run the following command:

```bash
python main.py --action [ACTION_NAME]
```
Replace [ACTION_NAME] with the desired action, such as ANN_MODEL, LSTM_MODEL, or any specific tuning or testing action as defined in the code.

## Technologies Used
- Python 3.8 (or above)
- PyTorch for neural network implementations. 
- PennyLane for quantum machine learning. 
- scikit-learn, numpy, pandas, and more for data handling and machine learning operations
------------------

## Directory Structure
    .
    ├── config
    ├── cross_validation
    ├── hyperparameters_tuning
    ├── improvements
    ├── model
    │   ├── loss_functions
    │   ├── train
    │   │   ├── base
    │   │   ├── hyperparams
    ├── notebook
    ├── notification
    ├── resources
    ├── runs
    └── utils

- **config**: This folder contains configuration files that are used to set up various parameters or environment settings necessary for the project. These configurations might include model parameters, API keys, or other settings that help modularize and manage changes more easily.
- **cross_validation**: Contains scripts or modules dedicated to cross-validating the machine learning models used in the project. This ensures the models are robust and perform well on unseen data. Typically, this involves dividing the data into multiple sets and testing each model's performance iteratively.
- **hyperparams_tuning**: Includes scripts for tuning the hyperparameters of the machine learning models. Hyperparameter tuning is crucial for optimizing model performance by systematically searching for the best combination of parameters (like learning rate, number of layers, etc.).
- **improvements**: This folder likely contains scripts or modules aimed at enhancing the existing models. Improvements could be in the form of algorithm tweaks, additional features, or other optimizations that enhance the performance or functionality of the models.
- **model**: Houses the model definitions and potentially training routines. This could include different types of models like ANN (Artificial Neural Networks), LSTM (Long Short-Term Memory networks), and quantum models such as QLSTM and VQR (Variational Quantum Regression).
- **notebooks**: Contains Jupyter notebooks that are typically used for exploratory data analysis, visualization, and try-out code snippets. Notebooks are useful for documenting the workflow and sharing results with visual support.
- **notification**: This folder might contain scripts that handle sending notifications related to model outcomes, alerts, or updates. This could involve sending emails, pushing updates to a dashboard, or other forms of notifications.
- **resources**: Includes additional files that support the project, such as datasets, images, or documents that are used within the project.
- **utils**: Contains utility scripts that provide functionality used across various parts of the project. These might include data preprocessing, logging, or common functions that are reused within different scripts or modules.
