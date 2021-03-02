# STGNN for COVID-19 prediction in Korea

## Introduction
This repository contains implementation of Spatio-Temporal Graph Neural Network and other approaches for dealing with COVID-19 case forecasting in South Korea dataset.

## Requirements

- Python 3
- Tensorflow 2.*
- Tensorflow GPU (recommended)
- Pytorch

## Installation
- Clone the repository and install the package
```
git clone https://github.com/KienMN/STGNN-for-Covid-in-Korea.git
cd STGNN-for-Covid-in-Korea
pip install -e .
```
- Install package using `pip`
```
pip install git+https://github.com/KienMN/STGNN-for-Covid-in-Korea.git
```

## Main components
There are some components/modules in our code. Please check our documentation in each file for more details.
### Datasets
This module handles:
- Load dataset from CSV file.
- Difference data.
- Split data according to date.
- Rescale data.
- Turn time series data into supervised sequence data for machine learning models.

### Models
This module contains different model classes.
- LSTM.
- Seq2Seq.
- Spatio-Temporal Graph neural network (STGNN).

### Trainer
Trainer contains every information (such as dataset, optimizer, loss function, etc) for training each type of models mentioned above.

### Callbacks
This module consists of callbacks which can be executed before/after some steps of training or testing model.

### Utils
Utility functions that can be used any where in the code.

## Experiments
All running files are stored in `tests` folder. Configuration of experiments can be found in `tests/configs`. Make `results` folder to store output of experiments.
```bash
mkdir tests/results
```
The script to run experiments is
```bash
python tests/test_{model_name}.py
```
The process in a test file is as follow, check each file for more details.
- Import necessary libraries.
- Setup configs and parameters.
- Load and process data.
- Define functions to handle post-prediction to make predictions back to original scale.
- Create model, select loss function and optimizer.
- Create a trainer containing necessary information.
- Train the model via the trainer.
- Make prediction and compute metrics.

After finishing, predictions and metric scores will be stored as CSV file in `tests/results` folder.

## References
1. COVID-19 in Korea datasets: https://kosis.kr/covid_eng/covid_index.do; https://www.kaggle.com/kimjihoo/coronavirusdataset.
2. STGCN in traffic: https://github.com/FelixOpolka/STGCN-PyTorch.
3. Seq2Seq in Neural machine translation: https://www.tensorflow.org/tutorials/text/nmt_with_attention.