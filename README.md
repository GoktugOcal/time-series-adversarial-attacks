# Effects of Adversarial Attacks on Time-Series Forecasting Tasks

The repository of the project [Effects of Adversarial Attacks on Time-Series Forecasting Tasks](https://github.com/GoktugOcal/time-series-adversarial-attacks) by Göktuğ Öcal for CMPE588 Testing and Verification in Machine Learning course in Boğaziçi University.

## Overview
Since the decision-making process in some businesses highly relies on Deep Neural Networks and there are lots of intelligent systems that use DNNs, the robustness of the DNNs is becoming more and more important every day in nearly every domain. Recent studies have shown that DNNs are vulnerable to perturbated attacks called adversarial attacks, which makes them non-robust. In the research community, the main example of the importance of the robustness of DNNs is the autonomous cars and car crashes. Yet there are other applications of DNNs such as time series forecasting which is very crucial for decision-making in businesses. In this study, I have investigated the effects of adversarial attacks on different DNN models, mostly based on Long Short-Term Memory(LSTM) networks and CNN based forecasting models. The project has couple of results including those topics;

- Adversarial Attacks to Forecasting Models.
- Adversarial Training for Improving the Robustness of Models

## Installation

1- Install required packages.

```shell
$ pip install -r requirements.txt
```

2- Download datasets.
```shell
$ pip download_data.py
```

## Usage

### Model Training

Trains the selected models for selected datasets and saves models and error metrics to a specific directory.

```shell
$ python model_training.py [--datasets <"dataset_name_1, dataset_name_2, ...">] [--models <"model_name_1, model_name_2, ...">]
```

### Adversarial Attacks

Attacks to the models in a specific directory where the model training step saves at.

```shell
$ python attacking_to_models.py [--setting <"mto, mtm">] [--attack <"FGSM, PGD">]
```

### Adversarial Training

Trains the models after generating adversarial samples that is added to training and testing tests. Saves the trained models with their error metrics.

```shell
$ python adversarial_training.py [--datasets <"dataset_name_1, dataset_name_2, ...">] [--models <"model_name_1, model_name_2, ...">] [--attack <"FGSM, PGD">]
```




