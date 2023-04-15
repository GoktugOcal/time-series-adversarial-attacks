# Effects of Adversarial Attacks on Time-Series Forecasting Tasks

The repository of the project [Effects of Adversarial Attacks on Time-Series Forecasting Tasks](https://github.com/GoktugOcal/time-series-adversarial-attacks/blob/main/paper/Effects%20of%20Adversarial%20Attacks%20on%20Time-Series%20Forecasting%20Tasks%20-%20Draft.pdf) by Göktuğ Öcal for CMPE588 Testing and Verification in Machine Learning course in Boğaziçi University.

## Overview
Since the decision-making process in some businesses highly relies on Deep Neural Networks and there are lots of intelligent systems that use DNNs, the robustness of the DNNs is becoming more and more important every day in nearly every domain. Recent studies have shown that DNNs are vulnerable to perturbated attacks called adversarial attacks, which makes them non-robust. In the research community, the main example of the importance of the robustness of DNNs is the autonomous cars and car crashes. Yet there are other applications of DNNs such as time series forecasting which is very crucial for decision-making in businesses. In this study, I have investigated the effects of adversarial attacks on different DNN models, mostly based on Long Short-Term Memory(LSTM) networks and CNN based forecasting models. The project has couple of results including those topics;

- Adversarial Attacks to Forecasting Models.
- Adversarial Training for Improving the Robustness of Models

## Installation

1- Install required packages.

```shell
$ pip install -r requirements.txt
```
<!-- 
2- Download datasets.
```shell
$ pip download_data.py
```
-->
## Usage

### Model Training

Trains the selected models for selected datasets and saves models and error metrics to a specific directory.

```shell
$ python model_training.py [--setting <"mtm | mto">] [--datasets <"dataset_name_1, dataset_name_2, ...">] [--models <"model_name_1, model_name_2, ...">]
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

# Definitions

## Models

| Model Name           | Structure    |
|----------------------|--------------|
| Single Layer LSTM    | <img src="https://github.com/GoktugOcal/time-series-adversarial-attacks/blob/main/paper/images/single-layer-lstm.jpg?raw=true" width=33%>   |
| Double Layer LSTM    | <img src="https://github.com/GoktugOcal/time-series-adversarial-attacks/blob/main/paper/images/double-layer-lstm.jpg?raw=true" width=33%>   |
| Bidirectional LSTM   | <img src="https://github.com/GoktugOcal/time-series-adversarial-attacks/blob/main/paper/images/bdlstm.jpg?raw=true" width=33%>              |
| CNN1D                | <img src="https://github.com/GoktugOcal/time-series-adversarial-attacks/blob/main/paper/images/1dcnn.jpg?raw=true" width=33%>               |

# Some Experiment Results

## Adversarial Samples

### PGD Attack

> (Many-to-one) Original sample, perturbation, and Adversarial sample generated with **PGD attack** and **LSTM model** for Solar Generation dataset with **input size of 23**:

<div style="background-color:white; color:black">

Original Sample                        |  Perturbation by PGD                        |  Adversarial Sample  |
:-------------------------------------:|:-------------------------------------------:|:--:|
![](https://github.com/GoktugOcal/time-series-adversarial-attacks/blob/main/paper/images/adv/original-24.png?raw=true)  |  ![](https://github.com/GoktugOcal/time-series-adversarial-attacks/blob/main/paper/images/adv/perturbation-24.png?raw=true)  |   ![](https://github.com/GoktugOcal/time-series-adversarial-attacks/blob/main/paper/images/adv/adversarial-sample-24.png?raw=true)  |

</div>


> (Many-to-one) Original sample, perturbation, and Adversarial sample generated with **PGD attack** and **LSTM model** for Electricity Transformer dataset with **input size of 92**:

<div style="background-color:white; color:black">

Original Sample                        |  Perturbation by PGD                        |  Adversarial Sample  |
:-------------------------------------:|:-------------------------------------------:|:--:|
![](https://github.com/GoktugOcal/time-series-adversarial-attacks/blob/main/paper/images/adv/original-92.png?raw=true)  |  ![](https://github.com/GoktugOcal/time-series-adversarial-attacks/blob/main/paper/images/adv/perturbation-92.png?raw=true)  |   ![](https://github.com/GoktugOcal/time-series-adversarial-attacks/blob/main/paper/images/adv/adversarial-sample-92.png?raw=true)  |

</div>

> (Many-to-one) Original sample, perturbation, and Adversarial sample generated with **PGD attack** and **CNN model** for Electricity Transformer dataset with **input size of 92**:

<div style="background-color:white; color:black">

Original Sample                        |  Perturbation by PGD                        |  Adversarial Sample  |
:-------------------------------------:|:-------------------------------------------:|:--:|
![](https://github.com/GoktugOcal/time-series-adversarial-attacks/blob/main/paper/images/adv/original-24-cnn.png?raw=true)  |  ![](https://github.com/GoktugOcal/time-series-adversarial-attacks/blob/main/paper/images/adv/perturbation-24-cnn.png?raw=true)  |   ![](https://github.com/GoktugOcal/time-series-adversarial-attacks/blob/main/paper/images/adv/adversarial-sample-24-cnn.png?raw=true)  |

</div>

> (Many-to-many) Original sample, perturbation, and Adversarial sample generated with **PGD attack** and **LSTM model** for Electricity Transformer dataset with **input size of 168**:

<div style="background-color:white; color:black">

Original Sample                        |  Perturbation by PGD                        |  Adversarial Sample  |
:-------------------------------------:|:-------------------------------------------:|:--:|
![](https://github.com/GoktugOcal/time-series-adversarial-attacks/blob/main/paper/images/adv/mtm_original_24.png?raw=true)  |  ![](https://github.com/GoktugOcal/time-series-adversarial-attacks/blob/main/paper/images/adv/mtm_perturbation_24.png?raw=true)  |   ![](https://github.com/GoktugOcal/time-series-adversarial-attacks/blob/main/paper/images/adv/mtm_adversarial_sample_24.png?raw=true)  |

</div>

### FGSM vs PGD Attack

> (Many-to-one) Comparison of Adversarial Samples generated with FGSM and PGD attacks and LSTM model for Solar Generation dataset with input size of 23

<div style="background-color:white; color:black; width:75%">

![](https://github.com/GoktugOcal/time-series-adversarial-attacks/blob/main/paper/images/adv/fgsm-vs-pgd-24.png?raw=true)

</div>

> (Many-to-one) Comparison of Adversarial Samples generated with FGSM and PGD attacks and LSTM model for Electricity Transformer dataset with input size of 92

<div style="background-color:white; color:black; width:75%">

![](https://github.com/GoktugOcal/time-series-adversarial-attacks/blob/main/paper/images/adv/fgsm-vs-pgd-92.png?raw=true)

</div>

> (Many-to-one) Comparison of Adversarial Samples generated with FGSM and PGD attacks and CNN model for Electricity Transformer dataset with input size of 92:

<div style="background-color:white; color:black; width:75%">

![](https://github.com/GoktugOcal/time-series-adversarial-attacks/blob/main/paper/images/adv/fgsm-vs-pgd-92-cnn.png?raw=true)

</div>

> (Many-to-many) Comparison of Adversarial Samples generated with FGSM and PGD attacks and LSTM model for Electricity Transformer dataset with input size of 168:

<div style="background-color:white; color:black; width:75%">

![](https://github.com/GoktugOcal/time-series-adversarial-attacks/blob/main/paper/images/adv/mtm_fgsm_vs_pgd_168.png?raw=true)

</div>


## Adversarial Training

All the results are the Mean Absolute Errors (MAE) values. Original column refers to forecasting error for the model trained with original data. Attack columns refers to forecasting error for adversarial samples generated with corresponding attacks type and given to the model trained with original data. Training and Test columns are the forecasting errors for the models trained with both original and adversarial samples and both training and test set includes both of the samples.


|                         |                    | Original |        |   FGSM   |        |        |   PGD    |        |
|:-----------------------:|:------------------:|:--------:|:------:|:--------:|:------:|:------:|:--------:|:------:|
|                         |                    |          | Attack | Training |  Test  | Attack | Training |  Test  |
| Electricity Transformer |  Single Layer LSTM |   0,25   |   1,7  |   0,38   |  0,42  |  1,32  |   0,39   |  0,34  |
|                         |  Double Layer LSTM |   0,22   |  1,58  |    0,4   |  0,24  |  1,38  |   0,47   |  0,38  |
|                         | Bidirectional LSTM |   0,23   |  1,56  |   0,33   |  0,27  |  1,32  |   0,44   |   0,4  |
|                         |        CNN1D       |   0,39   |  1,94  |   0,61   |  0,41  |  2,34  |   0,61   |  0,37  |
|      Metro Traffic      |  Single Layer LSTM |  321,11  | 710,75 |  389,55  | 314,98 | 684,65 |  416,73  |  373,3 |
|                         |  Double Layer LSTM |  306,83  | 825,71 |  346,07  |  281,6 | 805,16 |   351,6  | 304,21 |
|                         | Bidirectional LSTM |  305,38  | 723,97 |  380,06  | 313,59 | 697,62 |   370,6  | 311,65 |
|                         |        CNN1D       |  325,54  | 771,21 |  361,46  | 324,38 | 801,82 |  356,77  | 317,47 |
|       Air-Quality       |  Single Layer LSTM |   10,58  |  31,01 |   13,23  |  14,07 |  27,41 |   13,02  |  13,59 |
|                         |  Double Layer LSTM |   10,77  |  32,09 |   12,7   |  12,47 |  25,83 |   12,6   |  13,26 |
|                         | Bidirectional LSTM |   10,79  |  31,33 |   11,81  |  11,56 |  24,48 |   12,28  |  12,69 |
|                         |        CNN1D       |   12,59  |  42,28 |   13,09  |  13,98 |  36,91 |   11,86  |  12,41 |
|     Solar Generation    |  Single Layer LSTM |   8,16   |  33,14 |   7,36   |  8,52  |  32,54 |   7,66   |  9,88  |
|                         |  Double Layer LSTM |   6,12   |  27,58 |   5,42   |  6,27  |  28,08 |   5,61   |  6,66  |
|                         | Bidirectional LSTM |   8,94   |  42,34 |   6,62   |  7,59  |  39,5  |   7,48   |  9,26  |
|                         |        CNN1D       |   7,11   |  28,7  |   7,91   |  9,33  |  28,05 |   6,71   |  7,93  |
