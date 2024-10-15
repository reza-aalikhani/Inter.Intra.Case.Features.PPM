# Inter.Intra.Case.Features.PPM
# Repository Overview

This repository contains supplementary material for the article titled **"Enhancing Remaining Time Prediction in Business Processes by Considering System-Level and Resource-Level Inter-Case Features"** by Reza Aalikhani, Mohammad Fathian, and Mohammad Reza Rasouli.

## Purpose

This repository provides implementations of various models for remaining time prediction using a Predictive Process Monitoring approach. The experiments conducted aim to evaluate the impact of inter-case features on the accuracy of time predictions.

## Methodology

To enhance time prediction accuracy, we incorporate inter-case features—such as resource multitasking, open cases, and other metrics related to resource behavior—in addition to intra-case features. The primary objective of these techniques is to predict the completion time of cases while minimizing the Mean Absolute Error (MAE).

## Benchmark Details

The benchmark includes implementations of five sequence encodings (for further details, refer to the paper):

- **Last State Encoding**
- **Aggregation Encoding**
- **Previous Encoding**
- **Combined Encoding**
- **Index-Based Encoding**

Additionally, the repository features four bucketing methods (see the paper for more information):

- **No Bucketing**
- **State-Based Bucketing**
- **Clustering**
- **Prefix Length Based**

The benchmark experiments utilize four regressors:

- **Random Forest**
- **Gradient Boosted Trees (XGBoost)**
- **Decision Tree**
- **Support Vector Regressor (SVR)**
## Datasets
In addition to the code, we provide eight datasets used in the evaluation section of the paper. These datasets correspond to predictions formulated on three publicly available event logs: [BPIC 2011](https://data.4tu.nl/datasets/5ea5bb88-feaa-4e6f-a743-6460a755e05b/1), [BPIC 2015](https://data.4tu.nl/datasets/64fce6ea-5ca8-403b-aa09-82b53517af8a/1), and [BPIC 2017](https://data.4tu.nl/datasets/5c9717a0-4c22-4b78-a3ad-d2234208bfd7/1). The preprocessed benchmark datasets can be accessed [here]([https://drive.google.com/open?id=154hcH-HGThlcZJW5zBvCJMZvjOQDsnPR](https://drive.google.com/file/d/1fDBDdAx_GYfpJAqDQxYGp1qgjQu4FwOf/view?usp=sharing)).
