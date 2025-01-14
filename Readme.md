# DP_Quantiles 

This repository contains code for evaluating different mechanisms using various statistical distributions. The experiments compute and compare the Mean Absolute Error (MAE) of multiple mechanisms under controlled conditions.

## Features

- **Mechanisms Supported**:
  - Exponential Mechanism (`EXP`) - Pure DP
- Centered Exponential Mechanism (`CEXP`) - Pure DP
- Gaussian exponential mechanism (`GEXP`) - Pure DP
- Randomized data Quantile Mechanism (`RQM`) - Approximate DP - Data is perturbed with Gaussian noise
- Unbiased Exponential Mechanism (`UBEXP`) - Pure / Approximate DP (uniform / truncated normal data)
