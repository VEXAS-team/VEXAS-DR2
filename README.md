# VEXAS-DR2
## VEXAS: VISTA EXtension to Auxiliary Surveys - Data Release 2

Repository contains source code for VEXAS DR2. It includes magnitude imputation model and meta-learning for classification of VEXAS sources.

## Repository structure
```bash
├── imputation      # code for VEXAS magnitude imputation
│
│
├── classification  # code for ensembling classification of VEXAS sources
│
│
└── results         # scores of trained models from ensemble
       ├── DESW
       ├── PSW
       └── SMW
```


## Results

#### Meta model results over single 32 models for each VEXAS sample (DES, PS, SM):
| VEXAS sample | MCC    | Accuracy | Precision | Recall | F1-score |
|------------|--------|----------|-----------|--------|----------|
| DESW | 0.9836 |   0.9916 |    0.9884 | 0.9831 |   0.9857 |
| PSW | 0.9776 |   0.9882 |    0.9833 |  0.977 |   0.9801 |
| SMW | 0.9863 | 0.9927   | 0.9874    | 0.98   | 0.9836   |

Detailed scores for each model from ensemble are given in `results/{DESW,PSW,SMW}/metrics.txt` files.
