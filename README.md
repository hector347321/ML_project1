# ML Project 1 — Predicting coronary heart disease (MICHD) from BRFSS data


We predict whether a respondent of the Behavioral Risk Factor Surveillance System (BRFSS)
survey has had a myocardial infarction or angina (MICHD), a binary classification task.
The final model is an L2-regularized logistic regression with class weights, trained with
Nesterov-accelerated gradient descent. Everything is implemented with NumPy only.

## Results

| | F1 | Accuracy |
|---|---|---|
| 4-fold stratified cross-validation | `<CV F1>` | `<CV accuracy>` |
| AIcrowd test set | `<AIcrowd F1>` | `<AIcrowd accuracy>` |

## Repository structure

| File | Content |
|---|---|
| `implementations.py` | The six required methods (`mean_squared_error_gd`, `mean_squared_error_sgd`, `least_squares`, `ridge_regression`, `logistic_regression`, `reg_logistic_regression`) and their helper functions (losses, gradients, sigmoid). |
| `helpers.py` | Course-provided functions to load the data (`load_csv_data`) and write the submission file (`create_csv_submission`). |
| `run.py` | Full pipeline: data cleaning, preprocessing, model training, hyperparameter search, ablation study and submission. |
| `README.md` | This file. |

## Setup

Requirements: Python 3 and NumPy (no other library is used).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy
```

Download the data from the
[AIcrowd challenge page](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1)
and place the three files, with their original names, in a `data/` folder at the root of the
repository:

```
data/
├── x_train.csv
├── y_train.csv
└── x_test.csv
```

The data is not included in this repository (it is listed in `.gitignore`).

## Reproducing the submission

```bash
python run.py
```

This loads the data, preprocesses it, trains the final model on the whole training set with
the selected hyperparameters, and writes the predictions to `submission.csv`. This file is the
submission we uploaded to AIcrowd.

Loading the CSV files can take a few minutes, during which nothing is printed.

### Other modes

The options at the top of `run.py` control what the script does:

| Option | Effect |
|---|---|
| `TUNE = True` | Re-runs the hyperparameter search (grid over λ and the positive-class weight, evaluated by 4-fold stratified CV), prints the results and uses the best values. Slower. |
| `ABLATION = True` | Compares the preprocessing variants by 4-fold stratified CV and prints the table used in the report. Does not write a submission. |
| `QUICK = True` | Uses 1 training row out of 50, for fast debugging only. |

The default configuration (`TUNE = False`, `ABLATION = False`, `QUICK = False`) regenerates our
final submission.

## Method

1. **Cleaning of special codes.** Survey codes meaning "don't know" or "refused" (e.g. 7/9,
   77/99, 777/999) are replaced by missing values, and "none" codes (88) by 0. The rules are
   decided on the training set only and applied identically to the test set.
2. **Preprocessing** (fitted on the training set only):
   - columns with more than 80 % missing values or with no variance are dropped;
   - missing values are imputed with the column median;
   - numerical features are standardized;
   - columns with few distinct values (3 to `MAX_LEVELS`) are additionally one-hot encoded;
   - optionally, a 0/1 indicator is added for each column with at least 5 % missing values.
3. **Model.** Logistic regression with L2 regularization (the bias is not regularized) and a
   weight on the positive class to handle class imbalance. It is trained with
   Nesterov-accelerated gradient descent, with a step size of 1/L, where L is the Lipschitz
   constant of the gradient, estimated by power iteration.
4. **Model selection.** The regularization strength λ, the positive-class weight and the
   decision threshold are chosen to maximize the F1 score in 4-fold stratified
   cross-validation. Cleaning and preprocessing are re-fitted inside each fold, so no
   information leaks from the validation fold.

Predictions are mapped back to {−1, 1} for the submission.
