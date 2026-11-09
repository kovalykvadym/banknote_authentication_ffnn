# Feed forward neural network on the Banknote Authentication dataset

The project contains a single-layer perceptron, multilayer perceptron (2 layers,
4-1 neurons, tanh) and an RBF network for comparison. Everything is implemented
from scratch with NumPy.

Dataset: [Banknote Authentication](https://archive.ics.uci.edu/dataset/267)
(1372 banknotes, 4 features: variance, skewness, curtosis, entropy;
class 0 = genuine, 1 = forged). The data is split 70/30 (stratified,
seed 42) into 960 training and 412 test samples. The features are
standardized with the mean and deviation of the training set only.

## Setup

Python 3.12. From the project root:

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

The dataset must be in `data/banknote_authentication.csv`.

## Project structure

    ffnn_lab/               the library
        activations.py      activation functions and derivatives
        config.py           all settings in one place
        data.py             loading, validation, split, scaling
        evaluation.py       error and classification time of a model
        mlp.py              multilayer perceptron with backpropagation
        optimizers.py       SGD, SGD with momentum, RMSprop, Adam
        perceptron.py       single-layer perceptron (Rosenblatt and delta rules)
        rbf.py              RBF network (k-means + least squares)
        runner.py           train, evaluate and save one experiment
        storage.py          saving a run to disk
        utils.py            timer, error, training result
    build_summary.py        summary table (CSV and Excel)
    check_env.py            prints the Python and package versions
    check_gradients.py      numerical check of the backpropagation gradients
    explore_data.py         first look at the dataset
    plot_results.py         all graphs of the report
    run_batch_study.py      batch size study (group E)
    run_lr_study.py         learning rate study (group D)
    run_mlp.py              experiment group B
    run_rbf.py              experiment group C
    run_single_layer.py     experiment group A

## How to reproduce the results

Run the scripts from the project root in this order:

    python check_gradients.py       # optional: gradients must match numerically
    python run_single_layer.py      # about 10 seconds
    python run_mlp.py               # about 5 minutes
    python run_rbf.py               # a few seconds
    python build_summary.py         # needs the three groups above
    python run_lr_study.py          # about 5 minutes
    python run_batch_study.py       # a few minutes
    python plot_results.py          # redraws all graphs in seconds

The graphs are in Ukrainian by default. Set `LANGUAGE = "en"` in
`plot_results.py` for English.

## Results

Everything is written to `results/`:

    results/runs/<run_id>/      one folder per run: config.json, metrics.json,
                                initial_params.npz, trained_params.npz,
                                predictions_train.csv, predictions_test.csv
    results/summary.csv         all runs, English column names
    results/summary.xlsx        the report table, Ukrainian column names
    results/lr_study*.csv       learning rate study
    results/batch_study*.csv    batch size study
    results/figures/            graphs

## Notes

* The training time is the time until the stopping criterion fires (the
  loss has stopped improving), not the time to the exact minimum.
* The results are reproducible with the same seeds; the training time
  differs a little from run to run, so it is averaged over repeats.
* The dataset contains 24 duplicate rows, so a few test samples may have a
  twin in the training set.