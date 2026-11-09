"""Central settings of the project.

All numbers that define the experiments live here, so that they can be
changed in one place. The experiment scripts and the library read them
from this module.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "banknote_authentication.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

# Data
FEATURE_COLUMNS = (
    "variance",
    "skewness",
    "curtosis",
    "entropy",
)
TARGET_COLUMN = "class"  # 0 = genuine banknote, 1 = forged banknote.

# Share of the samples that goes to the test set. The split is stratified
# (both sets keep the share of the two classes). The same seed gives the same
# split and the same initial weights, so the results can be reproduced.
TEST_SIZE = 0.3
RANDOM_SEED = 42

# Early stopping: training stops when the training loss (mean squared error)
# has not improved by at least MIN_DELTA for PATIENCE consecutive epochs, or
# after MAX_EPOCHS. The limit is high so that most runs stop by the criterion;
# the slowest settings (a small learning rate with a strict MIN_DELTA) still
# reach it, and such runs are marked as not stopped early in the results.
MAX_EPOCHS = 5000
PATIENCE = 10
MIN_DELTA = 0.0001  # The main tolerance; 1e-6 is used as a stricter variant.

# Training
BATCH_SIZE = 32  # Samples per weight update (mini-batch).
# Batch sizes of the batch study: 1 = online, 32 = mini-batch, None = the whole
# training set per update (full batch).
BATCH_SIZE_STUDY_VALUES = (1, 32, None)
DEFAULT_LEARNING_RATE = 0.01
MOMENTUM = 0.9  # Share of the previous step kept by SGD with momentum.
TIMING_REPEATS = 20  # Repeated measurements of the classification time.

# Learning rates of the learning rate study (log scale, from very small to
# large enough to make the training unstable).
LEARNING_RATE_GRID = (
    0.001,
    0.003,
    0.01,
    0.03,
    0.1,
    0.3,
    1.0,
    3.0,
    10.0,
)
# Repeats with different seeds in the studies, and repeats of a training whose
# time is averaged in the main experiments.
N_REPEATS = 5

# Multilayer perceptron (2 layers, 4-1, tanh)
MLP_HIDDEN_UNITS = 4
MLP_OUTPUT_UNITS = 1
# Pairs (hidden layer, output layer) of activation functions.
MLP_ACTIVATION_SETS = {
    "tanh-tanh": ("tanh", "tanh"),
    "sigmoid-sigmoid": ("sigmoid", "sigmoid"),
    "relu-sigmoid": ("relu", "sigmoid"),
}
MLP_OPTIMIZERS = ("sgd", "sgd_momentum", "rmsprop", "adam")

# Single-layer perceptron: pairs (activation function, learning rule).
SINGLE_LAYER_SETUPS = (
    ("threshold", "rosenblatt"),
    ("linear", "delta"),
    ("sigmoid", "delta"),
    ("tanh", "delta"),
)

# RBF network: numbers of hidden neurons (k-means centers) to try.
RBF_CENTER_COUNTS = (5, 10, 20, 40, 80)
