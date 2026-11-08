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
TARGET_COLUMN = "class"

TEST_SIZE = 0.3
RANDOM_SEED = 42

# Early stopping: training stops when the training error has not improved
# by at least MIN_DELTA for PATIENCE consecutive epochs, or after MAX_EPOCHS.
# The limit is high enough that slow settings (small learning rate, sigmoid)
# stop by themselves instead of being cut off.
MAX_EPOCHS = 5000
PATIENCE = 10
MIN_DELTA = 0.0001

# Training
BATCH_SIZE = 32
BATCH_SIZE_STUDY_VALUES = (1, 32, None)
DEFAULT_LEARNING_RATE = 0.01
MOMENTUM = 0.9
TIMING_REPEATS = 20

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
N_REPEATS = 5

# Multilayer perceptron (2 layers, 4-1, tanh)
MLP_HIDDEN_UNITS = 4
MLP_OUTPUT_UNITS = 1
MLP_ACTIVATION_SETS = {
    "tanh-tanh": ("tanh", "tanh"),
    "sigmoid-sigmoid": ("sigmoid", "sigmoid"),
    "relu-sigmoid": ("relu", "sigmoid"),
}
MLP_OPTIMIZERS = ("sgd", "sgd_momentum", "rmsprop", "adam")

# Single-layer perceptron
SINGLE_LAYER_SETUPS = (
    ("threshold", "rosenblatt"),
    ("linear", "delta"),
    ("sigmoid", "delta"),
    ("tanh", "delta"),
)

# RBF network
RBF_CENTER_COUNTS = (5, 10, 20, 40, 80)
