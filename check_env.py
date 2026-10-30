import sys

import keras
import matplotlib
import numpy
import pandas
import sklearn
import tensorflow as tf

print("Python:", sys.version)
print("NumPy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("scikit-learn:", sklearn.__version__)
print("Matplotlib:", matplotlib.__version__)
print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)

# Check that the dataset can be read
df = pandas.read_csv("data/banknote_authentication.csv")
print("Dataset shape:", df.shape)
print(df.head())
print("Class counts:")
print(df["class"].value_counts())
