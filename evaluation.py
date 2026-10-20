import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt


# Assigns the target and feature columns names
columns = ["label", "family", "domain"]

# Import data from Training and Test Sets
dga = pd.read_csv("dga_domains_full.csv", header=None, names=columns)

# removing string labels
dga["label"] = dga["label"].str.strip().str.lower()

# Create binary target values - 0 (legit) and 1 (dga)s
dga["label"] = (dga["label"] == "dga").astype(int)

print(dga.head())

