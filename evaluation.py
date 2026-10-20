import pandas as pd
import numpy as np
import math
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

## Data Import

# Assigns the target and feature columns names
columns = ["label", "family", "domain"]

# Load the DGA dataset
dga = pd.read_csv("dga_domains_full.csv", header=None, names=columns)

# Clean label text: trim spaces and lowercase
dga["label"] = dga["label"].str.strip().str.lower()

# Create binary target values - 0 (legit) and 1 (dga)
dga["label"] = (dga["label"] == "dga").astype(int)

## Data Preparation
print("Class Balance between Regular (0) and Malicious Domains (1)")

print(dga["label"].value_counts()) 

print("Class Balance Across all Families")

print(dga["family"].value_counts())

print("Rows before removing duplicates:", len(dga))

dga_clean = dga.drop_duplicates()

print("Rows after removing duplicates:", len(dga_clean))

print("Total domain duplicates: ", dga_clean["domain"].duplicated().sum())

# Separate legit vs. malicious domainss
legit = dga_clean[dga_clean["label"] == 0]
dga_only = dga_clean[dga_clean["label"] == 1]

# Split legit domains randomly
legit_train, legit_test = train_test_split(legit, test_size=0.3, random_state=0)

# Randomly choose from the domain family set a shuffle, choosing roughly 8 for testing
families = dga_only["family"].unique()
rng = np.random.default_rng(0)
shuffled = rng.permutation(families)

# First 8 families are held out, the remaining 17 are used for training
unseen_families = shuffled[:8]
seen_families = shuffled[8:]

print("Held-out families:", unseen_families)

dga_unseen = dga_only[dga_only["family"].isin(unseen_families)]
dga_seen = dga_only[dga_only["family"].isin(seen_families)]

# Split malicious/DGA domains

dga_seen_train, dga_seen_test = train_test_split(dga_seen, test_size=0.2, random_state=0, stratify=dga_seen["family"])

train = pd.concat([legit_train, dga_seen_train]) # Represents regular and already seen families

test_seen = pd.concat([legit_test, dga_seen_test]) # Testing of legit vs. seen testing

test_unseen = pd.concat([legit_test, dga_unseen]) # Ultimate test to see how model reacts to unseen data

# Extracting domain name
def get_name(domain):
    domain = domain.lower().strip()
    if domain.startswith("www."):
        domain = domain[4:]   
    parts = domain.split(".")
    return parts[0]

# print(get_name("mzltrack.com"))
# print(get_name("plasticbags.sa.com"))
# print(get_name("cvyh1po636avyrsxebwbkn7.ddns.net"))

# Feature 1 - Domain name length:
def length(name):
    return len(name)

# Feature 2 - Fraction of digits to total letters in domain name:
def digit_ratio(name):
    if len(name) == 0:
        return 0
    digits = sum(c.isdigit() for c in name)
    return digits / len(name)

# Feature 3 - Fraction of vowels to total letters in domain name:
def vowel_ratio(name):
    letters = [c for c in name if c.isalpha()]  
    if len(letters) == 0:
        return 0
    vowels = sum(c in "aeiou" for c in letters)
    return vowels / len(letters)

# Feature 4 - Seeing the frequency of the same character in a domain name:
def entropy(name):
    if len(name) == 0:
        return 0
    counts = Counter(name)   
    total = len(name)
    return -sum((n / total) * math.log2(n / total) for n in counts.values())

# Feature 5 - Determining if there are a more than three consenants in a row in domain name:
def max_consonant_run(name):
    longest = 0
    current = 0
    for c in name:
        if c.isalpha() and c not in "aeiou":   
            current += 1
            longest = max(longest, current)
        else:                                  
            current = 0
    return longest

# Future 6 - Count hyphens in domain name:
def hyphen_count(name):
    return name.count("-")

# Feature 7 - Determining ratio of unique characters to domain name:
def unique_char_ratio(name):
    if len(name) == 0:
        return 0
    return len(set(name)) / len(name)

feature_cols = ["length", "digit_ratio", "vowel_ratio", "entropy",
                "max_consonant_run", "hyphen_count", "unique_char_ratio"]

# Apply features to all three sets of split data:
def add_features(data):
    data = data.copy()
    data["name"] = data["domain"].apply(get_name)
    data["length"] = data["name"].apply(length)
    data["digit_ratio"] = data["name"].apply(digit_ratio)
    data["vowel_ratio"] = data["name"].apply(vowel_ratio)
    data["entropy"] = data["name"].apply(entropy)
    data["max_consonant_run"] = data["name"].apply(max_consonant_run)
    data["hyphen_count"] = data["name"].apply(hyphen_count)
    data["unique_char_ratio"] = data["name"].apply(unique_char_ratio)
    return data

# Add features to the three pre-split data sets
train = add_features(train)
test_seen = add_features(test_seen)
test_unseen = add_features(test_unseen)

# print(train.groupby("label")[["length", "digit_ratio", "vowel_ratio", "entropy"]].mean())

#  Define training and test sets across all three categories:
X_train = train[feature_cols]
y_train = train["label"]

X_test_seen = test_seen[feature_cols]
y_test_seen = test_seen["label"]

X_test_unseen = test_unseen[feature_cols]
y_test_unseen = test_unseen["label"]

# Scale the features to be on a similar scale:
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_seen_scaled = scaler.transform(X_test_seen)
X_test_unseen_scaled = scaler.transform(X_test_unseen)

# Dummy classifier:
majority = DummyClassifier(strategy="most_frequent")
majority.fit(X_train_scaled, y_train)

# Training data on logistic regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)

# Precision, recall, f1-score, and support across seen and unseen family datasets
for name, model in [("Majority", majority), ("Logistic regression", logreg)]:
    print("=====", name, "- seen families =====")
    print(classification_report(y_test_seen, model.predict(X_test_seen_scaled), zero_division=0))
    print("=====", name, "- unseen families =====")
    print(classification_report(y_test_unseen, model.predict(X_test_unseen_scaled), zero_division=0))
    
# Implement LASSO penalty algorithm - Supervised
lasso = LogisticRegressionCV(
    Cs=10,               
    cv=5,                
    penalty="l1",        
    solver="liblinear",  
    scoring="f1",        
    max_iter=1000,
)
lasso.fit(X_train_scaled, y_train)

best_C = lasso.C_[0]
print("Chosen C:", best_C)
print("Chosen lambda:", 1 / best_C)

# Features LASSO retained and dropped
coefs = pd.DataFrame({"feature": feature_cols, "coefficient": lasso.coef_[0]})
print(coefs.sort_values("coefficient"))

# LASSO Results
print("===== LASSO - seen families =====")
print(classification_report(y_test_seen, lasso.predict(X_test_seen_scaled), zero_division=0))
print("===== LASSO - unseen families =====")
print(classification_report(y_test_unseen, lasso.predict(X_test_unseen_scaled), zero_division=0))

# Decision Tree 
tree = DecisionTreeClassifier(max_depth=5, random_state=0)
tree.fit(X_train, y_train)

for name, X_test, y_test in [("seen families", X_test_seen, y_test_seen),
                             ("unseen families", X_test_unseen, y_test_unseen)]:
    print("===== Decision tree -", name, "=====")
    print(classification_report(y_test, tree.predict(X_test), zero_division=0))
    
# Results

# Each model's score for every test domain
lasso_scores = lasso.predict_proba(X_test_unseen_scaled)[:, 1]
tree_scores = tree.predict_proba(X_test_unseen)[:, 1]

# ROC curve
fig, ax = plt.subplots()
RocCurveDisplay.from_predictions(y_test_unseen, lasso_scores, name="LASSO", ax=ax)
RocCurveDisplay.from_predictions(y_test_unseen, tree_scores, name="Decision tree", ax=ax)
ax.set_title("ROC curve (unseen families)")
plt.show()

# Precision recall curve
fig, ax = plt.subplots()
PrecisionRecallDisplay.from_predictions(y_test_unseen, lasso_scores, name="LASSO", ax=ax)
PrecisionRecallDisplay.from_predictions(y_test_unseen, tree_scores, name="Decision tree", ax=ax)
ax.set_title("Precision-recall curve (unseen families)")
plt.show()

# Alert precision from results
def alert_precision(tpr, fpr, base_rate):
    return (tpr * base_rate) / (tpr * base_rate + fpr * (1 - base_rate))

# True Positive Rate (TPR) and False Positive Rate (FPR) from the unseen families results
print("LASSO:", alert_precision(0.62, 0.13, 0.001))
print("Decision tree:", alert_precision(0.41, 0.04, 0.001))