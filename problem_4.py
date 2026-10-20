import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# Declare a variable that defines the features for future assortment
columns = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty"
]

# Import data from Training and Test Sets
train = pd.read_csv("KDDTrain+.txt", header=None, names=columns)
test = pd.read_csv("KDDTest+.txt", header=None, names=columns)

# Remove last two target columns
X_train = train.iloc[:, :-2]
X_test = test.iloc[:, :-2]

# removing string labels to ensure set remains 2 X 2
y_train = train["label"].str.strip()
y_test = test["label"].str.strip()

# Create binary target values - 0 and 1 
y_train = (y_train != "normal").astype(int)
y_test = (y_test != "normal").astype(int)

# Pick out categorical features
categorical_features = X_train.iloc[:, [1, 2, 3]].columns

numerical_features = [
    column for column in X_train.columns
    if column not in categorical_features
]

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical",
        (OneHotEncoder(handle_unknown="ignore", sparse_output=True)), # ensuring to one-hot encode categorical features
         categorical_features),

        ("numerical",
         StandardScaler(),
         numerical_features)
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(X_train_processed.shape)

# Penalize Logistic Regression
lambdas = np.logspace(-4, 2, 30)
Cs = 1 / lambdas

# Training logistic regression, using lasso to pentalize unesseary features
lasso = LogisticRegressionCV(
    Cs=Cs,
    cv=5,
    penalty="l1", #ensure q =1
    solver="saga",
    max_iter=500, # reduce iteration count
    n_jobs=-1
)

lasso.fit(X_train_processed, y_train)

best_C = lasso.C_[0]
λ = 1 / best_C

print("The Chosen Lambda Value is:", λ)

# Creating a path for the coefficients
coefficients = []

model = LogisticRegression(
    penalty="l1",
    solver="saga",
    max_iter=1000,
    warm_start=True
)

for C in Cs:
    model.set_params(C=C)
    model.fit(X_train_processed, y_train)
    coefficients.append(model.coef_[0].copy())
    
# Plot the coefficient line 
coefficients = np.array(coefficients)

# Generate a plot graph to represent the coefficients
for j in range(coefficients.shape[1]):
    plt.plot(np.log(lambdas), coefficients[:, j])

plt.axvline(np.log(λ), linestyle="--")

plt.xlabel("log(lambda)")
plt.ylabel("Coefficient")
plt.title("LASSO Coefficient Path")

plt.show()

# Use function to get features out
feature_names = preprocessor.get_feature_names_out()

# coefficients (weights)
lasso_coefficients = lasso.coef_[0]

# identify removed features close to 0
eliminated = feature_names[np.isclose(lasso_coefficients, 0)]

print("\nFeatures eliminated by LASSO:")
for feature in eliminated:
    print(feature)

# identify the top retained features
retained = ~np.isclose(lasso_coefficients, 0)

retained_names = feature_names[retained]
retained_coefficients = lasso_coefficients[retained]

top_indices = np.argsort(
    np.abs(retained_coefficients)
)[-5:][::-1]

#display the top 5 retained features
print("\nTop 5 retained features:")
for i in top_indices:
    print(retained_names[i], ":", retained_coefficients[i])
    
    
# Regular Lasso MOdel
lasso_predictions = lasso.predict(X_test_processed)

# Accuracy reflected
lasso_accuracy = accuracy_score(y_test, lasso_predictions)
lasso_f1 = f1_score(y_test, lasso_predictions)


# Logistic regression without normailzation
unregularized = LogisticRegression(
    penalty=None,
    solver="lbfgs",
    max_iter=1000
)

unregularized.fit(X_train_processed, y_train)

unregularized_predictions = unregularized.predict(X_test_processed)

unregularized_accuracy = accuracy_score(
    y_test, unregularized_predictions
)

unregularized_f1 = f1_score(
    y_test, unregularized_predictions
)

# Count how many features LASSO kept and removed
number_retained = np.sum(~np.isclose(lasso_coefficients, 0))
number_eliminated = np.sum(np.isclose(lasso_coefficients, 0))

print("LASSO Accuracy:", lasso_accuracy)
print("LASSO F1:", lasso_f1)

print("Unregularized Accuracy:", unregularized_accuracy)
print("Unregularized F1:", unregularized_f1)

print("\nFeatures retained:", number_retained)
print("Features eliminated:", number_eliminated)