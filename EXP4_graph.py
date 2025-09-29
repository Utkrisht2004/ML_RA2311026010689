# Step 1: Import all necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# --- PART 1: Accuracy on Full Dataset (30 Features) ---

# 1a. Load the full dataset
cancer = load_breast_cancer()
X_full, y = cancer.data, cancer.target

# 1b. Split the full data
# We use the same 'y_train' and 'y_test' for both parts of the analysis
X_full_train, X_full_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.3, random_state=42
)

# 1c. Scale the full feature data
scaler_full = StandardScaler()
X_full_train_scaled = scaler_full.fit_transform(X_full_train)
X_full_test_scaled = scaler_full.transform(X_full_test)

# 1d. Train and evaluate Logistic Regression on all features
log_reg_full = LogisticRegression(random_state=42)
log_reg_full.fit(X_full_train_scaled, y_train)
y_pred_lr_full = log_reg_full.predict(X_full_test_scaled)
accuracy_lr_full = accuracy_score(y_test, y_pred_lr_full)

# 1e. Train and evaluate SVM on all features
svm_clf_full = SVC(kernel='linear', random_state=42)
svm_clf_full.fit(X_full_train_scaled, y_train)
y_pred_svm_full = svm_clf_full.predict(X_full_test_scaled)
accuracy_svm_full = accuracy_score(y_test, y_pred_svm_full)

# --- PART 2: Visualization on Simplified Dataset (2 Features) ---

# 2a. Select only the first two features for visualization
X_viz = cancer.data[:, :2]

# 2b. Split the 2-feature data
# Note: We are splitting X_viz, but the y_train/y_test sets are identical to Part 1
X_viz_train, X_viz_test, _, _ = train_test_split(
    X_viz, y, test_size=0.3, random_state=42
)

# 2c. Scale the 2-feature data
scaler_viz = StandardScaler()
X_viz_train_scaled = scaler_viz.fit_transform(X_viz_train)
X_viz_test_scaled = scaler_viz.transform(X_viz_test)

# 2d. Train new models specifically for visualization
log_reg_viz = LogisticRegression(random_state=42)
log_reg_viz.fit(X_viz_train_scaled, y_train)

svm_clf_viz = SVC(kernel='linear', random_state=42)
svm_clf_viz.fit(X_viz_train_scaled, y_train)

# 2e. Define the plotting function
def plot_decision_boundary(X, y, model, ax, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    ax.set_title(title)
    ax.set_xlabel(cancer.feature_names[0])
    ax.set_ylabel(cancer.feature_names[1])
    ax.legend(handles=scatter.legend_elements()[0], labels=['Malignant', 'Benign'])

# --- Display All Results ---

# Print the accuracy scores from Part 1
print("--- MODEL ACCURACY ON FULL DATASET (30 FEATURES) ---")
print(f"Logistic Regression Accuracy: {accuracy_lr_full:.4f} ({accuracy_lr_full:.2%})")
print(f"Support Vector Machine Accuracy: {accuracy_svm_full:.4f} ({accuracy_svm_full:.2%})")
print("-" * 55)

# Create and show the plots from Part 2
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plot_decision_boundary(X_viz_test_scaled, y_test, log_reg_viz, axes[0], 'Logistic Regression (on 2 features)')
plot_decision_boundary(X_viz_test_scaled, y_test, svm_clf_viz, axes[1], 'SVM (on 2 features)')
plt.tight_layout()
plt.show()