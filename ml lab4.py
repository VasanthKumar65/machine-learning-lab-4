import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# === STEP 1: Load CSV & Prepare Dataset ===
csv_path = "sample_0_venom_nonvenom.csv"
image_folder = "images"

# Load the dataset
df = pd.read_csv(csv_path)

# Filter rows where image exists
df["image_exists"] = df["uuid_image"].apply(lambda x: os.path.exists(os.path.join(image_folder, x)))
df = df[df["image_exists"]]

# OPTIONAL: Show dataset summary
print("Dataset loaded successfully:")
print(df[['uuid_image', 'poisonous']].head())

# === STEP 2: Load True & Predicted Labels ===
# NOTE: poisonous is the true label (0 = non-venomous, 1 = venomous)

y_true = df["poisonous"].values

# Dummy predictions (for demo) — replace with real model predictions
np.random.seed(42)
y_pred = np.random.randint(0, 2, size=len(y_true))

# === STEP 3: A1 – Confusion Matrix & Classification Report ===

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=["Non-Venomous", "Venomous"]))

cm = confusion_matrix(y_true, y_pred)

# Plot Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Non-Venomous", "Venomous"],
            yticklabels=["Non-Venomous", "Venomous"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

#A3

# === A3 ===
# Generate 20 random points with 2 features (X, Y) between 1 and 10
import matplotlib.pyplot as plt
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Generate random features
X = np.random.uniform(1, 10, (20, 2))  # shape (20, 2)

# Assign random class labels (0 or 1)
y = np.random.randint(0, 2, 20)

# Create a scatter plot
plt.figure(figsize=(8, 6))

for i in range(20):
    if y[i] == 0:
        plt.scatter(X[i, 0], X[i, 1], color='blue', label='Class 0 (Blue)' if i == 0 else "")
    else:
        plt.scatter(X[i, 0], X[i, 1], color='red', label='Class 1 (Red)' if i == 0 else "")

plt.title("A3: Scatter Plot of 20 Random Points (Class 0 - Blue, Class 1 - Red)")
plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.legend()
plt.grid(True)
plt.show()

# === A4 ===
from sklearn.neighbors import KNeighborsClassifier

# Reuse the X and y from A3 (20 points)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Generate test set with X and Y from 0 to 10 with 0.1 increments
x_test_vals = np.arange(0, 10.1, 0.1)
y_test_vals = np.arange(0, 10.1, 0.1)
xx, yy = np.meshgrid(x_test_vals, y_test_vals)
test_points = np.c_[xx.ravel(), yy.ravel()]  # shape ~ (10,000, 2)

# Predict the class for each test point
test_preds = knn.predict(test_points)

# Plot the decision boundary
plt.figure(figsize=(10, 8))
plt.scatter(test_points[:, 0], test_points[:, 1], c=test_preds, cmap='bwr', alpha=0.3, s=5)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k', s=80, label="Training Points")
plt.title("A4: k-NN Classification with k=3 (Blue=Class 0, Red=Class 1)")
plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.legend()
plt.grid(True)
plt.show()

# === A5 ===
for k in [1, 3, 5, 7, 11]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    test_preds = knn.predict(test_points)
   
    plt.figure(figsize=(10, 8))
    plt.scatter(test_points[:, 0], test_points[:, 1], c=test_preds, cmap='bwr', alpha=0.3, s=5)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k', s=80, label="Training Points")
    plt.title(f"A5: k-NN with k={k}")
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.legend()
    plt.grid(True)
    plt.show()


# === A6 ===
# Simulate 2 numeric features from real dataset
df_real = pd.read_csv('sample_0_venom_nonvenom.csv')
df_real = df_real[['uuid_image', 'poisonous']].dropna()
df_real['poisonous'] = df_real['poisonous'].astype(int)

# Generate two fake features between 1 and 10 (real datasets would use CNN embeddings or image stats)
np.random.seed(42)
df_real['feature1'] = np.random.uniform(1, 10, len(df_real))
df_real['feature2'] = np.random.uniform(1, 10, len(df_real))

# Select 20 training points
train_real = df_real.sample(20, random_state=42)
X_real = train_real[['feature1', 'feature2']].values
y_real = train_real['poisonous'].values

# Train KNN
knn_real = KNeighborsClassifier(n_neighbors=3)
knn_real.fit(X_real, y_real)

# Test grid
test_preds_real = knn_real.predict(test_points)

# Plot decision boundary
plt.figure(figsize=(10, 8))
plt.scatter(test_points[:, 0], test_points[:, 1], c=test_preds_real, cmap='bwr', alpha=0.3, s=5)
plt.scatter(X_real[:, 0], X_real[:, 1], c=y_real, cmap='bwr', edgecolor='k', s=80, label="Snake Training Points")
plt.title("A6: k-NN on Real Snake Data (Simulated Features)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()


# === A7 ===
from sklearn.model_selection import GridSearchCV

# Search for best k in range 1 to 15
param_grid = {'n_neighbors': list(range(1, 16))}

grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=3, scoring='f1')
grid.fit(X_real, y_real)

print("Best k:", grid.best_params_['n_neighbors'])
print("Best F1-Score:", grid.best_score_)