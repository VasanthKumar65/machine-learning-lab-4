"""A1. Please evaluate confusion matrix for your classification problem. From confusion matrix, the 
other performance metrics such as precision, recall and F1-Score measures for both training and test 
data. Based on your observations, infer the models learning outcome (underfit / regularfit / overfit).  """

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Load dataset
data = np.load(r"C:\Users\Vasanth Kumar\Desktop\features_pca.npz", allow_pickle=True)
X, y = data['Xp'], data['y']

# Split into train (70%) and test (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train kNN with k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predictions
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

# Confusion matrices
cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

# Reports
print("=== TRAIN REPORT ===")
print(classification_report(y_train, y_train_pred, digits=4))
print("Confusion Matrix (train):\n", cm_train)

print("\n=== TEST REPORT ===")
print(classification_report(y_test, y_test_pred, digits=4))
print("Confusion Matrix (test):\n", cm_test)


