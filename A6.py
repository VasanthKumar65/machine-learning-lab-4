"""A4. Generate test set data with values of X & Y varying between 0 and 10 with increments of 0.1. 
This creates a test set of about 10,000 points. Classify these points with above training data using 
kNN classifier (k = 3). Make a scatter plot of the test data output with test points colored as per their 
predicted class colors (all points predicted class0 are labeled blue color). Observe the color spread 
and class boundary lines in the feature space. 
A5. Repeat A4 exercise for various values of k and observe the change in the class boundary lines.
A6. Repeat the exercises A3 to A5 for your project data considering any two features and classes. """

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Step 1: Load the PCA features from the dataset
data = np.load(r"C:\Users\Vasanth Kumar\Desktop\features_pca.npz", allow_pickle=True)
X, y = data['Xp'], data['y']

# Step 2: Select two features for classification (e.g., Feature 0 and Feature 1)
X_train = X[:1000, :2]  # Use first 20 samples, select first two features
y_train = y[:1000]  # Corresponding class labels for venomous (1) and non-venomous (0)

# Step 3: Generate test set data with values of X & Y varying between 0 and 10 with increments of 0.1
x_min, x_max = 0, 10
y_min, y_max = 0, 10
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
test_set = np.c_[xx.ravel(), yy.ravel()]  # Create the test set (10,000 points)

# Step 4: Train kNN classifier and make predictions on the test set
kNN = KNeighborsClassifier(n_neighbors=3)
kNN.fit(X_train, y_train)

# Step 5: Make predictions for the test set
y_pred = kNN.predict(test_set)

# Step 6: Reshape predictions to match the grid shape for visualization
zz = y_pred.reshape(xx.shape)

# Step 7: Plot decision boundary for k=3
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, zz, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor='k', cmap=plt.cm.coolwarm, marker='o', s=100)
plt.title('kNN Classifier Decision Boundaries (k=3)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar()
plt.show()

# Step 8: Repeat for different values of k
k_values = [1, 3, 5, 7, 9]
fig, axes = plt.subplots(3, 2, figsize=(12, 12))

for i, k in enumerate(k_values):
    ax = axes[i // 2, i % 2]  # Get the current subplot
    kNN = KNeighborsClassifier(n_neighbors=k)
    kNN.fit(X_train, y_train)
    y_pred = kNN.predict(test_set)
    zz = y_pred.reshape(xx.shape)
    
    ax.contourf(xx, yy, zz, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor='k', cmap=plt.cm.coolwarm, marker='o', s=100)
    ax.set_title(f'kNN Decision Boundary (k={k})')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

plt.tight_layout()
plt.show()
