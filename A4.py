"""A4. Generate test set data with values of X & Y varying between 0 and 10 with increments of 0.1. 
This creates a test set of about 10,000 points. Classify these points with above training data using 
kNN classifier (k = 3). Make a scatter plot of the test data output with test points colored as per their 
predicted class colors (all points predicted class0 are labeled blue color). Observe the color spread 
and class boundary lines in the feature space. """

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Step 1: Generate the test set
x_min, x_max = 0, 10
y_min, y_max = 0, 10
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
test_set = np.c_[xx.ravel(), yy.ravel()]  # Create 10,000 test points

# Step 2: Load the training data (from previous step A3)
# Assuming you have already trained your kNN classifier on some training data (class_labels, X_train, y_train)
# Example (you should replace this with the actual kNN training logic):
X_train = np.random.uniform(1, 10, size=(20, 2))  # 20 random training points (replace with actual)
y_train = np.random.choice([0, 1], size=20)  # Random class labels (replace with actual)
kNN = KNeighborsClassifier(n_neighbors=3)
kNN.fit(X_train, y_train)

# Step 3: Predict classes for test set
y_pred = kNN.predict(test_set)  # Predictions for all test points

# Step 4: Plot the results
# Reshape the predicted values to match the grid shape (xx, yy)
zz = y_pred.reshape(xx.shape)

# Plot the decision boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, zz, cmap=plt.cm.coolwarm, alpha=0.8)  # Class boundary visualization
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor='k', cmap=plt.cm.coolwarm, marker='o', s=100)  # Training points
plt.title('kNN Classifier Decision Boundaries (k=3)')
plt.xlabel('X feature')
plt.ylabel('Y feature')
plt.colorbar()  # Show color scale
plt.show()
