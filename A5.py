"""A4. Generate test set data with values of X & Y varying between 0 and 10 with increments of 0.1. 
This creates a test set of about 10,000 points. Classify these points with above training data using 
kNN classifier (k = 3). Make a scatter plot of the test data output with test points colored as per their 
predicted class colors (all points predicted class0 are labeled blue color). Observe the color spread 
and class boundary lines in the feature space. 
A5. Repeat A4 exercise for various values of k and observe the change in the class boundary lines."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Step 1: Generate the test set (same as A4)
x_min, x_max = 0, 10
y_min, y_max = 0, 10
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
test_set = np.c_[xx.ravel(), yy.ravel()]  # Create 10,000 test points

# Step 2: Load the training data (replace with your actual data)
X_train = np.random.uniform(1, 10, size=(20, 2))  # 20 random training points (replace with actual)
y_train = np.random.choice([0, 1], size=20)  # Random class labels (replace with actual)
 
# Step 3: Create subplots for different values of k
fig, axes = plt.subplots(3, 2, figsize=(12, 12))  # Create 3x2 grid of subplots
k_values = [1, 3, 5, 7, 9]  # k values to test

# Step 4: Train and plot for different values of k
for i, k in enumerate(k_values):
    ax = axes[i // 2, i % 2]  # Get the current subplot
    
    # Step 4.1: Train the kNN model with the current value of k
    kNN = KNeighborsClassifier(n_neighbors=k)
    kNN.fit(X_train, y_train)

    # Step 4.2: Predict for the test set
    y_pred = kNN.predict(test_set)
    
    # Step 4.3: Reshape the predicted labels to match the grid shape
    zz = y_pred.reshape(xx.shape)
    
    # Step 4.4: Plot the decision boundary
    ax.contourf(xx, yy, zz, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor='k', cmap=plt.cm.coolwarm, marker='o', s=100)
    
    # Step 4.5: Title and labels for each subplot
    ax.set_title(f'kNN Decision Boundary (k={k})')
    ax.set_xlabel('X Feature')
    ax.set_ylabel('Y Feature')
    ax.set_xticks(np.arange(0, 11, 1))
    ax.set_yticks(np.arange(0, 11, 1))

# Show the complete plot with subplots for different k values
plt.tight_layout()
plt.show()
