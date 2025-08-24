"""A3. Generate 20 data points (training set data) consisting of 2 features (X & Y) whose values vary 
randomly between 1 & 10. Based on the values, assign these 20 points to 2 different classes (class0 - 
Blue & class1 – Red). Make a scatter plot of the training data and color the points as per their class 
color. Observe the plot."""

import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate 20 data points with random X and Y values between 1 and 10
np.random.seed(42)  # For reproducibility
X = np.random.uniform(1, 10, 20)  # 20 random X values between 1 and 10
Y = np.random.uniform(1, 10, 20)  # 20 random Y values between 1 and 10

# Step 2: Assign class labels (randomly) – class0 = 0 (Blue), class1 = 1 (Red)
class_labels = np.random.choice([0, 1], size=20)

# Step 3: Create the scatter plot
plt.figure(figsize=(6,6))
plt.scatter(X[class_labels == 0], Y[class_labels == 0], color='blue', label='Class 0 (Blue)', s=100)
plt.scatter(X[class_labels == 1], Y[class_labels == 1], color='red', label='Class 1 (Red)', s=100)

# Adding labels and title
plt.title('Scatter Plot of Training Data (A3)', fontsize=14)
plt.xlabel('X Feature', fontsize=12)
plt.ylabel('Y Feature', fontsize=12)
plt.legend()

# Show the plot
plt.show()
