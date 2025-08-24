"""A7. Use RandomSearchCV() or GridSearchCV() operations to find the ideal ‘k’ value for your 
kNN classifier. This is called hyper-parameter tuning. """

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the data (adjust the path as needed)
data = np.load(r"C:\Users\Vasanth Kumar\Desktop\features_pca.npz", allow_pickle=True)
X, y = data['Xp'], data['y']

# Debug: Check the shape of X and y
print("X shape:", X.shape)
print("y shape:", y.shape)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Step 1: Define the kNN model
knn = KNeighborsClassifier()

# Step 2: Specify hyperparameter grid (for GridSearchCV)
param_grid = {
    'n_neighbors': np.arange(1, 5),  # Try k values from 1 to 4 (for testing)
    'weights': ['uniform'],  # Try only 'uniform' for simplicity
    'metric': ['euclidean']  # Use only 'euclidean' for simplicity
}

# Step 3: Use GridSearchCV for tuning the kNN hyperparameters
grid_search = GridSearchCV(knn, param_grid, cv=5, n_jobs=1, verbose=3)  # Added verbose=3 for debugging
grid_search.fit(X_train, y_train)

# Check if the GridSearchCV finished and print the best hyperparameters
if grid_search.best_params_:
    print("Best hyperparameters from GridSearchCV:")
    print(grid_search.best_params_)
else:
    print("No valid parameters found.")

# Step 4: Use the best k value from GridSearchCV to evaluate accuracy
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Test accuracy with best k: {accuracy * 100:.2f}%")
