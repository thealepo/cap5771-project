from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# Hyperparameter grids from Part I
hyperparameters = {
    'Decision_Tree': {
        'max_depth': [3, 9, None],
        'min_samples_split': [2, 10, 50],
        'min_samples_leaf': [1, 5, 25],
    },

    'kNN': {
        'algorithm': ['ball_tree', 'kd_tree', 'brute'],
        'p': [1, 2, 3],
        'n_neighbors': [3, 9, 27],
    },

    'Random_Forest': {
        'n_estimators': [50, 100, 200],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [5, 10, None],
    }
}


# Classifier definitions
models = {
    'Decision_Tree': DecisionTreeClassifier(),
    'kNN': KNeighborsClassifier(),
    'Random_Forest': RandomForestClassifier(),
}