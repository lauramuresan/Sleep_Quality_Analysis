import numpy as np
import os
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


class TreeTrainer:
    def __init__(self, data_folder, results_folder):
        self.data_folder = data_folder
        self.results_folder = results_folder
        if not os.path.exists(self.results_folder): os.makedirs(self.results_folder)

    def train(self):
        X_train = np.load(os.path.join(self.data_folder, 'X_train.npy'))
        y_train = np.load(os.path.join(self.data_folder, 'y_train.npy'))
        X_test = np.load(os.path.join(self.data_folder, 'X_test.npy'))
        y_test = np.load(os.path.join(self.data_folder, 'y_test.npy'))
        feature_names = joblib.load(os.path.join(self.data_folder, 'feature_names.joblib'))

        dtree = DecisionTreeClassifier(random_state=42)

        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 6],
            'class_weight': ['balanced', None]
        }

        search = GridSearchCV(dtree, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"\n=====================================")
        print(f"🌲 Accuracy on the test set: {acc * 100:.2f}%")
        print(f"=====================================\n")

        joblib.dump(best_model, os.path.join(self.results_folder, 'tree_model.joblib'))
        joblib.dump(search.best_params_, os.path.join(self.results_folder, 'tree_best_params.joblib'))

        return best_model


if __name__ == "__main__":
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    trainer = TreeTrainer(os.path.join(ROOT_DIR, 'data', 'processed'), os.path.join(ROOT_DIR, 'Models', 'saved_models'))
    trainer.train()