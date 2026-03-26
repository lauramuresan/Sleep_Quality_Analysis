import numpy as np
import os
import joblib
import pandas as pd
import logging

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TreeTrainer:
    def __init__(self, data_folder, results_folder):
        self.data_folder = data_folder
        self.results_folder = results_folder
        self.best_model = None

        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)

    def load_data(self):
        X_train = np.load(os.path.join(self.data_folder, 'X_train.npy'))
        y_train = np.load(os.path.join(self.data_folder, 'y_train.npy'))
        X_test = np.load(os.path.join(self.data_folder, 'X_test.npy'))
        y_test = np.load(os.path.join(self.data_folder, 'y_test.npy'))
        feature_names = joblib.load(os.path.join(self.data_folder, 'feature_names.joblib'))
        return X_train, y_train.ravel(), X_test, y_test.ravel(), feature_names

    def train(self):
        X_train, y_train, X_test, y_test, feature_names = self.load_data()
        logging.info(f"Antrenare Decision Tree pe {X_train.shape[1]} variabile...")

        dtree = DecisionTreeClassifier(random_state=42)
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [3, 5, 7, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'sqrt', 'log2'],
            'class_weight': [None, 'balanced']
        }

        search = GridSearchCV(dtree, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
        search.fit(X_train, y_train)
        self.best_model = search.best_estimator_

        y_pred = self.best_model.predict(X_test)
        logging.info("Evaluare pe test set:")
        print(classification_report(y_test, y_pred))

        importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': self.best_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        joblib.dump(self.best_model, os.path.join(self.results_folder, 'tree_model.joblib'))
        joblib.dump(search.best_params_, os.path.join(self.results_folder, 'tree_best_params.joblib'))
        importance.to_csv(os.path.join(self.results_folder, 'feature_importance.csv'), index=False)

        tree_rules = export_text(self.best_model, feature_names=list(feature_names), max_depth=3)
        with open(os.path.join(self.results_folder, "tree_logic.txt"), "w") as f:
            f.write(tree_rules)

        logging.info(f"Best Params: {search.best_params_}")
        logging.info(f"Top Predictor: {importance.iloc[0]['Feature']}")
        return self.best_model

if __name__ == "__main__":
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(ROOT_DIR, 'data', 'processed')
    models_dir = os.path.join(ROOT_DIR, 'Models', 'saved_models')

    trainer = TreeTrainer(data_folder=data_dir, results_folder=models_dir)
    try:
        best_tree = trainer.train()
    except Exception as e:
        logging.error(f"Eroare la antrenare: {e}")