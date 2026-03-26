import numpy as np
import os
import joblib
import logging

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class KNNTrainer:
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
        return X_train, y_train.ravel(), X_test, y_test.ravel()

    def train(self):
        X_train, y_train, X_test, y_test = self.load_data()

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier())
        ])

        param_dist = {
            'knn__n_neighbors': [3, 5, 7, 9, 11, 15, 21],
            'knn__weights': ['uniform', 'distance'],
            'knn__metric': ['minkowski'],
            'knn__p': [1, 2]
        }

        search = RandomizedSearchCV(
            pipeline, param_distributions=param_dist,
            n_iter=20, cv=5, scoring='accuracy',
            verbose=1, n_jobs=-1, random_state=42
        )

        search.fit(X_train, y_train)
        self.best_model = search.best_estimator_

        y_pred = self.best_model.predict(X_test)
        logging.info("Evaluare pe test set:")
        print(classification_report(y_test, y_pred))

        joblib.dump(self.best_model, os.path.join(self.results_folder, 'knn_pipeline.joblib'))
        joblib.dump(search.best_params_, os.path.join(self.results_folder, 'knn_best_params.joblib'))

        logging.info(f"Best Params: {search.best_params_}")
        return self.best_model

if __name__ == "__main__":
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(ROOT_DIR, 'data', 'processed')
    models_dir = os.path.join(ROOT_DIR, 'Models', 'saved_models')

    trainer = KNNTrainer(data_folder=data_dir, results_folder=models_dir)
    try:
        best_knn = trainer.train()
    except Exception as e:
        logging.error(f"Eroare la antrenare: {e}")