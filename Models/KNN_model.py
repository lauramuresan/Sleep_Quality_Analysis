import numpy as np
import os
import joblib

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class KNNTrainer:
    def _init_(self, data_folder='data_output/', results_folder='results/models/'):
        self.data_folder = data_folder.rstrip('/')
        self.results_folder = results_folder.rstrip('/')

        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)

    def load_data(self):
        X_train = np.load(os.path.join(self.data_folder, 'X_train.npy'))
        y_train = np.load(os.path.join(self.data_folder, 'y_train.npy'))

        return X_train, y_train.ravel()

    def train(self):
        X_train, y_train = self.load_data()


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
            pipeline,
            param_distributions=param_dist,
            n_iter=20,
            cv=5,
            scoring='accuracy',
            verbose=1,
            n_jobs=-1,
            random_state=42
        )

        search.fit(X_train, y_train)

        self.best_model = search.best_estimator_

        joblib.dump(self.best_model, os.path.join(self.results_folder, 'knn_pipeline.joblib'))
        joblib.dump(search.best_params_, os.path.join(self.results_folder, 'best_params.joblib'))


        return self.best_model


if _name_ == "_main_":
    trainer = KNNTrainer(data_folder='data_output')

    try:
        best_knn = trainer.train()
    except Exception as e:
        print(f":{e}")