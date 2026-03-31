import numpy as np
import os
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


class KNNTrainer:
    def __init__(self, data_folder, results_folder):
        self.data_folder = data_folder
        self.results_folder = results_folder
        if not os.path.exists(self.results_folder): os.makedirs(self.results_folder)

    def train(self):
        X_train = np.load(os.path.join(self.data_folder, 'X_train.npy'))
        y_train = np.load(os.path.join(self.data_folder, 'y_train.npy'))
        X_test = np.load(os.path.join(self.data_folder, 'X_test.npy'))
        y_test = np.load(os.path.join(self.data_folder, 'y_test.npy'))

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier())
        ])

        param_grid = {
            'knn__n_neighbors': [5, 7, 9, 11, 15, 21, 25, 31, 41],
            'knn__weights': ['uniform', 'distance'],
            'knn__metric': ['euclidean', 'manhattan']
        }

        search = GridSearchCV(
            pipeline, param_grid=param_grid,
            cv=5, scoring='accuracy', n_jobs=-1
        )

        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"\n=====================================")
        print(f"🔵 Accuracy pe test set (KNN): {acc * 100:.2f}%")
        print(f"=====================================\n")

        joblib.dump(best_model, os.path.join(self.results_folder, 'knn_pipeline.joblib'))
        joblib.dump(search.best_params_, os.path.join(self.results_folder, 'knn_best_params.joblib'))

        return best_model


if __name__ == "__main__":
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    trainer = KNNTrainer(os.path.join(ROOT_DIR, 'data', 'processed'), os.path.join(ROOT_DIR, 'Models', 'saved_models'))
    trainer.train()