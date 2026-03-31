import numpy as np
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report

class RandomForestTrainer:
    def __init__(self, data_folder, results_folder):
        self.data_folder = data_folder
        self.results_folder = results_folder
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)

    def train(self):
        # Încărcăm datele preprocesate
        X_train = np.load(os.path.join(self.data_folder, 'X_train.npy'))
        y_train = np.load(os.path.join(self.data_folder, 'y_train.npy'))
        X_test = np.load(os.path.join(self.data_folder, 'X_test.npy'))
        y_test = np.load(os.path.join(self.data_folder, 'y_test.npy'))
        feature_names = joblib.load(os.path.join(self.data_folder, 'feature_names.joblib'))



        rf = RandomForestClassifier(random_state=42)

        param_dist = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', 'balanced_subsample', None]
        }

        search = RandomizedSearchCV(
            rf, param_distributions=param_dist,
            n_iter=30, cv=5, scoring='accuracy',
            n_jobs=-1, random_state=42
        )

        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        print(f"\n=========================================")
        print(f"🏆 Accuracy pe test set (Random Forest): {acc * 100:.2f}%")
        print(f"=========================================\n")

        importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': best_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)



        joblib.dump(best_model, os.path.join(self.results_folder, 'rf_model.joblib'))
        joblib.dump(search.best_params_, os.path.join(self.results_folder, 'rf_best_params.joblib'))

        return best_model

if __name__ == "__main__":
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    trainer = RandomForestTrainer(
        os.path.join(ROOT_DIR, 'data', 'processed'),
        os.path.join(ROOT_DIR, 'Models', 'saved_models')
    )
    trainer.train()