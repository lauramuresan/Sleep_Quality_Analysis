import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os
import joblib


class GamingDataPreprocessor:
    def __init__(self, input_path, output_folder):
        self.input_path = input_path
        self.output_folder = output_folder
        self.num_imputer = SimpleImputer(strategy='median')
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def run_pipeline(self):
        df = pd.read_csv(self.input_path)
        cols_to_drop = ['record_id', 'primary_game']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

        sleep_mapping = {
            'Excellent': 1, 'Good': 1, 'Fair': 1,
            'Poor': 0, 'Very Poor': 0, 'Insomnia': 0
        }

        df['sleep_quality'] = df['sleep_quality'].map(sleep_mapping)
        X = df.drop(columns=['sleep_quality'])
        y = df['sleep_quality'].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        num_cols = X_train.select_dtypes(include=['number']).columns.tolist()
        cat_cols = X_train.select_dtypes(exclude=['number']).columns.tolist()

        if num_cols:
            X_train[num_cols] = self.num_imputer.fit_transform(X_train[num_cols])
            X_test[num_cols] = self.num_imputer.transform(X_test[num_cols])

        if cat_cols:
            X_train[cat_cols] = self.cat_imputer.fit_transform(X_train[cat_cols].astype(object))
            X_test[cat_cols] = self.cat_imputer.transform(X_test[cat_cols].astype(object))

        X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
        X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        np.save(os.path.join(self.output_folder, 'X_train.npy'), X_train_scaled)
        np.save(os.path.join(self.output_folder, 'X_test.npy'), X_test_scaled)
        np.save(os.path.join(self.output_folder, 'y_train.npy'), y_train)
        np.save(os.path.join(self.output_folder, 'y_test.npy'), y_test)

        joblib.dump(self.scaler, os.path.join(self.output_folder, 'scaler.joblib'))
        joblib.dump(X_train.columns.tolist(), os.path.join(self.output_folder, 'feature_names.joblib'))



if __name__ == "__main__":
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    input_file = os.path.join(ROOT_DIR, 'data', 'raw', 'Gaming and Mental Health.csv')
    output_dir = os.path.join(ROOT_DIR, 'data', 'processed')
    preprocessor = GamingDataPreprocessor(input_file, output_dir)
    preprocessor.run_pipeline()