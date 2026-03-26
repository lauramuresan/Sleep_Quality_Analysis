import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os
import joblib


class GamingDataPreprocessor:
    def __init__(self, input_path, output_folder):
        self.input_path = input_path
        self.output_folder = output_folder

        # Inițializăm transformările
        self.num_imputer = SimpleImputer(strategy='median')
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def run_pipeline(self):
        # 1. Încărcare date
        df = pd.read_csv(self.input_path)

        # 2. Curățare inițială
        if 'record_id' in df.columns:
            df = df.drop(columns=['record_id'])

        # 3. Separare Target (y) de Features (X)
        if 'sleep_quality' not in df.columns:
            raise ValueError("Coloana 'sleep_quality' lipsește din setul de date!")

        X = df.drop(columns=['sleep_quality'])
        y = df['sleep_quality']

        # 4. Label Encoding pentru Target (Fit pe tot, deoarece clasele sunt fixe)
        y_encoded = self.label_encoder.fit_transform(y.astype(str))

        # 5. Split (80/20) - FACEM SPLIT-UL ÎNAINTE DE ORICE PROCESARE DE VALORI
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # 6. Identificare coloane pe tipuri
        num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X_train.select_dtypes(include=['object', 'bool']).columns.tolist()

        # 7. Imputare
        # Fit pe TRAIN, Transform pe TRAIN și TEST
        if num_cols:
            X_train[num_cols] = self.num_imputer.fit_transform(X_train[num_cols])
            X_test[num_cols] = self.num_imputer.transform(X_test[num_cols])

        if cat_cols:
            X_train[cat_cols] = self.cat_imputer.fit_transform(X_train[cat_cols])
            X_test[cat_cols] = self.cat_imputer.transform(X_test[cat_cols])

        # 8. One-Hot Encoding (Features categorice)
        X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
        X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)

        # Aliniem coloanele din Test cu cele din Train (în caz că lipsesc categorii în test)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        # 9. Scalare
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 10. Salvare Rezultate
        # Date procesate
        np.save(os.path.join(self.output_folder, 'X_train.npy'), X_train_scaled)
        np.save(os.path.join(self.output_folder, 'X_test.npy'), X_test_scaled)
        np.save(os.path.join(self.output_folder, 'y_train.npy'), y_train)
        np.save(os.path.join(self.output_folder, 'y_test.npy'), y_test)

        joblib.dump(self.scaler, os.path.join(self.output_folder, 'scaler.joblib'))
        joblib.dump(self.label_encoder, os.path.join(self.output_folder, 'label_encoder.joblib'))
        joblib.dump(X_train.columns.tolist(), os.path.join(self.output_folder, 'feature_names.joblib'))

        joblib.dump(self.num_imputer, os.path.join(self.output_folder, 'num_imputer.joblib'))

        print(f"✅ Preprocesare completă!")
        print(f"Dimensiuni: Train {X_train_scaled.shape}, Test {X_test_scaled.shape}")