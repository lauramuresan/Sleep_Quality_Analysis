import numpy as np
import joblib
import os
import json
import logging
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ModelAnalyzer:
    """
    Incarca modelele antrenate si calculeaza statistici complete
    pentru comparatia KNN vs Decision Tree.
    """

    def __init__(self, data_folder='data_output/', models_folder='results/models/'):
        self.data_folder = data_folder.rstrip('/')
        self.models_folder = models_folder.rstrip('/')
        self.results = {}

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self):
        logging.info("Se incarca datele de test...")
        X_train = np.load(os.path.join(self.data_folder, 'X_train.npy'))
        y_train = np.load(os.path.join(self.data_folder, 'y_train.npy')).ravel()
        X_test  = np.load(os.path.join(self.data_folder, 'X_test.npy'))
        y_test  = np.load(os.path.join(self.data_folder, 'y_test.npy')).ravel()
        feature_names = joblib.load(os.path.join(self.data_folder, 'feature_names.joblib'))
        return X_train, y_train, X_test, y_test, feature_names

    def load_models(self):
        logging.info("Se incarca modelele...")
        tree_model = joblib.load(os.path.join(self.models_folder, 'tree_model.joblib'))
        knn_model  = joblib.load(os.path.join(self.models_folder, 'knn_pipeline.joblib'))
        tree_params = joblib.load(os.path.join(self.models_folder, 'tree_best_params.joblib'))
        knn_params  = joblib.load(os.path.join(self.models_folder, 'knn_best_params.joblib'))
        return tree_model, knn_model, tree_params, knn_params

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def evaluate_model(self, model, X_train, y_train, X_test, y_test, model_name):
        logging.info(f"Evaluare: {model_name}")

        # Predict time
        start = time.perf_counter()
        y_pred = model.predict(X_test)
        predict_time_ms = (time.perf_counter() - start) * 1000

        # Train time (refit on train for timing only)
        start = time.perf_counter()
        _ = model.predict(X_train)
        train_predict_time_ms = (time.perf_counter() - start) * 1000

        # Probabilities for AUC (if available)
        auc = None
        classes = np.unique(y_test)
        try:
            if len(classes) == 2:
                y_proba = model.predict_proba(X_test)[:, 1]
                auc = round(roc_auc_score(y_test, y_proba), 4)
            else:
                y_proba = model.predict_proba(X_test)
                auc = round(roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro'), 4)
        except Exception:
            pass

        acc_train = round(accuracy_score(y_train, model.predict(X_train)), 4)
        acc_test  = round(accuracy_score(y_test, y_pred), 4)
        overfit   = round(acc_train - acc_test, 4)

        report_dict = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred).tolist()

        per_class = {}
        for cls in [str(c) for c in classes]:
            if cls in report_dict:
                per_class[cls] = {
                    'precision': round(report_dict[cls]['precision'], 4),
                    'recall':    round(report_dict[cls]['recall'], 4),
                    'f1':        round(report_dict[cls]['f1-score'], 4),
                    'support':   int(report_dict[cls]['support'])
                }

        return {
            'model_name': model_name,
            'accuracy_train': acc_train,
            'accuracy_test':  acc_test,
            'overfit_gap':    overfit,
            'precision_macro': round(precision_score(y_test, y_pred, average='macro', zero_division=0), 4),
            'recall_macro':    round(recall_score(y_test, y_pred, average='macro', zero_division=0), 4),
            'f1_macro':        round(f1_score(y_test, y_pred, average='macro', zero_division=0), 4),
            'auc_roc':         auc,
            'predict_time_ms': round(predict_time_ms, 3),
            'confusion_matrix': cm,
            'per_class': per_class,
            'classes': [str(c) for c in classes]
        }

    # ------------------------------------------------------------------
    # Tree-specific extras
    # ------------------------------------------------------------------

    def tree_extras(self, tree_model, feature_names):
        estimator = tree_model  # already the DecisionTreeClassifier
        fi = estimator.feature_importances_
        top_n = min(10, len(feature_names))
        indices = np.argsort(fi)[::-1][:top_n]
        top_features = [
            {'feature': feature_names[i], 'importance': round(float(fi[i]), 5)}
            for i in indices
        ]
        return {
            'max_depth_used':  estimator.get_depth(),
            'n_leaves':        estimator.get_n_leaves(),
            'n_features_total': len(feature_names),
            'top_features': top_features
        }

    # ------------------------------------------------------------------
    # KNN-specific extras
    # ------------------------------------------------------------------

    def knn_extras(self, knn_pipeline, knn_params):
        return {
            'n_neighbors': knn_params.get('knn__n_neighbors', '?'),
            'weights':     knn_params.get('knn__weights', '?'),
            'metric':      knn_params.get('knn__metric', '?'),
            'p':           knn_params.get('knn__p', '?'),
            'uses_scaler': True
        }

    # ------------------------------------------------------------------
    # Run full analysis
    # ------------------------------------------------------------------

    def run(self):
        X_train, y_train, X_test, y_test, feature_names = self.load_data()
        tree_model, knn_model, tree_params, knn_params = self.load_models()

        tree_metrics = self.evaluate_model(tree_model, X_train, y_train, X_test, y_test, 'Decision Tree')
        knn_metrics  = self.evaluate_model(knn_model,  X_train, y_train, X_test, y_test, 'KNN')

        tree_metrics['extras'] = self.tree_extras(tree_model, list(feature_names))
        tree_metrics['best_params'] = tree_params

        knn_metrics['extras'] = self.knn_extras(knn_model, knn_params)
        knn_metrics['best_params'] = knn_params

        # Winner per metric
        comparison = {
            'accuracy':  'tree' if tree_metrics['accuracy_test'] >= knn_metrics['accuracy_test'] else 'knn',
            'f1':        'tree' if tree_metrics['f1_macro'] >= knn_metrics['f1_macro'] else 'knn',
            'speed':     'tree' if tree_metrics['predict_time_ms'] <= knn_metrics['predict_time_ms'] else 'knn',
            'overfit':   'tree' if abs(tree_metrics['overfit_gap']) <= abs(knn_metrics['overfit_gap']) else 'knn',
        }
        if tree_metrics['auc_roc'] and knn_metrics['auc_roc']:
            comparison['auc'] = 'tree' if tree_metrics['auc_roc'] >= knn_metrics['auc_roc'] else 'knn'

        self.results = {
            'tree': tree_metrics,
            'knn':  knn_metrics,
            'comparison': comparison,
            'dataset': {
                'train_samples': int(X_train.shape[0]),
                'test_samples':  int(X_test.shape[0]),
                'n_features':    int(X_train.shape[1]),
                'n_classes':     int(len(np.unique(y_train)))
            }
        }

        logging.info("Analiza completa!")
        return self.results

    def save_json(self, path='results/analysis_results.json'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        logging.info(f"Rezultate salvate in: {path}")