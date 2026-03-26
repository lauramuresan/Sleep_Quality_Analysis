import argparse
import logging
import os
import sys
import webbrowser

from analyzer import ModelAnalyzer
from report_generator import ReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)


def parse_args():
    ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser(description='Analiza comparativa KNN vs Decision Tree')

    # Rutele standard bazate pe structura ta
    parser.add_argument('--data', type=str, default=os.path.join(ROOT_DIR, 'data', 'processed'),
                        help='Folder date procesate')
    parser.add_argument('--models', type=str, default=os.path.join(ROOT_DIR, 'Models', 'saved_models'),
                        help='Folder modele antrenate')
    parser.add_argument('--out', type=str, default=os.path.join(ROOT_DIR, 'reports'), help='Folder output rapoarte')
    parser.add_argument('--no-browser', action='store_true', help='Nu deschide browserul')

    return parser.parse_args()


def check_required_files(data_folder, models_folder):
    required = {
        data_folder: ['X_train.npy', 'y_train.npy', 'X_test.npy', 'y_test.npy', 'feature_names.joblib'],
        models_folder: ['tree_model.joblib', 'knn_pipeline.joblib', 'tree_best_params.joblib', 'knn_best_params.joblib']
    }
    missing = []
    for folder, files in required.items():
        for f in files:
            path = os.path.join(folder, f)
            if not os.path.exists(path):
                missing.append(path)
    return missing


def print_summary(results):
    t, k, comp, ds = results['tree'], results['knn'], results['comparison'], results['dataset']
    sep = '─' * 56
    print(f'\n{sep}\n  📊  REZUMAT COMPARATIV — KNN vs Decision Tree\n{sep}')
    print(
        f'  Dataset: {ds["train_samples"]:,} train / {ds["test_samples"]:,} test / {ds["n_features"]} features / {ds["n_classes"]} clase\n{sep}')
    print(f'  {"Metrica":<25} {"Decision Tree":>14}  {"KNN":>10}\n  {"-" * 25} {"-" * 14}  {"-" * 10}')

    rows = [
        ('Accuracy (Test)', t['accuracy_test'], k['accuracy_test']),
        ('F1 Macro', t['f1_macro'], k['f1_macro']),
        ('Precision Macro', t['precision_macro'], k['precision_macro']),
        ('Recall Macro', t['recall_macro'], k['recall_macro']),
        ('Train Accuracy', t['accuracy_train'], k['accuracy_train']),
        ('Overfit Gap', t['overfit_gap'], k['overfit_gap']),
        ('Predict Time (ms)', t['predict_time_ms'], k['predict_time_ms']),
    ]

    for label, tv, kv in rows:
        try:
            tf, kf = float(tv), float(kv)
            t_marker = ' ◀' if tf > kf else '  '
            k_marker = ' ◀' if kf > tf else '  '
            if 'Overfit' in label or 'Time' in label:  # Mai mic = mai bine
                t_marker = ' ◀' if tf < kf else '  '
                k_marker = ' ◀' if kf < tf else '  '
            print(f'  {label:<25} {tv:>13.4f}{t_marker}  {kv:>9.4f}{k_marker}')
        except:
            print(f'  {label:<25} {str(tv):>14}  {str(kv):>10}')

    tree_wins = sum(1 for v in comp.values() if v == 'tree')
    knn_wins = sum(1 for v in comp.values() if v == 'knn')
    winner = 'Decision Tree 🌲' if tree_wins >= knn_wins else 'KNN 🔵'
    print(f'{sep}\n  Castigator: {winner}  (Tree: {tree_wins}W / KNN: {knn_wins}W)\n{sep}')


def main():
    args = parse_args()

    missing = check_required_files(args.data, args.models)
    if missing:
        logging.error('Fisiere lipsa:')
        for f in missing: logging.error(f'  ✗ {f}')
        logging.error('Ruleaza intai data_cleaner.py, apoi decisionTree_model.py si KNN_model.py')
        sys.exit(1)

    logging.info('Pasul 1/3: Incarcare modele & calcul metrici...')
    analyzer = ModelAnalyzer(data_folder=args.data, models_folder=args.models)
    results = analyzer.run()
    analyzer.save_json(os.path.join(args.out, 'analysis_results.json'))

    logging.info('Pasul 2/3: Generare dashboard HTML...')
    report = ReportGenerator(results)
    html_path = report.save(os.path.join(args.out, 'dashboard.html'))

    logging.info('Pasul 3/3: Rezumat final...')
    print_summary(results)

    if not args.no_browser:
        webbrowser.open(f'file://{os.path.abspath(html_path)}')


if __name__ == '__main__':
    main()