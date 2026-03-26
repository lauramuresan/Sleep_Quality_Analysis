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
    parser = argparse.ArgumentParser(
        description='Analiza comparativa KNN vs Decision Tree'
    )
    parser.add_argument(
        '--data', type=str, default='data_output/',
        help='Folder cu datele .npy si feature_names.joblib (default: data_output/)'
    )
    parser.add_argument(
        '--models', type=str, default='results/models/',
        help='Folder cu modelele .joblib antrenate (default: results/models/)'
    )
    parser.add_argument(
        '--out', type=str, default='results/',
        help='Folder output pentru dashboard si JSON (default: results/)'
    )
    parser.add_argument(
        '--no-browser', action='store_true',
        help='Nu deschide browserul automat dupa generare'
    )
    return parser.parse_args()


def check_required_files(data_folder, models_folder):
    """Verifica daca toate fisierele necesare exista."""
    required = {
        data_folder:   ['X_train.npy', 'y_train.npy', 'X_test.npy', 'y_test.npy', 'feature_names.joblib'],
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
    """Afiseaza un rezumat compact in terminal."""
    t = results['tree']
    k = results['knn']
    comp = results['comparison']
    ds = results['dataset']

    sep = '─' * 56
    print(f'\n{sep}')
    print(f'  📊  REZUMAT COMPARATIV — KNN vs Decision Tree')
    print(sep)
    print(f'  Dataset: {ds["train_samples"]:,} train / {ds["test_samples"]:,} test / {ds["n_features"]} features / {ds["n_classes"]} clase')
    print(sep)
    print(f'  {"Metrica":<25} {"Decision Tree":>14}  {"KNN":>10}')
    print(f'  {"-"*25} {"-"*14}  {"-"*10}')

    rows = [
        ('Accuracy (Test)',   t['accuracy_test'],  k['accuracy_test']),
        ('F1 Macro',          t['f1_macro'],        k['f1_macro']),
        ('Precision Macro',   t['precision_macro'], k['precision_macro']),
        ('Recall Macro',      t['recall_macro'],    k['recall_macro']),
        ('AUC-ROC',           t.get('auc_roc') or '-', k.get('auc_roc') or '-'),
        ('Train Accuracy',    t['accuracy_train'],  k['accuracy_train']),
        ('Overfit Gap',       t['overfit_gap'],     k['overfit_gap']),
        ('Predict Time (ms)', t['predict_time_ms'], k['predict_time_ms']),
    ]

    for label, tv, kv in rows:
        try:
            tf = float(tv)
            kf = float(kv)
            t_marker = ' ◀' if tf > kf else '  '
            k_marker = ' ◀' if kf > tf else '  '
            print(f'  {label:<25} {tv:>13.4f}{t_marker}  {kv:>9.4f}{k_marker}')
        except (ValueError, TypeError):
            print(f'  {label:<25} {str(tv):>14}  {str(kv):>10}')

    print(sep)
    tree_wins = sum(1 for v in comp.values() if v == 'tree')
    knn_wins  = sum(1 for v in comp.values() if v == 'knn')
    winner = 'Decision Tree 🌲' if tree_wins >= knn_wins else 'KNN 🔵'
    print(f'  Castigator: {winner}  (Tree: {tree_wins}W / KNN: {knn_wins}W)')
    print(sep)


def main():
    args = parse_args()

    logging.info('=' * 56)
    logging.info('  ML Analysis — KNN vs Decision Tree')
    logging.info('=' * 56)

    # 1. Verificare fisiere
    missing = check_required_files(args.data, args.models)
    if missing:
        logging.error('Fisiere lipsa:')
        for f in missing:
            logging.error(f'  ✗ {f}')
        logging.error('Asigura-te ca ai rulat tree_trainer.py si knn_trainer.py inainte.')
        sys.exit(1)

    # 2. Analiza
    logging.info('Pasul 1/3: Incarcare modele & calcul metrici...')
    analyzer = ModelAnalyzer(
        data_folder=args.data,
        models_folder=args.models
    )
    results = analyzer.run()

    # 3. Salveaza JSON
    json_path = os.path.join(args.out, 'analysis_results.json')
    analyzer.save_json(json_path)

    # 4. Genereaza dashboard HTML
    logging.info('Pasul 2/3: Generare dashboard HTML...')
    report = ReportGenerator(results)
    html_path = report.save(os.path.join(args.out, 'dashboard.html'))

    # 5. Rezumat terminal
    logging.info('Pasul 3/3: Rezumat final...')
    print_summary(results)

    logging.info(f'Dashboard:  {os.path.abspath(html_path)}')
    logging.info(f'JSON:       {os.path.abspath(json_path)}')

    # 6. Deschide browserul
    if not args.no_browser:
        abs_path = os.path.abspath(html_path)
        webbrowser.open(f'file://{abs_path}')
        logging.info('Dashboard deschis in browser.')

    logging.info('Analiza completa!')


if __name__ == '_main_':
    main()