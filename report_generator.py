import json
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ReportGenerator:
    """
    Genereaza un dashboard HTML interactiv cu rezultatele comparative
    ale modelelor KNN si Decision Tree.
    """

    def __init__(self, results: dict):
        self.r = results
        self.tree = results['tree']
        self.knn  = results['knn']
        self.comp = results['comparison']
        self.ds   = results['dataset']

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _winner_badge(self, metric_key, model_key):
        """Returns HTML badge if this model wins this metric."""
        if self.comp.get(metric_key) == model_key:
            return '<span class="badge-win">WINNER</span>'
        return ''

    def _delta(self, val_tree, val_knn, higher_is_better=True):
        diff = val_tree - val_knn
        if higher_is_better:
            cls = 'pos' if diff > 0 else 'neg' if diff < 0 else 'neutral'
            sign = '+' if diff > 0 else ''
        else:
            cls = 'pos' if diff < 0 else 'neg' if diff > 0 else 'neutral'
            sign = '+' if diff > 0 else ''
        return f'<span class="delta {cls}">{sign}{diff:.4f}</span>'

    def _bar(self, value, color_var, max_val=1.0):
        pct = min(100, (value / max_val) * 100)
        return f'<div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%;background:var({color_var})"></div></div>'

    def _cm_html(self, cm, classes, model_key):
        color = '--tree-color' if model_key == 'tree' else '--knn-color'
        rows = ''
        for i, row in enumerate(cm):
            cells = ''
            for j, val in enumerate(row):
                is_diag = (i == j)
                cells += f'<td class="{"cm-diag" if is_diag else "cm-cell"}">{val}</td>'
            rows += f'<tr><td class="cm-label">{classes[i] if i < len(classes) else i}</td>{cells}</tr>'
        header = '<th></th>' + ''.join(f'<th class="cm-header">P:{c}</th>' for c in (classes if classes else range(len(cm[0]))))
        return f'''
        <table class="cm-table" style="--cm-accent:var({color})">
            <thead><tr>{header}</tr></thead>
            <tbody>{rows}</tbody>
        </table>'''

    def _params_html(self, params):
        items = ''.join(f'<div class="param-item"><span class="param-key">{k}</span><span class="param-val">{v}</span></div>' for k, v in params.items())
        return f'<div class="params-grid">{items}</div>'

    def _feature_bars_html(self, top_features):
        if not top_features:
            return '<p style="color:var(--muted)">N/A</p>'
        max_imp = top_features[0]['importance'] if top_features else 1
        items = ''
        for i, f in enumerate(top_features[:10]):
            pct = (f['importance'] / max_imp * 100) if max_imp > 0 else 0
            items += f'''
            <div class="fi-row">
                <span class="fi-rank">#{i+1}</span>
                <span class="fi-name">{f["feature"]}</span>
                <div class="fi-bar-track"><div class="fi-bar-fill" style="width:{pct:.1f}%"></div></div>
                <span class="fi-val">{f["importance"]:.5f}</span>
            </div>'''
        return f'<div class="fi-list">{items}</div>'

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    def _header_html(self):
        now = datetime.now().strftime('%d %B %Y, %H:%M')
        tree_wins = sum(1 for v in self.comp.values() if v == 'tree')
        knn_wins  = sum(1 for v in self.comp.values() if v == 'knn')
        overall_winner = 'Decision Tree' if tree_wins >= knn_wins else 'KNN'
        overall_color  = 'var(--tree-color)' if tree_wins >= knn_wins else 'var(--knn-color)'
        return f'''
        <header class="hero">
            <div class="hero-bg"></div>
            <div class="hero-content">
                <div class="hero-eyebrow">ML Model Analysis Report</div>
                <h1 class="hero-title">KNN <span class="vs">vs</span> Decision Tree</h1>
                <div class="hero-sub">Comparatie completa · Antrenat {self.ds["train_samples"]:,} samples · Testat {self.ds["test_samples"]:,} samples · {self.ds["n_features"]} features · {self.ds["n_classes"]} clase</div>
                <div class="hero-verdict">
                    Castigator overall: <span style="color:{overall_color};font-weight:800">{overall_winner}</span>
                    <span class="score-chip tree-chip">{tree_wins}W</span><span class="vs-tiny"> vs </span><span class="score-chip knn-chip">{knn_wins}W</span>
                </div>
                <div class="hero-date">{now}</div>
            </div>
        </header>'''

    def _scorecard_html(self):
        t, k = self.tree, self.knn
        metrics = [
            ('Accuracy (Test)', t['accuracy_test'], k['accuracy_test'], 'accuracy', True, '1.0'),
            ('F1 Score (Macro)', t['f1_macro'], k['f1_macro'], 'f1', True, '1.0'),
            ('Precision (Macro)', t['precision_macro'], k['precision_macro'], None, True, '1.0'),
            ('Recall (Macro)', t['recall_macro'], k['recall_macro'], None, True, '1.0'),
            ('AUC-ROC', t.get('auc_roc') or 0, k.get('auc_roc') or 0, 'auc', True, '1.0'),
            ('Predict Time (ms)', t['predict_time_ms'], k['predict_time_ms'], 'speed', False, None),
            ('Train Accuracy', t['accuracy_train'], k['accuracy_train'], None, True, '1.0'),
            ('Overfit Gap', t['overfit_gap'], k['overfit_gap'], 'overfit', False, None),
        ]
        rows = ''
        for label, tv, kv, comp_key, hib, max_v in metrics:
            tw = self._winner_badge(comp_key, 'tree') if comp_key else ''
            kw = self._winner_badge(comp_key, 'knn') if comp_key else ''
            if max_v:
                t_bar = self._bar(float(tv), '--tree-color', float(max_v))
                k_bar = self._bar(float(kv), '--knn-color', float(max_v))
            else:
                mx = max(abs(float(tv)), abs(float(kv)), 0.001)
                t_bar = self._bar(abs(float(tv)), '--tree-color', mx)
                k_bar = self._bar(abs(float(kv)), '--knn-color', mx)
            delta_html = self._delta(float(tv), float(kv), hib)
            rows += f'''
            <tr>
                <td class="metric-label">{label}</td>
                <td class="metric-val tree-val">{tv} {tw}{t_bar}</td>
                <td class="metric-delta">{delta_html}</td>
                <td class="metric-val knn-val">{kv} {kw}{k_bar}</td>
            </tr>'''
        return f'''
        <section class="section">
            <h2 class="section-title">📊 Metrici Comparative</h2>
            <div class="table-wrap">
                <table class="score-table">
                    <thead>
                        <tr>
                            <th>Metrica</th>
                            <th class="tree-header">🌲 Decision Tree</th>
                            <th>Δ Delta</th>
                            <th class="knn-header">🔵 KNN</th>
                        </tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
        </section>'''

    def _confusion_matrices_html(self):
        t_cm = self._cm_html(self.tree['confusion_matrix'], self.tree['classes'], 'tree')
        k_cm = self._cm_html(self.knn['confusion_matrix'],  self.knn['classes'],  'knn')
        return f'''
        <section class="section">
            <h2 class="section-title">🔲 Matrice de Confuzie</h2>
            <div class="two-col">
                <div class="model-card tree-card">
                    <div class="model-card-header tree-header-bg">🌲 Decision Tree</div>
                    <div class="card-body">{t_cm}</div>
                </div>
                <div class="model-card knn-card">
                    <div class="model-card-header knn-header-bg">🔵 KNN</div>
                    <div class="card-body">{k_cm}</div>
                </div>
            </div>
        </section>'''

    def _per_class_html(self):
        t_classes = self.tree['per_class']
        k_classes = self.knn['per_class']
        all_classes = sorted(set(list(t_classes.keys()) + list(k_classes.keys())))
        rows = ''
        for cls in all_classes:
            t = t_classes.get(cls, {})
            k = k_classes.get(cls, {})
            rows += f'''
            <tr>
                <td class="cls-label">Clasa {cls}</td>
                <td>{t.get("precision","—")}</td><td>{t.get("recall","—")}</td><td>{t.get("f1","—")}</td><td>{t.get("support","—")}</td>
                <td class="sep"></td>
                <td>{k.get("precision","—")}</td><td>{k.get("recall","—")}</td><td>{k.get("f1","—")}</td><td>{k.get("support","—")}</td>
            </tr>'''
        return f'''
        <section class="section">
            <h2 class="section-title">📋 Performanta per Clasa</h2>
            <div class="table-wrap">
                <table class="per-class-table">
                    <thead>
                        <tr>
                            <th rowspan="2">Clasa</th>
                            <th colspan="4" class="tree-header">🌲 Decision Tree</th>
                            <th class="sep-header"></th>
                            <th colspan="4" class="knn-header">🔵 KNN</th>
                        </tr>
                        <tr>
                            <th>Prec.</th><th>Recall</th><th>F1</th><th>Support</th>
                            <th class="sep-header"></th>
                            <th>Prec.</th><th>Recall</th><th>F1</th><th>Support</th>
                        </tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
        </section>'''

    def _feature_importance_html(self):
        top = self.tree.get('extras', {}).get('top_features', [])
        fi_html = self._feature_bars_html(top)
        extras = self.tree.get('extras', {})
        return f'''
        <section class="section">
            <h2 class="section-title">🎯 Feature Importance (Decision Tree)</h2>
            <div class="two-col">
                <div class="fi-container">
                    {fi_html}
                </div>
                <div class="tree-info-box">
                    <div class="info-stat"><span class="info-label">Adancime maxima utilizata</span><span class="info-number tree-color">{extras.get("max_depth_used","?")}</span></div>
                    <div class="info-stat"><span class="info-label">Numar frunze</span><span class="info-number tree-color">{extras.get("n_leaves","?")}</span></div>
                    <div class="info-stat"><span class="info-label">Total features</span><span class="info-number">{extras.get("n_features_total","?")}</span></div>
                    <div class="info-note">KNN nu ofera feature importance nativa — distanta dintre puncte influenteaza predictia uniform.</div>
                </div>
            </div>
        </section>'''

    def _best_params_html(self):
        t_p = self._params_html(self.tree.get('best_params', {}))
        k_p = self._params_html(self.knn.get('best_params', {}))
        k_extras = self.knn.get('extras', {})
        knn_note = f'<div class="knn-note">StandardScaler aplicat: <strong>{"DA" if k_extras.get("uses_scaler") else "NU"}</strong> — KNN este sensibil la scala datelor.</div>'
        return f'''
        <section class="section">
            <h2 class="section-title">⚙️ Hiperparametri Optimi (GridSearch / RandomSearch)</h2>
            <div class="two-col">
                <div class="model-card tree-card">
                    <div class="model-card-header tree-header-bg">🌲 Decision Tree — Best Params</div>
                    <div class="card-body">{t_p}</div>
                </div>
                <div class="model-card knn-card">
                    <div class="model-card-header knn-header-bg">🔵 KNN — Best Params</div>
                    <div class="card-body">{k_p}{knn_note}</div>
                </div>
            </div>
        </section>'''

    def _pros_cons_html(self):
        tree_pros = [
            'Interpretabil — poti vedea regulile exacte (tree_logic.txt)',
            'Feature importance nativa',
            'Viteza de predictie foarte buna',
            'Nu necesita normalizarea datelor',
            'Bun cu variabile categorice si relatii neliniare',
        ]
        tree_cons = [
            'Tendinta de overfit fara pruning adecvat',
            'Instabil — schimbari mici in date = arbore diferit',
            'Nu performeaza optim pe date cu multe clase rare',
        ]
        knn_pros = [
            'Simplu conceptual — nicio ipoteza despre distributie',
            'Se adapteaza natural la granita de decizie complexa',
            'Bun cand clasele sunt bine separate in spatiu',
            'Fara antrenare explicita — lazy learner',
        ]
        knn_cons = [
            'Lent la predictie pe seturi mari (O(n) per query)',
            'Necesita scalare obligatorie (StandardScaler inclus)',
            'Sensibil la date cu zgomot si outlieri',
            'Nicio interpretabilitate — black box',
            'Consum mare de memorie (stocheaza toti vecinii)',
        ]
        def items(lst, cls):
            return ''.join(f'<li class="pros-cons-item {cls}">{x}</li>' for x in lst)
        return f'''
        <section class="section">
            <h2 class="section-title">⚖️ Plusuri & Minusuri</h2>
            <div class="two-col">
                <div class="model-card tree-card">
                    <div class="model-card-header tree-header-bg">🌲 Decision Tree</div>
                    <div class="card-body">
                        <h4 class="pros-title">✅ Plusuri</h4>
                        <ul class="pros-cons-list">{items(tree_pros,"pro")}</ul>
                        <h4 class="cons-title">❌ Minusuri</h4>
                        <ul class="pros-cons-list">{items(tree_cons,"con")}</ul>
                    </div>
                </div>
                <div class="model-card knn-card">
                    <div class="model-card-header knn-header-bg">🔵 KNN</div>
                    <div class="card-body">
                        <h4 class="pros-title">✅ Plusuri</h4>
                        <ul class="pros-cons-list">{items(knn_pros,"pro")}</ul>
                        <h4 class="cons-title">❌ Minusuri</h4>
                        <ul class="pros-cons-list">{items(knn_cons,"con")}</ul>
                    </div>
                </div>
            </div>
        </section>'''

    def _recommendation_html(self):
        t, k = self.tree, self.knn
        tree_wins = sum(1 for v in self.comp.values() if v == 'tree')
        knn_wins  = sum(1 for v in self.comp.values() if v == 'knn')
        if tree_wins > knn_wins:
            rec_model = 'Decision Tree'
            rec_color = 'var(--tree-color)'
            rec_icon  = '🌲'
            rec_reason = f'Decision Tree obtine scor superior pe {tree_wins} din {len(self.comp)} metrici. Ofera si interpretabilitate nativa.'
        else:
            rec_model = 'KNN'
            rec_color = 'var(--knn-color)'
            rec_icon  = '🔵'
            rec_reason = f'KNN obtine scor superior pe {knn_wins} din {len(self.comp)} metrici.'
        overfit_warning = ''
        if abs(t['overfit_gap']) > 0.05:
            overfit_warning += f'<div class="warning-box">⚠️ Decision Tree are un gap de overfit de <strong>{t["overfit_gap"]:.4f}</strong> — considera pruning mai agresiv (ccp_alpha, max_depth mai mic).</div>'
        if abs(k['overfit_gap']) > 0.05:
            overfit_warning += f'<div class="warning-box">⚠️ KNN are un gap de overfit de <strong>{k["overfit_gap"]:.4f}</strong> — incearca mai multi vecini (k mai mare).</div>'
        return f'''
        <section class="section recommendation-section">
            <h2 class="section-title">🏆 Recomandare Finala</h2>
            <div class="rec-box" style="border-color:{rec_color}">
                <div class="rec-icon">{rec_icon}</div>
                <div class="rec-content">
                    <div class="rec-model" style="color:{rec_color}">{rec_model}</div>
                    <div class="rec-reason">{rec_reason}</div>
                    <div class="rec-scores">
                        <span class="score-pill" style="background:var(--tree-bg);color:var(--tree-color)">Tree Acc: {t["accuracy_test"]}</span>
                        <span class="score-pill" style="background:var(--knn-bg);color:var(--knn-color)">KNN Acc: {k["accuracy_test"]}</span>
                        <span class="score-pill" style="background:var(--tree-bg);color:var(--tree-color)">Tree F1: {t["f1_macro"]}</span>
                        <span class="score-pill" style="background:var(--knn-bg);color:var(--knn-color)">KNN F1: {k["f1_macro"]}</span>
                    </div>
                </div>
            </div>
            {overfit_warning}
        </section>'''

    # ------------------------------------------------------------------
    # CSS
    # ------------------------------------------------------------------

    def _css(self):
        return '''
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');
        :root {
            --bg: #0d0f14;
            --surface: #13161e;
            --surface2: #1a1e2a;
            --border: #252a38;
            --text: #e8eaf0;
            --muted: #6b7280;
            --tree-color: #34d399;
            --tree-bg: rgba(52,211,153,0.08);
            --tree-border: rgba(52,211,153,0.25);
            --knn-color: #60a5fa;
            --knn-bg: rgba(96,165,250,0.08);
            --knn-border: rgba(96,165,250,0.25);
            --pos: #34d399;
            --neg: #f87171;
            --neutral: #6b7280;
            --font-display: 'Syne', sans-serif;
            --font-mono: 'DM Mono', monospace;
        }
        *, *::before, *::after { box-sizing:border-box; margin:0; padding:0; }
        body { background:var(--bg); color:var(--text); font-family:var(--font-display); min-height:100vh; }
        .container { max-width:1200px; margin:0 auto; padding:0 24px 80px; }

        /* HERO */
        .hero { position:relative; padding:80px 0 60px; text-align:center; overflow:hidden; margin-bottom:16px; }
        .hero-bg { position:absolute; inset:0; background:radial-gradient(ellipse 80% 60% at 50% 0%, rgba(52,211,153,0.08) 0%, transparent 70%), radial-gradient(ellipse 60% 40% at 80% 100%, rgba(96,165,250,0.06) 0%, transparent 60%); pointer-events:none; }
        .hero-content { position:relative; }
        .hero-eyebrow { font-size:11px; font-weight:600; letter-spacing:0.2em; text-transform:uppercase; color:var(--muted); margin-bottom:16px; }
        .hero-title { font-size:clamp(40px,7vw,80px); font-weight:800; line-height:1; letter-spacing:-0.03em; color:var(--text); margin-bottom:20px; }
        .vs { color:var(--muted); font-weight:400; }
        .hero-sub { font-size:14px; color:var(--muted); margin-bottom:24px; font-family:var(--font-mono); }
        .hero-verdict { font-size:18px; font-weight:600; margin-bottom:12px; }
        .hero-date { font-size:12px; color:var(--muted); font-family:var(--font-mono); }
        .score-chip { display:inline-block; padding:2px 10px; border-radius:20px; font-size:13px; font-weight:700; margin:0 3px; }
        .tree-chip { background:var(--tree-bg); color:var(--tree-color); border:1px solid var(--tree-border); }
        .knn-chip  { background:var(--knn-bg);  color:var(--knn-color);  border:1px solid var(--knn-border); }
        .vs-tiny { color:var(--muted); font-size:12px; }

        /* SECTIONS */
        .section { margin-bottom:48px; }
        .section-title { font-size:20px; font-weight:700; margin-bottom:24px; letter-spacing:-0.02em; display:flex; align-items:center; gap:10px; }
        .section-title::after { content:''; flex:1; height:1px; background:var(--border); }

        /* TWO COL */
        .two-col { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
        @media(max-width:768px) { .two-col { grid-template-columns:1fr; } }

        /* MODEL CARDS */
        .model-card { border-radius:12px; overflow:hidden; border:1px solid var(--border); background:var(--surface); }
        .tree-card { border-color:var(--tree-border); }
        .knn-card  { border-color:var(--knn-border); }
        .model-card-header { padding:12px 20px; font-size:14px; font-weight:700; letter-spacing:0.05em; }
        .tree-header-bg { background:var(--tree-bg); color:var(--tree-color); border-bottom:1px solid var(--tree-border); }
        .knn-header-bg  { background:var(--knn-bg);  color:var(--knn-color);  border-bottom:1px solid var(--knn-border); }
        .card-body { padding:20px; }

        /* SCORE TABLE */
        .table-wrap { overflow-x:auto; border-radius:12px; border:1px solid var(--border); }
        .score-table { width:100%; border-collapse:collapse; font-size:14px; }
        .score-table thead tr { background:var(--surface2); }
        .score-table th { padding:12px 16px; text-align:left; font-weight:700; font-size:12px; letter-spacing:0.08em; text-transform:uppercase; color:var(--muted); border-bottom:1px solid var(--border); }
        .score-table td { padding:10px 16px; border-bottom:1px solid var(--border); vertical-align:middle; }
        .score-table tr:last-child td { border-bottom:none; }
        .score-table tr:hover td { background:var(--surface2); }
        .tree-header { color:var(--tree-color) !important; }
        .knn-header  { color:var(--knn-color) !important; }
        .metric-label { font-weight:600; color:var(--text); white-space:nowrap; }
        .metric-val { font-family:var(--font-mono); font-size:13px; }
        .tree-val { color:var(--tree-color); }
        .knn-val  { color:var(--knn-color); }
        .metric-delta { text-align:center; }
        .delta { font-family:var(--font-mono); font-size:12px; font-weight:600; padding:2px 6px; border-radius:4px; }
        .delta.pos { color:var(--pos); background:rgba(52,211,153,0.1); }
        .delta.neg { color:var(--neg); background:rgba(248,113,113,0.1); }
        .delta.neutral { color:var(--neutral); }
        .badge-win { background:linear-gradient(135deg,#f59e0b,#ef4444); color:#fff; font-size:9px; font-weight:800; padding:2px 6px; border-radius:3px; letter-spacing:0.08em; margin-left:6px; vertical-align:middle; }
        .bar-track { height:4px; background:var(--border); border-radius:2px; margin-top:5px; width:100%; min-width:80px; }
        .bar-fill  { height:4px; border-radius:2px; transition:width 0.3s; }

        /* CONFUSION MATRIX */
        .cm-table { border-collapse:collapse; font-family:var(--font-mono); font-size:13px; }
        .cm-header { padding:6px 12px; font-size:11px; color:var(--muted); font-weight:600; text-align:center; }
        .cm-label  { padding:6px 12px; font-size:11px; color:var(--muted); font-weight:600; }
        .cm-cell   { padding:10px 16px; text-align:center; background:var(--surface2); border:1px solid var(--border); }
        .cm-diag   { padding:10px 16px; text-align:center; background:var(--cm-accent,var(--tree-color)); color:#000; font-weight:700; border:1px solid var(--border); }

        /* PER CLASS TABLE */
        .per-class-table { width:100%; border-collapse:collapse; font-size:13px; }
        .per-class-table th { padding:10px 12px; text-align:center; font-size:11px; letter-spacing:0.06em; text-transform:uppercase; color:var(--muted); border-bottom:1px solid var(--border); background:var(--surface2); }
        .per-class-table td { padding:9px 12px; text-align:center; border-bottom:1px solid var(--border); font-family:var(--font-mono); }
        .per-class-table tr:last-child td { border-bottom:none; }
        .cls-label { font-weight:700; color:var(--text); font-family:var(--font-display); }
        .sep, .sep-header { width:12px; background:var(--border); }

        /* FEATURE IMPORTANCE */
        .fi-container { display:flex; flex-direction:column; gap:0; }
        .fi-list { display:flex; flex-direction:column; gap:8px; }
        .fi-row { display:grid; grid-template-columns:28px 1fr 120px 70px; align-items:center; gap:10px; }
        .fi-rank { font-family:var(--font-mono); font-size:11px; color:var(--muted); text-align:right; }
        .fi-name { font-size:13px; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
        .fi-bar-track { height:6px; background:var(--border); border-radius:3px; }
        .fi-bar-fill  { height:6px; border-radius:3px; background:linear-gradient(90deg,var(--tree-color),#059669); }
        .fi-val { font-family:var(--font-mono); font-size:11px; color:var(--muted); text-align:right; }
        .tree-info-box { background:var(--surface2); border-radius:10px; padding:20px; border:1px solid var(--tree-border); display:flex; flex-direction:column; gap:16px; }
        .info-stat { display:flex; justify-content:space-between; align-items:center; padding-bottom:12px; border-bottom:1px solid var(--border); }
        .info-stat:last-of-type { border-bottom:none; }
        .info-label { font-size:13px; color:var(--muted); }
        .info-number { font-family:var(--font-mono); font-size:24px; font-weight:700; }
        .tree-color { color:var(--tree-color); }
        .info-note { font-size:12px; color:var(--muted); line-height:1.6; padding-top:4px; }

        /* PARAMS */
        .params-grid { display:flex; flex-direction:column; gap:8px; }
        .param-item { display:flex; justify-content:space-between; align-items:center; padding:8px 12px; background:var(--surface2); border-radius:6px; border:1px solid var(--border); }
        .param-key { font-family:var(--font-mono); font-size:12px; color:var(--muted); }
        .param-val { font-family:var(--font-mono); font-size:13px; font-weight:600; color:var(--text); }
        .knn-note  { margin-top:16px; padding:12px; background:var(--knn-bg); border:1px solid var(--knn-border); border-radius:8px; font-size:13px; color:var(--knn-color); }

        /* PROS CONS */
        .pros-title, .cons-title { font-size:14px; font-weight:700; margin-bottom:10px; margin-top:16px; }
        .pros-title { color:var(--pos); }
        .cons-title { color:var(--neg); }
        .pros-cons-list { list-style:none; display:flex; flex-direction:column; gap:6px; }
        .pros-cons-item { font-size:13px; padding:8px 12px; border-radius:6px; line-height:1.5; }
        .pro { background:rgba(52,211,153,0.06); border-left:3px solid var(--tree-color); color:var(--text); }
        .con { background:rgba(248,113,113,0.06); border-left:3px solid var(--neg); color:var(--text); }

        /* RECOMMENDATION */
        .rec-box { display:flex; gap:24px; align-items:flex-start; padding:28px; background:var(--surface2); border-radius:14px; border-width:2px; border-style:solid; }
        .rec-icon { font-size:48px; line-height:1; flex-shrink:0; }
        .rec-content { flex:1; }
        .rec-model { font-size:28px; font-weight:800; letter-spacing:-0.02em; margin-bottom:8px; }
        .rec-reason { font-size:15px; color:var(--muted); margin-bottom:16px; line-height:1.6; }
        .rec-scores { display:flex; flex-wrap:wrap; gap:8px; }
        .score-pill { padding:4px 12px; border-radius:20px; font-family:var(--font-mono); font-size:12px; font-weight:600; }
        .warning-box { margin-top:16px; padding:14px 18px; background:rgba(251,191,36,0.07); border:1px solid rgba(251,191,36,0.3); border-radius:8px; font-size:14px; color:#fbbf24; line-height:1.6; }
        '''

    # ------------------------------------------------------------------
    # Full HTML
    # ------------------------------------------------------------------

    def generate_html(self):
        data_json = json.dumps(self.r, indent=2)
        return f'''<!DOCTYPE html>
<html lang="ro">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Analysis — KNN vs Decision Tree</title>
    <style>{self._css()}</style>
</head>
<body>
    {self._header_html()}
    <div class="container">
        {self._scorecard_html()}
        {self._per_class_html()}
        {self._confusion_matrices_html()}
        {self._feature_importance_html()}
        {self._best_params_html()}
        {self._pros_cons_html()}
        {self._recommendation_html()}
    </div>
    <script>
    // Animate bars on load
    window.addEventListener('load', () => {{
        document.querySelectorAll('.bar-fill, .fi-bar-fill').forEach((el, i) => {{
            const w = el.style.width;
            el.style.width = '0';
            setTimeout(() => {{ el.style.width = w; }}, 100 + i * 30);
        }});
    }});
    </script>
</body>
</html>'''

    def save(self, path='results/dashboard.html'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        html = self.generate_html()
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
        logging.info(f"Dashboard salvat: {path}")
        return path