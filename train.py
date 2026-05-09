"""
Thyroid Disease Prediction - Training Script
Run this in your PyCharm terminal:
    python3 train.py
Make sure dataset.csv is in the same folder!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from xgboost import XGBClassifier
from collections import Counter
import joblib

np.random.seed(42)
print('All imports successful.')

# ── Load & Clean Data ─────────────────────────────────────────────────────────
df = pd.read_csv('dataset.csv')
print('Raw shape:', df.shape)

df = df.drop(columns=['patient_id', 'TBG', 'TBG_measured'], errors='ignore')

def map_target(val):
    val = str(val).strip()
    if val == '-':
        return 'Normal'
    hyper = {'A', 'B', 'C', 'D'}
    hypo  = {'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S'}
    primary = val.split('|')[0][0].upper()
    if primary in hyper:
        return 'Hyperthyroid'
    elif primary in hypo:
        return 'Hypothyroid'
    else:
        return 'Other'

df['diagnosis'] = df['target'].apply(map_target)
df = df[df['diagnosis'] != 'Other'].copy()
df = df.drop(columns=['target'])

print('\nClass distribution:')
print(df['diagnosis'].value_counts())

# ── Preprocessing ─────────────────────────────────────────────────────────────
df['age'] = df['age'].clip(upper=100)

binary_cols = [
    'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_meds', 'sick',
    'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid',
    'query_hyperthyroid', 'lithium', 'goitre', 'tumor',
    'hypopituitary', 'psych',
    'TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured'
]
for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].map({'t': 1, 'f': 0}).fillna(0).astype(int)

df['sex'] = df['sex'].map({'F': 0, 'M': 1})
df['sex'] = df['sex'].fillna(df['sex'].mode()[0]).astype(int)

ref_categories = sorted(df['referral_source'].dropna().unique())
ref_map = {v: i for i, v in enumerate(ref_categories)}
df['referral_source'] = df['referral_source'].map(ref_map).fillna(0).astype(int)

# ── Encode Target ─────────────────────────────────────────────────────────────
le = LabelEncoder()
y  = le.fit_transform(df['diagnosis'])
X  = df.drop(columns=['diagnosis'])

print('\nClasses:', list(enumerate(le.classes_)))
print('X shape:', X.shape)

numerical_cols = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
other_cols     = [c for c in X.columns if c not in numerical_cols]

# ── Train/Test Split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f'\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}')

# ── Preprocessor ─────────────────────────────────────────────────────────────
numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])
other_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))
])
preprocessor = ColumnTransformer([
    ('num',   numerical_transformer, numerical_cols),
    ('other', other_transformer,     other_cols)
])

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_results = {}

def evaluate_model(name, pipeline):
    print(f'\n{"="*60}')
    print(f'  MODEL: {name}')
    print(f'{"="*60}')
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=cv_strategy, scoring='f1_macro', n_jobs=-1
    )
    print(f'  CV F1 scores: {np.round(cv_scores, 3)}')
    print(f'  Mean CV F1  : {cv_scores.mean():.3f} | Std: {cv_scores.std():.3f}')
    pipeline.fit(X_train, y_train)
    y_pred  = pipeline.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, average='macro')
    gap     = cv_scores.mean() - test_f1
    status  = 'GOOD ✓' if gap < 0.05 else ('MILD △' if gap < 0.10 else 'OVERFIT ✗')
    print(f'  Test Macro F1: {test_f1:.3f}')
    print(f'  Gap: {gap:.3f} → {status}')
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    all_results[name] = {
        'pipeline': pipeline, 'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(), 'test_f1': test_f1,
        'gap': gap, 'status': status, 'y_pred': y_pred
    }
    return pipeline

# ── Model 1: SVM ──────────────────────────────────────────────────────────────
svm_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', SVC(C=1.0, kernel='rbf', gamma='scale',
                       class_weight='balanced', probability=True, random_state=42))
])
evaluate_model('SVM (RBF Kernel)', svm_pipeline)

# ── Model 2: XGBoost ─────────────────────────────────────────────────────────
class_counts   = Counter(y_train)
total          = len(y_train)
n_classes      = len(le.classes_)
sample_weights = np.array([total / (n_classes * class_counts[l]) for l in y_train])

xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.5, gamma=0.1,
        objective='multi:softmax', num_class=3,
        eval_metric='mlogloss', use_label_encoder=False,
        random_state=42, n_jobs=-1
    ))
])

print('\n{"="*60}')
print('  MODEL: XGBoost')
print('{"="*60}')
xgb_cv_scores = []
for fold, (tr_idx, val_idx) in enumerate(cv_strategy.split(X_train, y_train), 1):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]
    xgb_pipeline.fit(X_tr, y_tr, classifier__sample_weight=sample_weights[tr_idx])
    fold_f1 = f1_score(y_val, xgb_pipeline.predict(X_val), average='macro')
    xgb_cv_scores.append(fold_f1)
    print(f'  Fold {fold}: F1 = {fold_f1:.3f}')

xgb_cv_scores = np.array(xgb_cv_scores)
xgb_pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weights)
y_pred_xgb  = xgb_pipeline.predict(X_test)
xgb_test_f1 = f1_score(y_test, y_pred_xgb, average='macro')
gap_xgb     = xgb_cv_scores.mean() - xgb_test_f1
status_xgb  = 'GOOD ✓' if gap_xgb < 0.05 else ('MILD △' if gap_xgb < 0.10 else 'OVERFIT ✗')
print(f'  Mean CV F1: {xgb_cv_scores.mean():.3f} | Test F1: {xgb_test_f1:.3f} | Gap: {gap_xgb:.3f} {status_xgb}')
print(classification_report(y_test, y_pred_xgb, target_names=le.classes_))
all_results['XGBoost'] = {
    'pipeline': xgb_pipeline, 'cv_mean': xgb_cv_scores.mean(),
    'cv_std': xgb_cv_scores.std(), 'test_f1': xgb_test_f1,
    'gap': gap_xgb, 'status': status_xgb, 'y_pred': y_pred_xgb
}

# ── Model 3: Stacking ─────────────────────────────────────────────────────────
stacking_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', StackingClassifier(
        estimators=[
            ('xgb', XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                                   reg_alpha=0.1, reg_lambda=1.5, gamma=0.1,
                                   objective='multi:softmax', num_class=3,
                                   eval_metric='mlogloss', use_label_encoder=False,
                                   random_state=42, n_jobs=-1)),
            ('rf',  RandomForestClassifier(n_estimators=100, max_depth=8,
                                            min_samples_leaf=10, max_features='sqrt',
                                            class_weight='balanced', random_state=42, n_jobs=-1)),
            ('svm', SVC(C=1.0, kernel='rbf', gamma='scale',
                        class_weight='balanced', probability=True, random_state=42))
        ],
        final_estimator=LogisticRegression(C=0.5, max_iter=1000,
                                            class_weight='balanced', random_state=42),
        cv=5, passthrough=False, n_jobs=-1
    ))
])
evaluate_model('Stacking (XGB + RF + SVM → LR)', stacking_pipeline)

# ── Model 4: Bagging ──────────────────────────────────────────────────────────
bagging_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=6, min_samples_split=20,
                                          min_samples_leaf=10, class_weight='balanced',
                                          random_state=42),
        n_estimators=100, max_samples=0.8, max_features=0.8,
        bootstrap=True, bootstrap_features=True,
        oob_score=True, random_state=42, n_jobs=-1
    ))
])
evaluate_model('Bagging (100 Decision Trees)', bagging_pipeline)

# ── Results Summary ───────────────────────────────────────────────────────────
print(f'\n{"="*72}')
print('  FINAL RESULTS')
print(f'{"="*72}')
print(f'  {"Model":<35} {"CV F1":>8} {"Test F1":>9} {"Gap":>7} {"Status":>10}')
print(f'  {"-"*70}')
for name, res in all_results.items():
    print(f'  {name:<35} {res["cv_mean"]:>8.3f} {res["test_f1"]:>9.3f} {res["gap"]:>7.3f} {res["status"]:>10}')

# ── Save Best Model ───────────────────────────────────────────────────────────
best_name = max(all_results, key=lambda n: all_results[n]['test_f1'])
best      = all_results[best_name]
print(f'\nBest model: {best_name} | Test F1: {best["test_f1"]:.3f}')

joblib.dump(best['pipeline'], 'thyroidmodel.pkl')
joblib.dump(le,               'labelencoder.pkl')
joblib.dump(ref_map,          'referral_source_map.pkl')

print('\n✅ Saved: thyroidmodel.pkl, labelencoder.pkl, referral_source_map.pkl')
print('Now run: streamlit run myapp.py')

# ── ALL PLOTS ─────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ── Fig 1: Class Distribution ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
classes = ['Normal', 'Hypothyroid', 'Hyperthyroid']
counts  = [sum(y == i) for i, c in enumerate(le.classes_) for cc in [c] if cc in classes or True]
counts  = dict(zip(le.classes_, [sum(y == i) for i in range(len(le.classes_))]))
colors  = ['#185FA5', '#0F6E56', '#D85A30']
ax.bar(counts.keys(), counts.values(), color=colors, zorder=3)
for i, (k, v) in enumerate(counts.items()):
    ax.text(i, v + 30, str(v), ha='center', fontsize=11)
ax.set_title('Fig 1. Class Distribution', fontsize=12)
ax.set_ylabel('Count')
ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig('fig1_class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print('✅ fig1_class_distribution.png saved')

# ── Fig 2: Missing Values ─────────────────────────────────────────────────────
hormone_cols = ['TSH', 'T3', 'TT4', 'T4U', 'FTI']
missing = {col: df[col].isnull().sum() for col in hormone_cols if col in df.columns}
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(missing.keys(), missing.values(), color='#185FA5', zorder=3)
for i, (k, v) in enumerate(missing.items()):
    ax.text(i, v + 5, str(v), ha='center', fontsize=11)
ax.set_title('Fig 2. Missing Value Counts per Hormone Column (before imputation)', fontsize=11)
ax.set_ylabel('Missing Count')
ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig('fig2_missing_values.png', dpi=150, bbox_inches='tight')
plt.show()
print('✅ fig2_missing_values.png saved')

# ── Fig 3: XGBoost Feature Importance ─────────────────────────────────────────
xgb_model   = all_results['XGBoost']['pipeline'].named_steps['classifier']
feat_names  = (numerical_cols +
               all_results['XGBoost']['pipeline']
               .named_steps['preprocessor']
               .transformers_[1][2])
importances = xgb_model.feature_importances_
top_idx     = np.argsort(importances)[-15:]

fig, ax = plt.subplots(figsize=(8, 6))
colors_fi = ['#D85A30' if feat_names[i] in ['TSH','T3','TT4','T4U','FTI','age']
             else '#185FA5' for i in top_idx]
ax.barh([feat_names[i] for i in top_idx],
        [importances[i] for i in top_idx],
        color=colors_fi, zorder=3)
ax.set_title('Fig 3. Top 15 XGBoost Feature Importances\n(red = hormone levels)', fontsize=11)
ax.set_xlabel('F-score (importance)')
ax.xaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig('fig3_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print('✅ fig3_feature_importance.png saved')

# ── Fig 4: Train CV F1 vs Test F1 ────────────────────────────────────────────
model_names = list(all_results.keys())
cv_scores   = [all_results[m]['cv_mean'] for m in model_names]
test_scores = [all_results[m]['test_f1'] for m in model_names]
short_names = ['SVM\n(RBF)', 'XGBoost', 'Stacking\n(XGB+RF+SVM→LR)', 'Bagging\n(100 DT)']
x     = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, cv_scores,   width, label='Train CV F1', color='#185FA5', zorder=3)
bars2 = ax.bar(x + width/2, test_scores, width, label='Test F1',     color='#0F6E56', zorder=3)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)
ax.set_ylabel('Macro F1 Score', fontsize=12)
ax.set_title('Fig 4. Train CV F1 vs Test F1 — All Models\n(Similar heights = good generalisation)', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=11)
ax.set_ylim(0.60, 0.95)
ax.legend(fontsize=11)
ax.yaxis.grid(True, linestyle='--', alpha=0.6, zorder=0)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig('fig4_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print('✅ fig4_model_comparison.png saved')

# ── Fig 5: Confusion Matrices ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
plot_names = ['SVM (RBF Kernel)', 'XGBoost',
              'Stacking (XGB + RF + SVM → LR)', 'Bagging (100 Decision Trees)']
short_titles = ['SVM (RBF)', 'XGBoost', 'Stacking (XGB+RF+SVM→LR)', 'Bagging (100 DT)']

for i, name in enumerate(plot_names):
    cm  = confusion_matrix(y_test, all_results[name]['y_pred'])
    gap = all_results[name]['gap']
    tf1 = all_results[name]['test_f1']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=le.classes_)
    disp.plot(ax=axes[i], colorbar=False, cmap='Blues')
    axes[i].set_title(
        f'{short_titles[i]}\nTest Macro F1 = {tf1:.3f} | Gap = {gap:.3f} {all_results[name]["status"]}',
        fontsize=10)

fig.suptitle('Fig 5. Confusion Matrices — All Models on Test Set\n(Diagonal = correctly classified | Off-diagonal = misclassified)',
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig('fig5_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()
print('✅ fig5_confusion_matrices.png saved')

# ── EDA: Hormone Distribution by Class ───────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
eda_df = df.copy()
eda_df['diagnosis'] = le.inverse_transform(y)
colors_eda = {'Normal': '#185FA5', 'Hypothyroid': '#0F6E56', 'Hyperthyroid': '#D85A30'}

for i, col in enumerate(hormone_cols):
    for cls, clr in colors_eda.items():
        subset = eda_df[eda_df['diagnosis'] == cls][col].dropna()
        axes[i].hist(subset, bins=40, alpha=0.6, label=cls, color=clr)
    axes[i].set_title(f'{col} distribution by class', fontsize=11)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')
    axes[i].legend(fontsize=9)

axes[-1].axis('off')
fig.suptitle('EDA: Hormone Level Distributions by Thyroid Class', fontsize=13)
plt.tight_layout()
plt.savefig('eda_hormone_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print('✅ eda_hormone_distributions.png saved')
print('\n✅ ALL PLOTS SAVED!')