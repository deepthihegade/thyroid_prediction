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