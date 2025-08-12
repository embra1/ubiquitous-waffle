"""Reusable NLP classification pipeline utilities.

This module centralizes data loading, text cleaning, feature generation, vectorization,
model training, evaluation, and experiment logging.
"""
from __future__ import annotations
import re, string, json, time, random, hashlib, datetime as dt, subprocess
from pathlib import Path
from typing import Iterable, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
PUNCT_TABLE = str.maketrans('', '', string.punctuation)

PROJECT_ROOT = Path.cwd()
for parent in [PROJECT_ROOT] + list(PROJECT_ROOT.parents):
    if (parent / 'Data').is_dir():
        PROJECT_ROOT = parent
        break
DATA_DIR = PROJECT_ROOT / 'Data'
PROCESSED_DIR = PROJECT_ROOT / 'processed'
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'
PROCESSED_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)

RAW_SPLIT = 'training data'
COMMENTS_CSV = DATA_DIR / RAW_SPLIT / 'comments.csv'
TASK1_CSV = DATA_DIR / RAW_SPLIT / 'task1.csv'
TASK2_CSV = DATA_DIR / RAW_SPLIT / 'task2.csv'
CACHE_PARQUET = PROCESSED_DIR / 'training_clean.parquet'
CSV_FALLBACK = PROCESSED_DIR / 'training_clean.csv'

# ---------- Text Cleaning ----------

def clean_text(text: Any) -> str:
    if not isinstance(text, str):
        return ''
    t = text.lower()
    t = URL_PATTERN.sub(' ', t)
    t = t.translate(PUNCT_TABLE)
    return re.sub(r'\s+', ' ', t).strip()

# ---------- Feature Engineering ----------

def add_basic_nlp_features(df: pd.DataFrame) -> pd.DataFrame:
    tokens = df['clean_comment'].map(lambda s: s.split())
    df['token_count'] = tokens.map(len)
    df['char_count'] = df['clean_comment'].map(len)
    df['avg_token_len'] = df['char_count'] / df['token_count'].replace(0, np.nan)
    df['unique_ratio'] = tokens.map(lambda ts: len(set(ts)) / len(ts) if ts else 0)
    return df

# ---------- Data Loading ----------

def load_training_dataframe(force_rebuild: bool=False) -> pd.DataFrame:
    if CACHE_PARQUET.exists() and not force_rebuild:
        return pd.read_parquet(CACHE_PARQUET)
    if CSV_FALLBACK.exists() and not force_rebuild:
        return pd.read_csv(CSV_FALLBACK)
    comments = pd.read_csv(COMMENTS_CSV)
    task1 = pd.read_csv(TASK1_CSV)
    df = comments.merge(task1, on=['document','comment_id'], how='left')
    df['clean_comment'] = df['comment'].map(clean_text)
    df = add_basic_nlp_features(df)
    try:
        df.to_parquet(CACHE_PARQUET, index=False)
    except Exception:
        df.to_csv(CSV_FALLBACK, index=False)
    return df

# ---------- Vectorizer / Model ----------

def build_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    return TfidfVectorizer(max_features=max_features, ngram_range=(1,2), min_df=2)


def train_logreg(text: Iterable[str], labels: Iterable[str], max_features: int = 5000):
    vec = build_vectorizer(max_features)
    X = vec.fit_transform(text)
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X, labels)
    return model, vec

# ---------- Evaluation ----------

def evaluate(model, vec, text: Iterable[str], labels: Iterable[str]) -> Dict[str, float]:
    X = vec.transform(text)
    preds = model.predict(X)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1_macro': f1_score(labels, preds, average='macro'),
    }

# ---------- End-to-End Run ----------

def run_pipeline(df: Optional[pd.DataFrame]=None, target_col: str='flausch', test_size: float=0.2, max_features: int=5000) -> Dict[str, Any]:
    df = df if df is not None else load_training_dataframe()
    if target_col not in df.columns:
        # choose first suitable target
        candidates = [c for c in df.columns if c not in ['document','comment_id','comment','clean_comment'] and 1 < df[c].nunique() <= 5]
        if not candidates:
            raise ValueError('No suitable target column found.')
        target_col = candidates[0]
    X = df['clean_comment']
    y = df[target_col].astype(str)
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, stratify=y, test_size=test_size, random_state=RANDOM_SEED)
    model, vec = train_logreg(X_tr, y_tr, max_features=max_features)
    metrics = evaluate(model, vec, X_va, y_va)
    # Persist minimal artifacts
    joblib.dump(vec, ARTIFACTS_DIR / 'tfidf_vectorizer.joblib')
    joblib.dump(model, ARTIFACTS_DIR / 'best_model.joblib')
    return {'metrics': metrics, 'target_col': target_col, 'n_train': len(X_tr), 'n_valid': len(X_va)}

# ---------- Experiment Logging ----------

def current_git_commit() -> str:
    try:
        return subprocess.check_output(['git','rev-parse','HEAD'], cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        return 'UNKNOWN'


def log_experiment(payload: Dict[str, Any], log_file: Path = ARTIFACTS_DIR / 'experiments.log.jsonl') -> None:
    base = {
        'timestamp': dt.datetime.utcnow().isoformat(),
        'git_commit': current_git_commit(),
        'random_seed': RANDOM_SEED,
    }
    base.update(payload)
    with open(log_file, 'a') as f:
        f.write(json.dumps(base) + '\n')

__all__ = [
    'clean_text', 'add_basic_nlp_features', 'load_training_dataframe', 'build_vectorizer',
    'train_logreg', 'evaluate', 'run_pipeline', 'log_experiment',
    'train_task1_model', 'predict_task1', 'load_artifacts', 'predict_task1_dataset'
]

# ---------- Task 1 Convenience (binary classification on 'flausch') ----------

def train_task1_model(max_features: int = 5000, force_rebuild: bool = False, grid_search: bool = False):
    """Train a model for Task 1 (assumes target column 'flausch').

    Parameters
    ----------
    max_features : int
        Max TF-IDF features.
    force_rebuild : bool
        Force rebuild of cached training dataframe.
    grid_search : bool
        Whether to run a small param grid over C values for LogisticRegression.

    Returns
    -------
    dict with keys: model, vectorizer, metrics, target_col
    """
    df = load_training_dataframe(force_rebuild=force_rebuild)
    if 'flausch' not in df.columns:
        raise ValueError("Expected column 'flausch' in training data.")
    df = df.dropna(subset=['flausch'])
    # Ensure no NaN text rows remain
    df = df[df['clean_comment'].notna()].copy()
    X = df['clean_comment'].fillna('')
    y = df['flausch'].astype(str)
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, stratify=y, test_size=0.2, random_state=RANDOM_SEED)
    vec = build_vectorizer(max_features=max_features)
    X_tr_vec = vec.fit_transform(X_tr)
    X_va_vec = vec.transform(X_va)
    base_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model = base_model
    if grid_search:
        from sklearn.model_selection import GridSearchCV
        param_grid = {'C': [0.1, 1.0, 3.0]}
        gs = GridSearchCV(base_model, param_grid, scoring='f1_macro', cv=3, n_jobs=-1)
        gs.fit(X_tr_vec, y_tr)
        model = gs.best_estimator_
    else:
        model.fit(X_tr_vec, y_tr)
    metrics = evaluate(model, vec, X_va, y_va)
    # Persist artifacts
    joblib.dump(vec, ARTIFACTS_DIR / 'tfidf_vectorizer.joblib')
    joblib.dump(model, ARTIFACTS_DIR / 'best_model.joblib')
    log_experiment({'task': 'task1_train', 'metrics': metrics, 'target_col': 'flausch', 'grid_search': grid_search})
    return {'model': model, 'vectorizer': vec, 'metrics': metrics, 'target_col': 'flausch'}


def load_artifacts():
    """Load persisted vectorizer and model (raises FileNotFoundError if missing)."""
    vec_path = ARTIFACTS_DIR / 'tfidf_vectorizer.joblib'
    model_path = ARTIFACTS_DIR / 'best_model.joblib'
    if not vec_path.exists() or not model_path.exists():
        raise FileNotFoundError('Artifacts not found. Train a model first.')
    vec = joblib.load(vec_path)
    model = joblib.load(model_path)
    return model, vec


def predict_task1(comments: Iterable[str], model=None, vectorizer=None, auto_load: bool = True) -> pd.DataFrame:
    """Predict Task 1 labels for a list/iterable of raw comment strings.

    Parameters
    ----------
    comments : Iterable[str]
        Raw comment texts.
    model, vectorizer : optional
        Preloaded artifacts; if None and auto_load True, they'll be loaded from disk.
    auto_load : bool
        Auto-load persisted artifacts if model/vectorizer not provided.

    Returns
    -------
    DataFrame with columns: comment, clean_comment, prediction, proba (if available)
    """
    if model is None or vectorizer is None:
        if not auto_load:
            raise ValueError('Model/vectorizer not supplied and auto_load is False.')
        model, vectorizer = load_artifacts()
    rows = []
    for c in comments:
        cleaned = clean_text(c)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        proba = None
        if hasattr(model, 'predict_proba'):
            try:
                proba = float(model.predict_proba(vec)[0,1])
            except Exception:
                proba = None
        rows.append({'comment': c, 'clean_comment': cleaned, 'prediction': pred, 'proba': proba})
    return pd.DataFrame(rows)


def predict_task1_dataset(split: str = 'training data', out_path: Optional[Path] = None, batch_size: int = 512) -> Path:
    """Run Task1 predictions over an entire split's comments and persist results.

    Parameters
    ----------
    split : str
        One of the directory names under Data (e.g., 'training data').
    out_path : Path, optional
        Where to write the CSV (defaults to artifacts/task1_predictions_<split>.csv)
    batch_size : int
        Batch size (placeholder for future batching; current implementation processes all at once).

    Returns
    -------
    Path to written CSV file.
    """
    split_dir = DATA_DIR / split
    comments_path = split_dir / 'comments.csv'
    if not comments_path.exists():
        raise FileNotFoundError(f'Missing comments file: {comments_path}')
    comments_df = pd.read_csv(comments_path)
    # Ensure artifacts exist
    try:
        model, vec = load_artifacts()
    except FileNotFoundError:
        # Train first if missing
        train_task1_model()
        model, vec = load_artifacts()
    preds_df = predict_task1(comments_df['comment'].tolist(), model=model, vectorizer=vec, auto_load=False)
    merged = comments_df.join(preds_df[['prediction','proba']])
    if out_path is None:
        safe_split = split.replace(' ', '_')
        out_path = ARTIFACTS_DIR / f'task1_predictions_{safe_split}.csv'
    merged.to_csv(out_path, index=False)
    log_experiment({'task': 'task1_batch_predict', 'rows': len(merged), 'split': split})
    return out_path


def _cli():  # pragma: no cover - simple CLI helper
    import argparse, sys
    parser = argparse.ArgumentParser(description='Task 1 model training and prediction CLI')
    sub = parser.add_subparsers(dest='command', required=True)
    p_train = sub.add_parser('train', help='Train Task1 model')
    p_train.add_argument('--max-features', type=int, default=5000)
    p_train.add_argument('--grid', action='store_true', help='Run grid search over C')
    p_train.add_argument('--force-rebuild', action='store_true')
    p_pred = sub.add_parser('predict', help='Predict Task1 labels')
    p_pred.add_argument('input', nargs='*', help='Raw comments; if empty, read from STDIN lines')
    p_pred_ds = sub.add_parser('predict-dataset', help='Predict Task1 labels for a full split and write CSV')
    p_pred_ds.add_argument('--split', default='training data')
    p_pred_ds.add_argument('--out', default=None, help='Output CSV path')
    args = parser.parse_args()
    if args.command == 'train':
        out = train_task1_model(max_features=args.max_features, force_rebuild=args.force_rebuild, grid_search=args.grid)
        print('Metrics:', out['metrics'])
    elif args.command == 'predict':
        texts = args.input or [l.strip() for l in sys.stdin if l.strip()]
        df_pred = predict_task1(texts)
        print(df_pred.to_csv(index=False))
    elif args.command == 'predict-dataset':
        out_path = predict_task1_dataset(split=args.split, out_path=Path(args.out) if args.out else None)
        print(f'Wrote predictions to {out_path}')


if __name__ == '__main__':  # pragma: no cover
    _cli()
