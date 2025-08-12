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
    'train_logreg', 'evaluate', 'run_pipeline', 'log_experiment'
]
