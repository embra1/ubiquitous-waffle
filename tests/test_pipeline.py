import pandas as pd
from pathlib import Path
import importlib

# Import reusable functions from pipeline module
pipeline = importlib.import_module('pipeline')
clean_text = pipeline.clean_text
add_basic_nlp_features = pipeline.add_basic_nlp_features
run_pipeline = pipeline.run_pipeline


def test_clean_text_basic():
    assert clean_text('Hello, WORLD!!!') == 'hello world'
    assert clean_text(None) == ''


def test_feature_engineering_schema():
    df = pd.DataFrame({'clean_comment': ['a b c', 'd e']})
    df = add_basic_nlp_features(df)
    for col in ['token_count','char_count','avg_token_len','unique_ratio']:
        assert col in df.columns


def test_run_pipeline_smoke():
    # Minimal synthetic data to validate end-to-end
    data = {
        'document': ['d1','d1','d2','d2'],
        'comment_id': [1,2,1,2],
        'comment': ['Nice video','Great job','Not good','Quite nice'],
        'flausch': ['yes','yes','no','no']
    }
    df = pd.DataFrame(data)
    df['clean_comment'] = df['comment'].map(clean_text)
    df = add_basic_nlp_features(df)
    out = run_pipeline(df=df, target_col='flausch', test_size=0.5, max_features=50)
    assert 'metrics' in out and 'accuracy' in out['metrics']
