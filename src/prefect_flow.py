"""
Prefect workflow for tweet classification pipeline.
"""
from prefect import flow, task
from tweet_classification import (
    pipeline_load_and_preprocess,
    pipeline_eda,
    pipeline_baseline,
    pipeline_model_comparison,
    pipeline_register_best
)

@task
def load_and_preprocess_task():
    return pipeline_load_and_preprocess()

@task
def eda_task(df):
    pipeline_eda(df)

@task
def baseline_task(df):
    return pipeline_baseline(df)

@task
def model_comparison_task():
    return pipeline_model_comparison()

@task
def register_best_task():
    pipeline_register_best()

@flow
def tweet_classification_pipeline():
    df = load_and_preprocess_task()
    eda_task(df)
    baseline_task(df)
    model_comparison_task()
    register_best_task()

if __name__ == "__main__":
    tweet_classification_pipeline()
