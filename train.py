import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_preprocessing import load_and_merge_data, preprocess_data
from src.modeling import get_models
from src.train_utils import run_baseline_cv, get_search_space, tune_model, evaluate_model
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, MODEL_PATH
import joblib

def main():
    # === Load raw and preprocess ===
    print("Loading raw data...")
    df = load_and_merge_data(RAW_DATA_PATH)

    print("Preprocessing data...")
    df_processed = preprocess_data(df)
    df_processed.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed data saved to: {PROCESSED_DATA_PATH}")

    # === Train/test split ===
    X = df_processed.drop(["CustomerID", "ChurnStatus"], axis=1)
    y = df_processed["ChurnStatus"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # === Baseline CV ===
    models = get_models()
    print("Running baseline cross-validation...")
    best_model_name = run_baseline_cv(models, X_train, y_train)
    print(f"Best baseline model: {best_model_name}")

    # === Hyperparameter tuning ===
    print("Tuning best model...")
    search_space = get_search_space(best_model_name)
    opt = tune_model(models[best_model_name], search_space, X_train, y_train)
    print("Best Params:", opt.best_params_)
    print("Best CV score:", opt.best_score_)

    # === Final evaluation ===
    evaluate_model(opt.best_estimator_, X_test, y_test)

    # === Save model ===
    joblib.dump(opt.best_estimator_, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
