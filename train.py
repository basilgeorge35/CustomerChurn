from src.data_preprocessing import load_and_merge_data, preprocess_data
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH

def main():
    df = load_and_merge_data(RAW_DATA_PATH)
    df_processed = preprocess_data(df)
    df_processed.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed data saved to: {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    main()
