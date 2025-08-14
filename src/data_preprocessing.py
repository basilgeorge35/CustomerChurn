import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.feature_engineering import (
    aggregate_transactions, aggregate_service_data, engineer_online_features
)
from src.utils import cap_outliers

def load_and_merge_data(file_path):
    xls = pd.ExcelFile(file_path)
    df_demo = pd.read_excel(xls, 'Customer_Demographics')
    df_trans = pd.read_excel(xls, 'Transaction_History')
    df_service = pd.read_excel(xls, 'Customer_Service')
    df_online = pd.read_excel(xls, 'Online_Activity')
    df_churn = pd.read_excel(xls, 'Churn_Status')

    trans_agg = aggregate_transactions(df_trans)
    service_agg = aggregate_service_data(df_service)
    df_online = engineer_online_features(df_online)

    df = (
        df_demo
        .merge(trans_agg, on='CustomerID', how='left')
        .merge(service_agg, on='CustomerID', how='left')
        .merge(df_online, on='CustomerID', how='left')
        .merge(df_churn, on='CustomerID', how='left')
    )

    return df

def preprocess_data(df):
    # Missing values
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('CustomerID')
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols = df.select_dtypes(include='object').columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    # Cap outliers
    outlier_cols = ['TotalSpend', 'AvgSpend', 'NumTransactions', 'NumComplaints', 'Unresolved']
    for col in outlier_cols:
        cap_outliers(df, col)

    # Standardize
    scale_cols = ['Age', 'TotalSpend', 'AvgSpend', 'NumTransactions',
                  'TotalInteractions', 'NumComplaints', 'Unresolved',
                  'LoginFrequency', 'DaysSinceLastLogin']
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    # One-hot encode
    categorical_cols = ['Gender', 'MaritalStatus', 'IncomeLevel', 'TopCategory', 'ServiceUsage']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df_encoded
