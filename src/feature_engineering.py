import pandas as pd

def aggregate_transactions(df_trans):
    trans_agg = df_trans.groupby('CustomerID').agg({
        'AmountSpent': ['sum', 'mean', 'count'],
        'ProductCategory': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
    }).reset_index()
    trans_agg.columns = ['CustomerID', 'TotalSpend', 'AvgSpend', 'NumTransactions', 'TopCategory']
    return trans_agg

def aggregate_service_data(df_service):
    service_agg = df_service.groupby('CustomerID').agg({
        'InteractionID': 'count',
        'InteractionType': lambda x: (x == 'Complaint').sum(),
        'ResolutionStatus': lambda x: (x == 'Unresolved').sum()
    }).reset_index()
    service_agg.columns = ['CustomerID', 'TotalInteractions', 'NumComplaints', 'Unresolved']
    return service_agg

def engineer_online_features(df_online):
    df_online['DaysSinceLastLogin'] = (
        pd.to_datetime('today') - pd.to_datetime(df_online['LastLoginDate'])
    ).dt.days
    df_online.drop(columns='LastLoginDate', inplace=True)
    return df_online
