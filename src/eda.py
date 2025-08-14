import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_churn_by_income(df):
    df['ChurnStatus'] = pd.to_numeric(df['ChurnStatus'], errors='coerce')
    income_churn = df.groupby('IncomeLevel')['ChurnStatus'].mean().reset_index()
    plt.figure(figsize=(8,5))
    sns.barplot(data=income_churn, x='IncomeLevel', y='ChurnRate',
                order=['Low', 'Medium', 'High'], palette='coolwarm')
    plt.title('Churn Rate by Income Level')
    plt.show()

def plot_churn_by_complaints(df):
    df['HasUnresolvedComplaint'] = df['Unresolved'] > 0
    unresolved_churn = df.groupby('HasUnresolvedComplaint')['ChurnStatus'].mean().reset_index()
    unresolved_churn['HasUnresolvedComplaint'] = unresolved_churn['HasUnresolvedComplaint'].map({True: 'Yes', False: 'No'})
    plt.figure(figsize=(6,5))
    sns.barplot(data=unresolved_churn, x='HasUnresolvedComplaint', y='ChurnRate', palette='mako')
    plt.title('Churn Rate vs. Unresolved Complaints')
    plt.show()
