import matplotlib.pyplot as plt
import seaborn as sns
from src.data_preprocessing import load_and_merge_data
from src.config import RAW_DATA_PATH

if __name__ == "__main__":
    df = load_and_merge_data(RAW_DATA_PATH)
    # print(df.head())
    # print(df.info())

    numeric_cols = ['Age', 'TotalSpend', 'AvgSpend', 'NumTransactions',
                    'TotalInteractions', 'NumComplaints', 'Unresolved',
                    'LoginFrequency', 'DaysSinceLastLogin']

    df[numeric_cols].hist(figsize=(16, 12))
    plt.suptitle("Histograms of Numeric Features", fontsize=18)
    plt.show()

    # Calculate churn rate per income level
    income_churn = df.groupby('IncomeLevel')['ChurnStatus'].mean().reset_index()
    income_churn.columns = ['IncomeLevel', 'ChurnRate']

    plt.figure(figsize=(8,5))
    sns.barplot(data=income_churn, x='IncomeLevel', y='ChurnRate',
                order=['Low', 'Medium', 'High'], palette='coolwarm', hue = 'IncomeLevel')

    plt.title('Churn Rate by Income Level', fontsize=14)
    plt.ylabel('Churn Rate')
    plt.xlabel('Income Level')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Create a binary column: Has unresolved complaint (1 if unresolved complaints > 0)
    df['HasUnresolvedComplaint'] = df['Unresolved'] > 0

    # Group by unresolved complaints and calculate churn rate
    unresolved_churn = df.groupby('HasUnresolvedComplaint')['ChurnStatus'].mean().reset_index()
    unresolved_churn['HasUnresolvedComplaint'] = unresolved_churn['HasUnresolvedComplaint'].map({True: 'Yes', False: 'No'})
    unresolved_churn.columns = ['HasUnresolvedComplaint', 'ChurnRate']

    plt.figure(figsize=(6,5))
    sns.barplot(data=unresolved_churn, x='HasUnresolvedComplaint', y='ChurnRate', palette='mako', hue='HasUnresolvedComplaint')
    plt.title('Churn Rate vs. Unresolved Complaints')
    plt.ylabel('Churn Rate')
    plt.xlabel('Unresolved Complaints')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
