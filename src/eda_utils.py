import seaborn as sns
import matplotlib.pyplot as plt

def show_summary(df):
    print(df.describe())
    print('info:\n', df.info())
    print('duplicates: ', df.duplicated().sum())
    print("Missing values: ", df.isnull().sum().sum())
    print("\nData types:\n", df.dtypes)
    numerical_columns = df.select_dtypes(include=['float64', 'int64'])
    print("\nCorrelation matrix:\n", numerical_columns.corr())
    for col in df.columns:
        if df[col].dtype == 'object':
            print(f"\nUnique values in {col}:\n", df[col].value_counts())
    print("\nUnique values in each column:\n", df.nunique())

def plot_boxplot(df, column):
    sns.boxplot(x=df[column])
    plt.title(f"Boxplot of {column}")
    plt.show()