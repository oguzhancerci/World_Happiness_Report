# Importing modules

import numpy as np
import pandas as pd # Data processing
import matplotlib.pyplot as plt # Plotting
import seaborn as sns # Visualization
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
colorama_init()

# Loading Data
df_2015 = pd.read_csv('Datasets/2015.csv', delimiter=',')
df_2016 = pd.read_csv('Datasets/2016.csv', delimiter=',')
df_2017 = pd.read_csv('Datasets/2017.csv', delimiter=',')
df_2018 = pd.read_csv('Datasets/2018.csv', delimiter=',')
df_2019 = pd.read_csv('Datasets/2019.csv', delimiter=',')

# Description of Data
all_data = [df_2015, df_2016, df_2017, df_2018, df_2019]
years = [2015, 2016, 2017, 2018, 2019]
shape = (
    {"2015": {"Rows": df_2015.shape[0], "Columns": df_2015.shape[1]}},
    {"2016": {"Rows": df_2016.shape[0], "Columns": df_2016.shape[1]}},
    {"2017": {"Rows": df_2017.shape[0], "Columns": df_2017.shape[1]}},
    {"2018": {"Rows": df_2018.shape[0], "Columns": df_2018.shape[1]}},
    {"2019": {"Rows": df_2019.shape[0], "Columns": df_2019.shape[1]}}
)

# Information of each years
for df, year in zip(all_data, years):
    print(f" {Fore.GREEN}Informations of {year}{Style.RESET_ALL}")
    print(df.info())

# Null values of years
for df, year in zip(all_data, years):
    print(f" {Fore.GREEN}Null values of {year}{Style.RESET_ALL}")
    print(df.isnull().sum())


# Categorization of Categorical and Numerical columns

# I defined the numerical variables which has number of unique value less than 10 as a categorical variable. (numerical_but_categorical)
# And categorical variables which has number of unique value more than 20 as a numerical variable. (categorical_but_cardinal)

# 2015
for col in all_data:
    categorical_cols_2015 = [col for col in df_2015.columns if str(df_2015[col].dtypes) in ["category", "object", "bool"]]
    numerical_but_categorical_2015 = [col for col in df_2015.columns if df_2015[col].nunique() < 10 and df_2015[col].dtypes in ["int64", "float64"]]
    categorical_but_cardinal_2015 = [col for col in df_2015.columns if df_2015[col].nunique() > 20 and str(df_2015[col].dtypes) in ["category", "object"]]
    numerical_cols_2015 = [col for col in df_2015.columns if df_2015[col].dtypes in ["int64", "float64"]]
    numerical_cols_2015 = [col for col in df_2015.columns if col not in categorical_cols_2015]
    print(f"Categorical Variables: {categorical_cols_2015}, Numerical But Categorical Variables: {numerical_but_categorical_2015}, Categorical But Numerical Variables: {categorical_but_cardinal_2015} ")



# 2016
for col in all_data:
    categorical_cols_2016 = [col for col in df_2016.columns if str(df_2016[col].dtypes) in ["category", "object", "bool"]]
    numerical_but_categorical_2016 = [col for col in df_2016.columns if df_2016[col].nunique() < 10 and df_2016[col].dtypes in ["int64", "float64"]]
    categorical_but_cardinal_2016 = [col for col in df_2016.columns if df_2016[col].nunique() > 20 and str(df_2016[col].dtypes) in ["category", "object"]]
    numerical_cols_2016 = [col for col in df_2016.columns if df_2016[col].dtypes in ["int64", "float64"]]
    numerical_cols_2016 = [col for col in df_2016.columns if col not in categorical_cols_2016]
    print(f"Categorical Variables: {categorical_cols_2016}, Numerical But Categorical Variables: {numerical_but_categorical_2016}, Categorical But Numerical Variables: {categorical_but_cardinal_2016} ")



#2017
for col in all_data:
    categorical_cols_2017 = [col for col in df_2017.columns if str(df_2017[col].dtypes) in ["category", "object", "bool"]]
    numerical_but_categorical_2017 = [col for col in df_2017.columns if df_2017[col].nunique() < 10 and df_2017[col].dtypes in ["int64", "float64"]]
    categorical_but_cardinal_2017 = [col for col in df_2017.columns if df_2017[col].nunique() > 20 and str(df_2017[col].dtypes) in ["category", "object"]]
    numerical_cols_2017 = [col for col in df_2017.columns if df_2017[col].dtypes in ["int64", "float64"]]
    numerical_cols_2017 = [col for col in df_2017.columns if col not in categorical_cols_2017]
    print(f"Categorical Variables: {categorical_cols_2017}, Numerical But Categorical Variables: {numerical_but_categorical_2017}, Categorical But Numerical Variables: {categorical_but_cardinal_2017} ")



#2018
for col in all_data:
    categorical_cols_2018 = [col for col in df_2018.columns if str(df_2018[col].dtypes) in ["category", "object", "bool"]]
    numerical_but_categorical_2018 = [col for col in df_2018.columns if df_2018[col].nunique() < 10 and df_2015[col].dtypes in ["int64", "float64"]]
    categorical_but_cardinal_2018 = [col for col in df_2018.columns if df_2018[col].nunique() > 20 and str(df_2018[col].dtypes) in ["category", "object"]]
    numerical_cols_2018 = [col for col in df_2018.columns if df_2018[col].dtypes in ["int64", "float64"]]
    numerical_cols_2018 = [col for col in df_2018.columns if col not in categorical_cols_2018]
    print(f"Categorical Variables: {categorical_cols_2018}, Numerical But Categorical Variables: {numerical_but_categorical_2018}, Categorical But Numerical Variables: {categorical_but_cardinal_2018} ")



#2019
for col in all_data:
    categorical_cols_2019 = [col for col in df_2019.columns if str(df_2019[col].dtypes) in ["category", "object", "bool"]]
    numerical_but_categorical_2019 = [col for col in df_2019.columns if df_2019[col].nunique() < 10 and df_2019[col].dtypes in ["int64", "float64"]]
    categorical_but_cardinal_2019 = [col for col in df_2019.columns if df_2019[col].nunique() > 20 and str(df_2019[col].dtypes) in ["category", "object"]]
    numerical_cols_2019 = [col for col in df_2019.columns if df_2019[col].dtypes in ["int64", "float64"]]
    numerical_cols_2019 = [col for col in df_2019.columns if col not in categorical_cols_2019]
    print(f"Categorical Variables: {categorical_cols_2019}, Numerical But Categorical Variables: {numerical_but_categorical_2019}, Categorical But Numerical Variables: {categorical_but_cardinal_2019} ")



# Happiest and the most unhappy countries according to "Happiness Rank"

df_2015.loc[0:9, ["Happiness Rank", "Country"]] # Happiest countries of 2015
df_2015[-10:][["Happiness Rank", "Country"]].iloc[::-1] # The most unhappy countries of 2015

df_2016.loc[0:9, ["Happiness Rank", "Country"]] # Happiest countries of 2016
df_2016[-10:][["Happiness Rank", "Country"]].iloc[::-1] # The most unhappy countries of 2016

df_2017.loc[0:9, ["Happiness.Rank", "Country"]] # Happiest countries of 2017
df_2017[-10:][["Happiness.Rank", "Country"]].iloc[::-1] # The most unhappy countries of 2017

df_2018.loc[0:9, ["Overall rank", "Country or region"]] # Happiest countries of 2018
df_2018[-10:][["Overall rank", "Country or region"]].iloc[::-1] # The most unhappy countries of 2018

df_2019.loc[0:9, ["Overall rank", "Country or region"]] # Happiest countries of 2019
df_2019[-10:][["Overall rank", "Country or region"]].iloc[::-1] # The most unhappy countries of 2019


df_2019.loc[df_2019["Country or region"] == "Turkey"]

# For 2015 and 2016 datasets, country counts by each region using countplot

def countries_by_region(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("###################################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.title("Country counts by region")
        plt.ylabel('counts')
        plt.xticks(rotation=90)
        plt.tight_layout(pad=1.0, w_pad=10.0, h_pad=10.0)
        plt.show(block=True)

countries_by_region(df_2015, "Region", plot=True)
countries_by_region(df_2016, "Region", plot=True)


# Happiness scores by region for 2015's dataset
sns.barplot(x=df_2015["Region"], y=df_2015["Happiness Score"], data=df_2015)
plt.title("Happiness scores by region")
plt.xticks(rotation=90)
plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
plt.show(block=True)

# Effect of Economy (GDP per Capita) and Life Expectancy to Happiness Score using scatterplot
sns.scatterplot(x=df_2015["Economy (GDP per Capita)"], y=df_2015["Happiness Score"], hue= df_2015["Health (Life Expectancy)"], data=df_2015)
plt.title("Effect of Economy (GDP per Capita) to Happiness Score")
plt.xticks(rotation=90)
plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
plt.show(block=True)

# Effect of Family bounds on Happiness Score using scatterplot
sns.lineplot(x=df_2015["Family"], y=df_2015["Happiness Score"], data=df_2015)
plt.title("Effect of  Family bounds on Happiness Score")
plt.xticks(rotation=90)
plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
plt.show(block=True)

sns.scatterplot(x=df_2015["Freedom"], y=df_2015["Happiness Score"], data=df_2015)
plt.title("Effect of Freedom on Happiness Score")
plt.xticks(rotation=90)
plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
plt.show(block=True)

df = df_2015
def plotCorrelationMatrix(df, graphWidth):

    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {df}', fontsize=15)
    plt.show()




