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


# Top 10 Countries with the highest features

top_ten_score = df_2019.nlargest(10,"Score").loc[:, ["Country or region", "Score"]]
top_ten_gdp = df_2019.nlargest(10,"GDP per capita").loc[:, ["Country or region", "GDP per capita"]]
top_ten_ss = df_2019.nlargest(10,"Social support").loc[:, ["Country or region", "Social support"]]
top_ten_h = df_2019.nlargest(10,"Healthy life expectancy").loc[:, ["Country or region", "Healthy life expectancy"]]
top_ten_f = df_2019.nlargest(10,"Freedom to make life choices").loc[:, ["Country or region", "Freedom to make life choices"]]
top_ten_g = df_2019.nlargest(10,"Generosity").loc[:, ["Country or region", "Generosity"]]
top_ten_c = df_2019.nsmallest(10,"Perceptions of corruption").loc[:, ["Country or region", "Perceptions of corruption"]]


# VISUALIZATION USING SEABORN

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

# Relation between GDP per capita and Healthy life expectancy
palette=["6096B4"]
sns.lmplot(data=df_2019,x="GDP per capita", y="Healthy life expectancy", line_kws={'color': '#F94A29'},palette=palette)
plt.title("Relation between GDP per capita and Healthy life expectancy", fontsize=10)
plt.xticks(rotation=90)
plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
plt.show(block=True)



# Correlation of World Happiness Index using Heatmap [2015-2019]
for df, y in zip(all_data, years):
    fig, ax = plt.subplots(1,1, figsize=(15,5))
    sns.heatmap(df.corr(), annot=True).set_title(y)
    plt.suptitle("Correlation of World Happiness Index [2015-2019]")
    plt.xticks(rotation=90)
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    plt.show(block=True)

# Observations

# 1 - Happiness score is highly correlated with Economy (GDP per Capita), Family, Health (Life Expectancy) and less correlated with Generosity and Trust (Government Corruption)
# 2 - GDP per capita is highly correlated with Score, Social support, Healthy life expectancy and less correlated with Freedom to make life choices, Generosity and Perceptions of corruption.
# 3 - Freedom only correlated with Score and Happiness score



# Contribution of features to happiness ranking.(2019)

mean_2019 = df_2019.mean(numeric_only=True)
mean_2019.iloc[2:].plot(kind="bar", stacked=True)
plt.ylabel("Contribution")
plt.title("Factors Contribution")
plt.xticks(rotation=90)
plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
plt.show(block=True)





