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


top_ten_score = df_2019.nlargest(10,"Score")
top_ten_gdp = df_2019.nlargest(10,"GDP per capita")
top_ten_ss = df_2019.nlargest(10,"Social support")
top_ten_h = df_2019.nlargest(10,"Healthy life expectancy")
top_ten_f = df_2019.nlargest(10,"Freedom to make life choices")
top_ten_g = df_2019.nlargest(10,"Generosity")
top_ten_c = df_2019.nsmallest(10,"Perceptions of corruption")