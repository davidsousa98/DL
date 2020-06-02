# Import libraries
import pandas as pd
import numpy as np
import zipfile as zp
from sklearn.impute import KNNImputer

def get_files_zip():
    with zp.ZipFile("./data/Game_stats.zip") as myzip:
        list = myzip.namelist()
    return list

def append_csv(files):
    list_csv = []
    for file in files:
        with zp.ZipFile("./data/Game_stats.zip") as myzip:
            with myzip.open(file) as myfile:
                df = pd.read_csv(myfile)
                list_csv.append(df)
    df_appened = pd.concat(list_csv, axis=0)
    return df_appened


# Reading files from zip without extracting them
get_files_zip()

# Append all files into one sigle DataFrame
df_full = append_csv(get_files_zip()).reset_index()

# Slice DataFrame
df = df_full[['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST',
              'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']]

# Missing values treatment
# Drop rows with too many nulls
null = df.loc[df.HF.isnull()]
df = df[df['HomeTeam'].notna()]
df = df.loc[df['HS'].notna()]
missing_values = df.isnull().sum()

# Filling missing values based on the most similar employee
df.reset_index(inplace=True, drop=True)
temp_df = df.drop(columns=["Div", "Date", "HomeTeam", "AwayTeam"])
imputer = KNNImputer(n_neighbors=5)
filled_df = pd.DataFrame(imputer.fit_transform(temp_df))
filled_df.columns = temp_df.columns
filled_df = pd.concat([filled_df, df["Div"], df["Date"], df["HomeTeam"], df["AwayTeam"]], axis=1)

# Transform and create variables
# Season variable
filled_df['year'] = pd.DatetimeIndex(filled_df['Date']).year

filled_df["month"] = filled_df["Date"].str.split("/", n=2, expand=True)[1]
filled_df['month'] = filled_df['month'].astype(int)

filled_df["season1"] = filled_df["season2"] = np.nan
filled_df.loc[(filled_df["month"] >= 7) & (filled_df["month"] <= 12), "season1"] = filled_df["year"]
filled_df.loc[(filled_df["month"] >= 7) & (filled_df["month"] <= 12), "season2"] = filled_df["year"]+1
filled_df.loc[(filled_df["month"] >= 1) & (filled_df["month"] <= 6), "season1"] = filled_df["year"]-1
filled_df.loc[(filled_df["month"] >= 1) & (filled_df["month"] <= 6), "season2"] = filled_df["year"]

filled_df['season1'] = filled_df['season1'].astype(str)
filled_df['season2'] = filled_df['season2'].astype(str)
filled_df['season1'] = filled_df['season1'].str[:4]
filled_df['season2'] = filled_df['season2'].str[2:4]

filled_df["Season"] = filled_df["season1"] + "/" + filled_df["season2"]
filled_df.drop(columns=["Date", "season1", "season2", "year", "month"], inplace=True)

# League variable
filled_df.loc[filled_df["Div"] == "D1", "League"] = "Bundesliga"
filled_df.loc[filled_df["Div"] == "B1", "League"] = "Jupiler"
filled_df.loc[filled_df["Div"] == "N1", "League"] = "Eredivisie"
filled_df.loc[filled_df["Div"] == "SP1", "League"] = "La Liga"
filled_df.loc[filled_df["Div"] == "D1", "League"] = "Bundesliga"
filled_df.loc[filled_df["Div"] == "E0", "League"] = "Premier League"
filled_df.loc[filled_df["Div"] == "F1", "League"] = "Ligue 1"
filled_df.loc[filled_df["Div"] == "G1", "League"] = "Super League"
filled_df.loc[filled_df["Div"] == "I1", "League"] = "Serie A"
filled_df.loc[filled_df["Div"] == "T1", "League"] = "Super Lig"
filled_df.loc[filled_df["Div"] == "P1", "League"] = "Liga NOS"
filled_df.drop(columns="Div", inplace=True)














