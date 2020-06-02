import pandas as pd
import zipfile as zp


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
    df_appened = pd.concat(list_csv, axis =0)
    return df_appened


# Reading files from zip without extracting them
get_files_zip()

#Append all files into one sigle DataFrame
df_full = append_csv(get_files_zip()).reset_index()

#Filter with the needed variables:
df = df_full[['Div','Date','HomeTeam','AwayTeam','FTHG','FTAG','HTHG','HTAG','HS','AS','HST','AST','HF','AF','HC','AC','HY','AY',
         'HR','AR']]

#Drop rows with too many nulls
null = df.loc[df.HF.isnull()]

df = df[df['HomeTeam'].notna()]
df = df.loc[df['HS'].notna()]
missing_values = df.isnull().sum()



# ['HG', 'HHW', 'ABP', 'HBP', 'HO', 'AO', 'AWH', 'AG']
