import pandas as pd
import zipfile as zp
import numpy as np

def get_files_zip(zip):
    """
        Function that lists files inside the .zip folders.

        :param zip: name of the zip file.
    """
    if zip == 'game_stats':
        with zp.ZipFile("./data/Game_stats.zip") as myzip:
            list = myzip.namelist()
    elif zip =='league_table':
        with zp.ZipFile("./data/league_table.zip") as myzip:
            list = myzip.namelist()
    return list

def append_csv(files):
    """
        Function that reads and appends files inside the .zip folders.

        :param files: list of strings with paths to files.

        Returns:
            - DataFrame with all the files appended.
        """
    list_files = []
    for file in files:
        try:
            with zp.ZipFile("./data/Game_stats.zip") as myzip:
                with myzip.open(file) as myfile:
                    df = pd.read_csv(myfile)
                    list_files.append(df)

        except Exception:
            with zp.ZipFile("./data/league_table.zip") as myzip:
                with myzip.open(file) as myfile:
                    df = pd.read_excel(myfile)
                    list_files.append(df)

    df_appened = pd.concat(list_files, axis =0)
    return df_appened.reset_index()


# Reading files from zip without extracting them
get_files_zip('league_table')

#Append all files into one sigle DataFrame
import lxml
df_stats_full = append_csv(get_files_zip('game_stats'))
df_table_full = append_csv(get_files_zip('league_table'))

#Filter with the needed variables:
df_stats = df_stats_full[['Div','Date','HomeTeam','AwayTeam','FTHG','FTAG','HTHG','HTAG','HS','AS','HST','AST','HF','AF','HC','AC','HY','AY',
         'HR','AR']]

df_table = df_table_full[['Squad','GDiff', 'Pts', 'Season']]

# Check missings
missings_stats = df_stats.isnull().sum()
missings_table = df_table.isnull().sum()


nulls_stats = df_stats.loc[df_stats.HF.isnull()]
nulls_table = df_table.loc[df_table.Squad.isnull()]

# Drop rows with too many nulls

df_stats = df_stats[df_stats['HomeTeam'].notna()]
df_stats = df_stats.loc[df_stats['HS'].notna()]

df_table = df_table[df_table['Squad'].notna()]

# Groupby
df_grouped = df_stats.groupby('HomeTeam').count()


# ['HG', 'HHW', 'ABP', 'HBP', 'HO', 'AO', 'AWH', 'AG']
