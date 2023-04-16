#import modules
import pandas as pd
import numpy as np
import seaborn as sns
#sns.get_dataset_names()
from pandas_datareader import data
from sklearn.model_selection import train_test_split
import time
import datetime
import sqlite3
from sqlite3 import Error

conn = sqlite3.connect('aapl_database') 
c = conn.cursor()

#select ticker and define time range
ticker = 'AAPL'
period1 = int(time.mktime(datetime.datetime(2010, 1, 1, 23, 59).timetuple()))
period2 = int(time.mktime(datetime.datetime(2022, 2, 1, 23, 59).timetuple()))
interval = '1d'
query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
data = pd.read_csv(query_string)
print(data)
data.to_csv('APPL Prices')


def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()


if __name__ == '__main__':
    create_connection(r"C:\Users\TRAUTMANN\OneDrive - ZHAW\Master\FS23 - ADS\ADS Project\Scraping Yahoo Finance\aaplsqlite.db")
