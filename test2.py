import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pandas_datareader import data
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time
import datetime
import sqlite3

# Fetch data from Yahoo Finance
ticker = 'AAPL'
period1 = int(time.mktime(datetime.datetime(2010, 1, 1, 23, 59).timetuple()))
period2 = int(time.mktime(datetime.datetime(2022, 2, 1, 23, 59).timetuple()))
interval = '1d'
query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
data = pd.read_csv(query_string)

# establish connection to SQLite database
db_file = r"aaplsqlite.db"
with sqlite3.connect(db_file) as conn:
    # write data to a table in the database
    data.to_sql('aapl_prices', conn, if_exists='replace', index=False)

print("Data written to SQLite database successfully!")

# Set up connection to SQLite database
with sqlite3.connect(db_file) as conn:
    # Load data from database table
    df = pd.read_sql_query("SELECT * from aapl_prices", conn)

# Data cleaning
df = df.drop_duplicates()
df['Date'] = pd.to_datetime(df['Date'])

# Plotting
sns.set_style('whitegrid')
plt.figure(figsize=(12,6))
plt.title('Apple Stock Price')
plt.xlabel('Year')
plt.ylabel('Price ($)')
sns.lineplot(data=df, x='Date', y='Close')
plt.show()

# Map stock exchange

# Split the data into training and test sets
X = df['Open'].values.reshape(-1, 1)
y = df['Close'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to tensors
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()

# Define the model architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# Train the model
num_epochs = 5000
for epoch in range(num_epochs):
    # Forward pass
    y_pred = net(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Make predictions
X_tensor = torch.from_numpy(X).float()
