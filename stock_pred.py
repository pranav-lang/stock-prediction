import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

# set plot figure size
rcParams['figure.figsize'] = 20, 10

# read data
df = pd.read_csv("NSE-TATA.csv")

# convert date column to datetime
df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
df.index = df['Date']

# plot close price history
plt.figure(figsize=(16,8))
plt.plot(df["Close"], label='Close Price history')

# prepare data
data = df.sort_index(ascending=True, axis=0)
new_dataset = pd.DataFrame(index=range(0,len(df)), columns=['Date','Close'])
for i in range(0,len(data)):
    new_dataset["Date"][i] = data['Date'][i]
    new_dataset["Close"][i] = data["Close"][i]

new_dataset.index = new_dataset.Date
new_dataset.drop("Date", axis=1, inplace=True)

final_dataset = new_dataset.values

train_data = final_dataset[0:987, :]
valid_data = final_dataset[987:, :]

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(final_dataset)

x_train_data, y_train_data = [], []
for i in range(60, len(train_data)):
    x_train_data.append(scaled_data[i-60:i, 0])
    y_train_data.append(scaled_data[i, 0])

x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)

# train random forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
rf_model.fit(x_train_data, y_train_data)

# prepare test data
inputs_data = new_dataset[len(new_dataset)-len(valid_data)-60:].values
inputs_data = inputs_data.reshape(-1,1)
inputs_data = scaler.transform(inputs_data)

X_test = []
for i in range(60, inputs_data.shape[0]):
    X_test.append(inputs_data[i-60:i, 0])
X_test = np.array(X_test)

# predict closing price using random forest model
closing_price = rf_model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price.reshape(-1, 1))


filename = 'random_forest_model.sav'
pickle.dump(rf_model, open(filename, 'wb'))

# plot results
train_data = new_dataset[:987]
valid_data = new_dataset[987:]
valid_data['Predictions'] = closing_price
plt.plot(train_data["Close"])
plt.plot(valid_data[['Close',"Predictions"]])
