import pyupbit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
# print(pyupbit.get_tickers(fiat='KRW'))
from sklearn.preprocessing import MinMaxScaler
import keras

access = "**************"
secret = "^^^^^^^^^^^^^^"

upbit = pyupbit.Upbit(access, secret)

print(upbit.get_balance())

data = pyupbit.get_ohlcv("KRW-SXP", interval="minute1", count=1000)
print(data.head)

n_train_rows = int(data.shape[0]*.8) - 1

train = data.iloc[:n_train_rows, :]
test = data.iloc[n_train_rows:, :]

sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(train.values)
test_set_scaled = sc.fit_transform(test.values)

steps = 3

x_train = []
y_train = []

for i in range(steps, training_set_scaled.shape[0] - steps):
    x_train.append(training_set_scaled[i - steps: i, :])
    y_train.append(training_set_scaled[i, :])

x_train, y_train = np.array(x_train), np.array(y_train)

print(x_train.shape)

x_test = []
y_test = []

for i in range(steps, test_set_scaled.shape[0]):
    x_test.append(test_set_scaled[i-steps:i, :])
    y_test.append(test_set_scaled[i, :])

x_test, y_test = np.array(x_test), np.array(y_test)

print(x_test.shape)

model = keras.models.Sequential()

epochs = 100

model.add(keras.layers.LSTM(units=50, return_sequences = True, input_shape = (x_train.shape[1],5)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(units=50, return_sequences = True))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(units=50, return_sequences = True))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(units=50))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(units=5))
model.compile(loss="mse", optimizer="adam")
model.fit(x_train, y_train, batch_size = 5, epochs = epochs)
print(model.summary)

model.save("multiple_features_"+str(steps)+"_steps_"+str(epochs)+"_epochs.h5")


results = model.evaluate(x_test, y_test, batch_size = 32)
print('test lost, test acc:', results)

# Predict values from test data trained using training data
y_hat = model.predict(x_test)
print(y_hat)
y_hat = sc.inverse_transform(y_hat)

y_test = test[steps:].reset_index()

# Visualise the ask_price predictions
plt.figure(figsize = (18,9))
plt.plot(y_test['open'], color = 'red', label = 'y_test')
plt.plot(y_hat[:,0], color = 'blue', label = 'y_hat')
plt.title('y_hat["open"] vs y_test["open"]')
plt.ylabel('open')
plt.legend()
plt.show()

# Visualise the bid_price predictions
plt.figure(figsize = (18,9))
plt.plot(y_test['close'], color = 'red', label = 'y_test')
plt.plot(y_hat[:,3], color = 'blue', label = 'y_hat')
plt.title('y_hat["close"] vs y_test["close"]')
plt.ylabel('close')
plt.legend()
plt.show()

# while(1):
#     current = pyupbit.get_current_price("KRW-SXP") * upbit.get_balance("KRW-SXP")
#     mine = 5990 * upbit.get_balance("KRW-SXP")
#     print("The current price of Swipe is :", current)
#     print(" The ammount of Swipe I bought :", mine)
#
#     print("The profit is : ", int(current - mine), "won")
#     print("The percentage profit is : ", int(current-mine)/mine * 100, "%")
#     time.sleep(1)