import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

torch.manual_seed(0)
np.random.seed(0)

lookahead = 1  # num days ahead to forecast
train_window = 30  # num days lstm use to predict next days generation
test_data_date = 800 # num days in train set
lstm_epochs = 25
ff_epochs = 20

wind = pd.read_csv("../data/final_wind.csv")

wind["DATE"] = pd.to_datetime(wind["DATE"])
wind["DATE"] = (wind["DATE"] - wind["DATE"].min()).dt.days

avg_wind = wind.groupby("DATE").mean()
data = avg_wind.dropna(subset=["generation"])
train = wind[wind["DATE"] < test_data_date]
test = wind[wind["DATE"] >= test_data_date]

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

X_train = train_scaled[:,:-1]
y_train = train_scaled[:, -1]
X_test = test_scaled[:,:-1]
y_test = test_scaled[:, -1]

X_train = torch.FloatTensor(X_train)
y_train_normalized = torch.FloatTensor(y_train).view(-1)
X_test = torch.FloatTensor(X_test)
y_test_normalized = torch.FloatTensor(y_test).view(-1)

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

train_inout_seq = create_inout_sequences(y_train_normalized, train_window)

def inv_trans(preds, n_features=train.shape[0], n_obs=train.shape[1]):
    '''data for inverse transform must have same number of inputs as data it was fit too
       adds dummy zeros to make it same shape and run inverse transform
       brings data to normal scale for visualization'''
    dummy = np.zeros((n_obs, n_features))
    data = np.concatenate([dummy, preds], axis=1)
    return scaler.inverse_transform(data)[:, -1]

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

class Feedforward(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super(Feedforward, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, 1)
            self.sigmoid = torch.nn.Sigmoid()        
        
        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            output = self.fc2(relu)
            output = self.sigmoid(output)
            return output

model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(lstm_epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    print('Epoch {}: train loss: {}'.format(epoch, single_loss.item()))

fut_pred = len(test)

train_inputs = y_train_normalized[-train_window:].tolist()
test_inputs = y_test_normalized[-train_window:].tolist()

model.eval()

for i in range(len(test)):
    seq = torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        test_inputs.append(model(seq).item())

train_preds = inv_trans(np.array(train_inputs[train_window:]).reshape(-1, 1),n_obs=fut_pred)
train_resids = y_train - train_preds
test_preds = inv_trans(np.array(test_inputs[train_window:]).reshape(-1, 1),n_obs=fut_pred)
test_resids = y_test - test_preds

model = Feedforward(X_train.shape[0], 10)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

for epoch in range(ff_epochs):
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = loss_function(y_pred.squeeze(), train_resids)
   
    print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
    loss.backward()
    optimizer.step()

model.eval()
final_preds = model(X_test) + test_preds
final_preds = inv_trans(final_preds, n_obs=test.shape[1])
y_actual = test[:, -1]

score = r2_score(y_actual, final_preds)
print("The r2 score of the model is {:.3f}".format(score))

plt.plot(avg_wind.index, avg_wind.generation, '.', label="actual")
plt.plot(test.index, final_preds, '.', label="predicted")
plt.legend()
plt.title("Model Fit to Wind Power Generation {} Day Forecast".format(lookahead))
plt.show()
