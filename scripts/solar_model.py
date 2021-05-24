import torch
import torch.nn as nn

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

torch.manual_seed(0)
np.random.seed(0)

train_window = 30
test_data_date = 800
epochs = 25

solar = pd.read_csv("final_solar.csv").drop(["interconnect_long", "interconnect_short", "data_type", "date"], axis=1)
solar["LST_DATE"] = pd.to_datetime(solar["LST_DATE"])
solar["LST_DATE"] = (solar["LST_DATE"] - solar["LST_DATE"].min()).dt.days

train = solar[solar["LST_DATE"] < test_data_date]
test = solar[solar["LST_DATE"] >= test_data_date]

encoder = OneHotEncoder(categories='auto', sparse=False)
train["eia_short_name"] = encoder.fit_transform(train["eia_short_name"].to_numpy().reshape(-1, 1))
test["eia_short_name"] = encoder.transform(test["eia_short_name"].to_numpy().reshape(-1, 1))

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.fit_transform(test)

train_scaled = train_scaled[:, -1]
test_scaled = test_scaled[:, -1]

train_data_normalized = torch.FloatTensor(train_scaled).view(-1)

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

train_inout_seq = create_inout_sequences(train_data_normalized, train_window)

def inv_trans(preds, n_features=26, n_obs=27772):
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

model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

fut_pred = len(test)

test_inputs = train_data_normalized[-train_window:].tolist()

model.eval()

for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        test_inputs.append(model(seq).item())

preds = inv_trans(np.array(test_inputs[train_window:]).reshape(-1, 1),n_obs=fut_pred)

plt.title('Solar Generation')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(solar["LST_DATE"], solar['generation'])
plt.plot(test["LST_DATE"], preds)
plt.show()
