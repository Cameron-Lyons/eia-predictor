import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

train_window = 12
test_data_date = 730
epochs = 150
train_window = 12
fut_pred = 12

solar = pd.read_csv(r"C:\Users\clyons\ms1_project\data\final_solar.csv").drop(["interconnect_long", "interconnect_short", "data_type", "date"], axis=1)
solar["LST_DATE"] = pd.to_datetime(solar["LST_DATE"])
solar["LST_DATE"] = (solar["LST_DATE"] - solar["LST_DATE"].min()).dt.days

train_data = solar[solar["LST_DATE"] < test_data_date]
test_data = solar[solar["LST_DATE"] >= test_data_date]

encoder = OneHotEncoder(categories='auto',sparse=False)
train_data["eia_short_name"] = encoder.fit_transform(train_data["eia_short_name"].to_numpy().reshape(-1, 1))

scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data)

train_data_normalized = torch.FloatTensor(train_data_normalized)

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

train_inout_seq = create_inout_sequences(train_data_normalized, train_window)

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1),
                                                self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

model = LSTM(input_size=27)
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

    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

model.eval()

for _ in range(len(test_data)):
    seq = torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        test_inputs.append(model(seq).item())