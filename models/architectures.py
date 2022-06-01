import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
import math

class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=5):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]


class ShallowRegressionLSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = 3

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers,
            dropout=0.3
        )

        self.linear = nn.Sequential(
            nn.Linear(in_features=self.hidden_units, out_features=self.hidden_units),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_units),
            nn.Linear(in_features=self.hidden_units, out_features=1)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return out
    
    
class GRU(nn.Module):
    def __init__(self, num_sensors, hidden_units):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = 3

        self.GRU = nn.GRU(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers,
            dropout=0.3
        )

        self.linear = nn.Sequential(
            nn.Linear(in_features=self.hidden_units, out_features=self.hidden_units),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_units),
            nn.Linear(in_features=self.hidden_units, out_features=1)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        _, hn = self.GRU(x)
        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return out



class Encoder(nn.Module):
    def __init__(self, num_sensors, hidden_units):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = 1

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.linear = nn.Sequential(
            nn.Linear(in_features=self.hidden_units, out_features=self.hidden_units),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_units),
            nn.Linear(in_features=self.hidden_units, out_features=1)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()

        output, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return output, hn, cn


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        attention_dim = embed_dim // num_heads

        self.WQ = nn.Linear(embed_dim, attention_dim, bias=False)
        self.WK = nn.Linear(embed_dim, attention_dim, bias=False)
        self.WV = nn.Linear(embed_dim, attention_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # query, key, value: (batch_size, length, embed_dim)
        # mask: (batch_size, length, length)

        Q = self.WQ(query)
        K = self.WK(key)
        V = self.WV(value)
        # Q, K, V: (batch_size, length, attention_dim)

        norm_factor = math.sqrt(Q.shape[-1])
        dot_products = torch.bmm(Q, K.transpose(1, 2)) / norm_factor

        attention_score = nn.functional.softmax(dot_products, dim=-1)
        attention = torch.bmm(self.dropout(attention_score), V)
        # attention_score: (batch_size, length, length)
        # attention: (batch_size, length, attention_dim)

        return attention, attention_score


class Decoder(nn.Module):
    def __init__(self, num_sensors, hidden_units):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = 1

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.linear = nn.Sequential(
            nn.Linear(in_features=self.hidden_units, out_features=self.hidden_units),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_units),
            nn.Linear(in_features=self.hidden_units, out_features=1)
        )

    def forward(self, x, hn, cn):
        batch_size = x.shape[0]

        output, (hn, cn) = self.lstm(x, (hn, cn))
        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return output, hn, cn


class Seq2Seq(nn.Module):
    def __init__(self, num_sensors, hidden_units):
        super().__init__()
        self.num_sensors = num_sensors
        self.hidden_units = hidden_units
        self.encoder = Encoder(num_sensors, hidden_units)
        self.decoder = Decoder(num_sensors, hidden_units)
        self.attention = Attention(num_sensors, 1, 0.1)

        self.linear = nn.Sequential(
            nn.Linear(in_features=self.hidden_units, out_features=self.hidden_units),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_units),
            nn.Linear(in_features=self.hidden_units, out_features=1)
        )

        self.norm1 = nn.LayerNorm(num_sensors)

    def forward(self, x):
        out, hn, cn = self.encoder(x)
        attention, attention_score = self.attention(query=x, key=x,
                                                    value=x)
        #outputs = x + attention
        # print(x.shape, attention.shape, attention_score.shape)
        # outputs = torch.bmm(x, attention_score)

        # outputs = self.norm1(outputs)
        out, hn, cn = self.decoder(x, hn, cn)

        out = self.linear(hn[0]).flatten()

        return out