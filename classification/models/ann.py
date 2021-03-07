import torch
from torch import nn
from torch.nn import functional as F



class NeuralNetworkClassifier(nn.Module):
    def __init__(self, **kwargs):
        super(NeuralNetworkClassifier, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features=kwargs["input_shape"], out_features=kwargs["encoding_dim"]*5),
            nn.BatchNorm1d(kwargs["encoding_dim"]*5),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=kwargs["encoding_dim"]*5, out_features=kwargs["encoding_dim"]*4),
            nn.BatchNorm1d(kwargs["encoding_dim"]*4),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=kwargs["encoding_dim"]*4, out_features=kwargs["encoding_dim"]*3),
            nn.BatchNorm1d(kwargs["encoding_dim"]*3),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=kwargs["encoding_dim"]*3, out_features=kwargs["encoding_dim"]*2),
            nn.BatchNorm1d(kwargs["encoding_dim"]*2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=kwargs["encoding_dim"]*2, out_features=kwargs["encoding_dim"])
            )

        self.classification = nn.Sequential(
            nn.Linear(in_features=kwargs["encoding_dim"], out_features=kwargs["classes"]),
            nn.Softmax()
            )

    def forward(self, x):
        x = self.head(x)
        x = self.classification(x)
        return x
