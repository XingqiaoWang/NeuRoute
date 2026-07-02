import torch.nn as nn
from src.indexing_model_base import BaseModel

class Indexing_Model(BaseModel):
    def __init__(self, input_dim, hidden_dim, metric='euclidean'):
        super().__init__(metric)        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
        )
        


