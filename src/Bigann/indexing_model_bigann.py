import torch.nn as nn
from src.indexing_model_base import BaseModel

class Indexing_Model(BaseModel):
    def __init__(self, input_dim, hidden_dim, metric='euclidean'):
        super().__init__(metric)        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Linear(96, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
        )
        
