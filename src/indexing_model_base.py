import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, metric='cosine'):
        super().__init__()
        self.encoder = None
        self.metric = metric

    def forward(self, x):
        encoded = self.encoder(x)
        if self.metric == 'cosine':
            encoded = torch.nn.functional.normalize(encoded, p=2, dim=1)
        return encoded
        
    def similarity_loss(self, inputs, encoded, threshold=0.5, threshold2=0.7, gamma=0.9):
        """
        Computes similarity or distance between x1 and x2 based on self.metric.
        """
        if self.metric == 'cosine':
            return self._selective_similarity_loss_cosine(inputs, encoded, threshold, gamma = gamma)
        elif self.metric == 'euclidean':
            return self._selective_similarity_loss_eucli(inputs, encoded, threshold, gamma = gamma)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        
    def _selective_similarity_loss_cosine(self, inputs, encoded, threshold=0.5, threshold2=0.7, gamma=0.9):
        batch_size = inputs.size(0)
        
        if batch_size == 1:
            return torch.tensor(0.0, device=inputs.device)

        inputs_norm = nn.functional.normalize(inputs, p=2, dim=1)
        
        # Use encoded as-is
        S_in = torch.matmul(inputs_norm, inputs_norm.t())
        S_emb = torch.matmul(encoded, encoded.t())

        # Create positive pair mask
        M_positive = ((S_in > threshold) | (S_emb > threshold2)).float()
        identity_mask = torch.eye(batch_size, device=inputs.device)
        M_positive = M_positive * (1 - identity_mask)

        # Clip S_in and calculate main loss

        loss_positive = M_positive * ((1 - S_emb) - gamma * (1 - S_in)) ** 2
        
        num_positive_pairs = M_positive.sum()
        similarity_loss = loss_positive.sum() / num_positive_pairs if num_positive_pairs > 0 else torch.tensor(0.0, device=inputs.device)

        total_loss = similarity_loss
        
        return total_loss,0

    def _selective_similarity_loss_eucli(self, inputs, encoded, threshold=0.5,threshold2=0.7,gamma =0.9):
        
        def theoretical_distance_scaling(d_high, d_low):
            distance_ratio = (d_low / d_high) ** 0.5
            return distance_ratio
        
        def pairwise_euclidean(x):
            # x: (n, d)
            # return: (n, n) distance matrix
            x_norm = (x ** 2).sum(dim=1).view(-1, 1)  # (n, 1)
            dist = x_norm + x_norm.t() - 2.0 * torch.matmul(x, x.t())  # broadcasting
            return torch.sqrt(torch.clamp(dist, min=1e-12))  # numerical stability
        
        distance_ratio = theoretical_distance_scaling(inputs.size(1),encoded.size(1))
        batch_size = inputs.size(0)
        
        if batch_size == 1:
            return torch.tensor(0.0, device=inputs.device)
        
        S_in = pairwise_euclidean(inputs)  # Input pairwise Euclidean distance
        S_emb = pairwise_euclidean(encoded)  # Embedding pairwise Euclidean distance

        threshold2 = threshold *gamma*distance_ratio
        
        # Create mask for positive pairs based on threshold
        M_positive = ((S_in < threshold) | (S_emb < threshold2)).float()

        # Exclude self-similarity
        identity_mask = torch.eye(batch_size, device=inputs.device)
        M_positive = M_positive * (1 - identity_mask)

        # Compute loss only for selected pairs
        
        loss_positive = M_positive * (S_emb - gamma*distance_ratio*S_in) ** 2

        # Count the number of positive pairs
        num_positive_pairs = M_positive.sum()
        loss = loss_positive.sum() / torch.clamp(num_positive_pairs, min=1.0)

        return loss
    
    