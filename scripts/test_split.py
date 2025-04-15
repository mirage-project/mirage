from test_clip import process_model
import torch
import torch.nn as nn
import torch.nn.functional as F

class SplitModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SplitModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x1, x2 = torch.split(x, x.shape[1] // 2, dim=1)
    
        x1 = self.fc2(x1)
        x1 = F.relu(x1)
        
        # Reuse x2 without processing it (tensor reuse)
        x = torch.cat([x1, x2], dim=1)
        
        # Output layer
        x = self.fc_out(x)
        
        return x

UNSUPPORTED_OPS = []  # Add any unsupported operations
IGNORE_OPS = []       # Add any operations to ignore

input_dim = 128  
hidden_dim = 64
output_dim = 10
model = SplitModel(input_dim, hidden_dim, output_dim)

batch_size = 1
dummy_input = torch.randn(batch_size, input_dim)

subgraphs, operators = process_model(model, dummy_input, "SplitModel")

print(f"Processing complete. Got {len(subgraphs)} subgraphs.")

