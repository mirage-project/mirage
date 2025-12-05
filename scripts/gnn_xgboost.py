# imports
import os, math, json, random
import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.data import Data
import torch.nn.functional as F
from itertools import product
from sklearn.model_selection import KFold
import numpy as np
import xgboost as xgb
from typing import Dict, Any
import joblib  # optional, used to save fitted XGB in addition to .json

from utils import to_kernel_graph

# models
class SubgraphEncoder(nn.Module):
    """
    Same as your original model up to the graph embedding 'g'.
    Produces a [num_graphs, hidden] representation.
    """
    def __init__(self, node_in=60, edge_in=8, hidden=256, layers=8, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.node_in = node_in
        self.edge_in = edge_in

        self.x_proj = nn.Sequential(
            nn.Linear(node_in, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )

        def make_mlp():
            return nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, hidden),
            )

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GINEConv(make_mlp(), edge_dim=edge_in))
        self.bns.append(nn.BatchNorm1d(hidden))
        for _ in range(layers - 1):
            self.convs.append(GINEConv(make_mlp(), edge_dim=edge_in))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden // 2, 1),
        )

    @torch.no_grad()
    def embed(self, data):
        """
        Convenience helper that runs in eval/no-grad and returns embeddings.
        """
        was_training = self.training
        self.eval()
        g = self.forward(data, return_embeddings=True)  # [B, hidden]
        if was_training:
            self.train()
        return g

    def forward(self, data, return_embeddings=False):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Flatten node features [N,6,10] -> [N,60] if needed
        if x.dim() == 3:
            x = x.view(x.size(0), -1)

        # Flatten edge features [E,2,4] -> [E,8] if needed
        if edge_attr is not None and edge_attr.dim() == 3:
            edge_attr = edge_attr.view(edge_attr.size(0), -1)

        x = self.x_proj(x)  # [N, hidden]

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Graph-level representation
        if return_embeddings:
            return global_mean_pool(x, batch)  # [num_graphs, hidden]
        else:
            g = global_mean_pool(x, batch)  # [num_graphs, hidden]
            out = self.head(g).squeeze(-1)  # [num_graphs]
            return out

class SubgraphRegressorXGB(nn.Module):
    """
    Wraps the encoder and uses an XGBoost regressor as the head.
    - Call `fit_xgb(dataloader, y_tensor_or_extractor=...)` to train the XGB head.
    - Forward will return predictions if the XGB head is fitted; otherwise it returns embeddings.
    """
    def __init__(self, encoder, xgb_kwargs=None):
        super().__init__()
        self.encoder = encoder
        self.xgb = None
        self.xgb_kwargs = xgb_kwargs or dict(
            n_estimators=600,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
        )

    def forward(self, data):
        """
        If XGB is trained, returns predictions [num_graphs].
        If not, returns embeddings [num_graphs, hidden] so you can pretrain/diagnose.
        """
        g = self.encoder.embed(data)  # [B, hidden]
        if self.xgb is None:
            return g  # embeddings
        preds = self._xgb_predict_from_tensor(g)  # [B]
        return preds

    def _xgb_predict_from_tensor(self, g_tensor: torch.Tensor) -> torch.Tensor:
        g_cpu = g_tensor.detach().cpu().numpy()
        yhat = self.xgb.predict(g_cpu)
        return torch.from_numpy(yhat).to(g_tensor.device).float()

    @torch.no_grad()
    def extract_embeddings(self, dataloader, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Runs the encoder over a dataloader and returns (X, y) where:
          - X: np.ndarray of shape [num_graphs, hidden]
          - y: np.ndarray of shape [num_graphs]
        Assumes each batch item has `data.y` as the scalar regression target.
        """
        self.to(device)
        was_training = self.training
        self.eval()

        X, Y = [], []
        for batch in dataloader:
            batch = batch.to(device)
            g = self.encoder.embed(batch)  # [b, hidden]
            X.append(g.detach().cpu().numpy())
            if hasattr(batch, "y") and batch.y is not None:
                y = batch.y.view(-1).detach().cpu().numpy()
                Y.append(y)
        X = np.concatenate(X, axis=0)
        Y = np.concatenate(Y, axis=0) if len(Y) else None

        if was_training:
            self.train()
        return X, Y

    def fit_xgb(self, dataloader, device="cuda" if torch.cuda.is_available() else "cpu", X=None, y=None):
        """
        Fit the XGBoost head. You can either:
          - pass a dataloader (we'll extract embeddings + y), or
          - pass precomputed (X, y) numpy arrays.
        """
        if X is None or y is None:
            X, y = self.extract_embeddings(dataloader, device=device)
        self.xgb = xgb.XGBRegressor(**self.xgb_kwargs)
        self.xgb.fit(X, y)
        return self

    def predict_xgb(self, dataloader_or_tensor, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Inference helper:
          - If given a dataloader of pyg batches -> returns np.ndarray predictions for the whole set.
          - If given a torch.Tensor of embeddings -> returns np.ndarray predictions directly.
        """
        if self.xgb is None:
            raise RuntimeError("XGBoost head is not fitted. Call fit_xgb(...) first.")
        if isinstance(dataloader_or_tensor, torch.Tensor):
            return self.xgb.predict(dataloader_or_tensor.detach().cpu().numpy())

        # dataloader path
        self.to(device)
        was_training = self.training
        self.eval()

        preds = []
        for batch in dataloader_or_tensor:
            batch = batch.to(device)
            g = self.encoder.embed(batch)
            preds.append(self.xgb.predict(g.detach().cpu().numpy()))
        if was_training:
            self.train()
        return np.concatenate(preds, axis=0)

# dataset
NODE_FEATURE_DIM = 10
EDGE_FEATURE_DIM = 4

OP_TYPES = ['kn_input_op', 
            'kn_output_op', 
            'kn_add_op', 
            'kn_div_op',
            'kn_exp_op', 
            'kn_matmul_op', 
            'kn_mul_op',
            'kn_pow_op', 
            'kn_reduction_2_op', 
            'kn_sqrt_op']

def one_hot_op_type(op_type):
    out = torch.zeros(NODE_FEATURE_DIM, dtype=torch.float)
    out[OP_TYPES.index(op_type)] = 1.0
    return out

def compute_flops(op_type, input_tensors):
    if op_type == 'kn_matmul_op':
        M = input_tensors[0][-2]
        K = input_tensors[0][-1]
        N = input_tensors[1][-1]
        return 2 * M * N * K
    elif op_type in ['kn_add_op', 
                     'kn_div_op', 
                     'kn_exp_op', 
                     'kn_mul_op', 
                     'kn_pow_op', 
                     'kn_sqrt_op', 
                     'kn_reduction_2_op']:
        num_elements = 1
        for dim in input_tensors[0]:
            num_elements *= dim
        return num_elements
    elif op_type in ['kn_input_op', 'kn_output_op']:
        return 0
    else:
        raise ValueError(f"Unknown op type: {op_type}")

def get_node_features(node):
    # one-hot op type
    h_op_type = one_hot_op_type(node['op_type'])
    
    # in tensor dimensions
    in_tensors = []
    h_in_dims = torch.zeros(NODE_FEATURE_DIM, dtype=torch.float)
    for i, t in enumerate(node['input_tensors']):
        in_tensors.append(t['dim'])
        for j, dim in enumerate(t['dim']):
            h_in_dims[j + (i * 4)] = dim
    
    # out tensor dimensions
    out_tensors = []
    h_out_dims = torch.zeros(NODE_FEATURE_DIM, dtype=torch.float)
    for i, t in enumerate(node['output_tensors']):
        out_tensors.append(t['dim'])
        for j, dim in enumerate(t['dim']):
            h_out_dims[j + (i * 4)] = dim
    
    # in tensor sizes
    h_in_size = torch.zeros(NODE_FEATURE_DIM, dtype=torch.float)
    for i, t in enumerate(in_tensors):
        h_in_size[i] = math.prod(t)
    
    # out tensor sizes
    h_out_size = torch.zeros(NODE_FEATURE_DIM, dtype=torch.float)
    for i, t in enumerate(out_tensors):
        h_out_size[i] = math.prod(t)

    # computation cost in FLOPs
    flops = compute_flops(node['op_type'], in_tensors)
    h_flops = torch.zeros(NODE_FEATURE_DIM, dtype=torch.float)
    h_flops[0] = flops

    return torch.vstack([h_op_type, h_in_dims, h_out_dims, h_in_size, h_out_size, h_flops])

def get_edge_features(tensor):
    h_edge = torch.zeros(EDGE_FEATURE_DIM, dtype=torch.float)
    for i, dim in enumerate(tensor['dim']):
        h_edge[i] = dim
    
    h_edge_size = torch.zeros(EDGE_FEATURE_DIM, dtype=torch.float)
    h_edge_size[0] = math.prod(tensor['dim'])

    return torch.vstack([h_edge, h_edge_size])

# Load model from checkpoint
def load_gnn_xgb_model(encoder_ckpt: str, encoder_cfg: Dict[str, Any], xgb_model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Load a pretrained GNN encoder + XGBoost head from checkpoints.
    
    Args:
        encoder_ckpt: Path to the encoder .pt checkpoint
        encoder_cfg: Dict with 'hidden', 'layers', 'dropout' keys
        xgb_model_path: Path to the XGBoost model (.json or .pkl)
        device: Device to load the encoder on
    
    Returns:
        SubgraphRegressorXGB with loaded encoder and XGBoost head
    """
    # Load encoder
    encoder = SubgraphEncoder(
        node_in=60,
        edge_in=8,
        hidden=encoder_cfg.get("hidden", 256),
        layers=encoder_cfg.get("layers", 4),
        dropout=encoder_cfg.get("dropout", 0.2),
    ).to(device)
    encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device))
    encoder.eval()
    
    # Create wrapper and load XGBoost head
    model = SubgraphRegressorXGB(encoder=encoder)
    
    if xgb_model_path.endswith('.pkl'):
        model.xgb = joblib.load(xgb_model_path)
    elif xgb_model_path.endswith('.json'):
        model.xgb = xgb.XGBRegressor()
        model.xgb.load_model(xgb_model_path)
    else:
        raise ValueError(f"Unsupported XGBoost model format: {xgb_model_path}")
    
    print(f"Loaded encoder from {encoder_ckpt}")
    print(f"Loaded XGBoost head from {xgb_model_path}")
    
    return model

def infer(model: SubgraphRegressorXGB, subgraph_json_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Perform inference on a single subgraph from a JSON file.
    
    Args:
        model: Trained SubgraphRegressorXGB model
        subgraph_json_path: Path to the subgraph JSON file
        device: Device to run inference on
    
    Returns:
        Predicted execution time (float)
    """
    model.to(device)
    model.eval()
    
    # Load and parse the subgraph JSON
    with open(subgraph_json_path, 'r') as f:
        json_graph = json.load(f)
    
    # Build PyG Data object (same logic as SubgraphDataset._raw_get)
    nodes = []
    edge_index = []
    edge_attr = []
    producer_of = {}
    
    for node_idx, node in enumerate(json_graph):
        nodes.append(get_node_features(node))
        for t in node['output_tensors']:
            producer_of[t['guid']] = node_idx
    
    for dst_idx, node in enumerate(json_graph):
        for t in node['input_tensors']:
            src_idx = producer_of[t['guid']]
            edge_index.append([src_idx, dst_idx])
            edge_attr.append(get_edge_features(t))
    
    data = Data(
        x=torch.stack(nodes, dim=0),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.stack(edge_attr, dim=0),
    ).to(device)
    
    # Run inference
    with torch.no_grad():
        prediction = model(data)
        if isinstance(prediction, torch.Tensor):
            if prediction.dim() == 0:
                return prediction.item()
            else:
                return prediction.view(-1)[0].item()
        else:
            return float(prediction[0])
        
class GNNXGBoost:
    def __init__(self, encoder_ckpt: str, encoder_cfg: Dict[str, Any], xgb_model_path: str):
        self.cost_model = load_gnn_xgb_model(encoder_ckpt, encoder_cfg, xgb_model_path)
    
    def __call__(self, nodes):
        nodes_list = {node: None for node in nodes}
        kernel_graph, _ = to_kernel_graph(nodes_list)
        kernel_graph.to_json("tmp.json")
        exec_time = infer(self.cost_model, "tmp.json")
        os.remove("tmp.json")
        return exec_time