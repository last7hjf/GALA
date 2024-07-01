import argparse
import os
import torch
os.environ['DGLBACKEND'] = 'pytorch'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import dgl
from dgl.nn.pytorch import GraphConv, GATConv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import torchmetrics.functional as MF

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats, allow_zero_in_degree=True)
        self.conv2 = GraphConv(h_feats, out_feats, allow_zero_in_degree=True)

    def forward(self, g, features):
        x = self.conv1(g, features)
        x = F.relu(x)
        x = self.conv2(g, x)
        return x

class GAT(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, num_heads):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feats, h_feats, num_heads, allow_zero_in_degree=True)
        self.conv2 = GATConv(h_feats * num_heads, num_classes, 1, allow_zero_in_degree=True)

    def forward(self, g, features):
        x = F.elu(self.conv1(g, features).flatten(1))
        x = self.conv2(g, x).mean(1)
        return x

def perpare():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Flickr")
    parser.add_argument('--model', type=str, default='GCN')
    parser.add_argument('--mode', type=str, default='ablation')
    parser.add_argument("--gpus", type=int, default=0)

    args = parser.parse_args()
    return args

@torch.no_grad()
def evaluate(model, g, features, labels, num_classes, mask):
    model.eval()
    logits = model(g, features)
    logits = logits[mask]
    labels = labels[mask]
    accuracy = MF.accuracy(logits, labels, task='multiclass', num_classes=num_classes)
    return accuracy

args = perpare()
g = dgl.load_graphs(f"self_graph/{args.dataset}.bin")[0][0]
device = torch.device("cuda:0")
gpus = ['5,0', '5,4', '0,4', '5,1'][args.gpus]


if args.model == "GCN":
    model = GCN(g.ndata['feat'].shape[1], 256, g.ndata['label'].max().item() + 1)

elif args.model == "GAT":
    if args.dataset == 'Reddit':
        model = GAT(g.ndata['feat'].shape[1], 128, g.ndata['label'].max().item() + 1, 4).to(device)
    elif args.dataset == 'ogbn-products':
        model = GAT(g.ndata['feat'].shape[1], 32, g.ndata['label'].max().item() + 1, 4).to(device)
    else:
        model = GAT(g.ndata['feat'].shape[1], 256, g.ndata['label'].max().item() + 1, 8).to(device)
else:
    raise ValueError(f"Undefined model: {args.model}")


if args.mode == 'train':
    print("train mode test...")
    state_dict = torch.load(f'GALA_trainres/{args.model}_{args.dataset}_combo{args.gpus}.pth')
elif args.mode == 'pretrain':
    print("pretrain mode test...")
    state_dict = torch.load(f'GALA_trainres/{args.model}_{args.dataset}_combo{args.gpus}_noCASE.pth')
elif args.mode == 'ablation':
    print("ablation mode test...")
    state_dict = torch.load(f'GALA_trainres/{args.model}_{args.dataset}_combo{args.gpus}_ablation.pth')
else:
    raise ValueError(f"Undefined mode: {args.mode}")

new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
model = model.to(device)
g = g.to(device)
features = g.ndata['feat']
labels = g.ndata['label']
num_classes = g.ndata['label'].max().item() + 1
test_mask = g.ndata['test_mask'].to(torch.bool)
print(f"testing {args.model} model on {args.dataset}...")
test_acc = evaluate(model, g, features, labels, num_classes, test_mask)
print(f'test acc:{test_acc.item():.4f}')