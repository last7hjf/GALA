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
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.join import Join


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

@torch.no_grad()
def evaluate(model, g, features, labels, num_classes, mask):
    model.eval()
    logits = model(g, features)
    logits = logits[mask]
    labels = labels[mask]
    accuracy = MF.accuracy(logits, labels, task='multiclass', num_classes=num_classes)
    return accuracy

def test(local_rank, nprocs, g, model, device):
    features = g.ndata['feat']
    labels = g.ndata['label']
    test_mask = g.ndata["test_mask"].to(torch.bool)
    test_acc = evaluate(model, g, features, labels, g.ndata['label'].max().item()+1, test_mask).to(device) / nprocs
    dist.reduce(tensor=test_acc, dst=0)
    dist.barrier()
    if local_rank == 0:
        print(f"test accuracy为{test_acc.item()}")

def train(model, g, local_rank, device):
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask'].to(torch.bool)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    time_log = {}
    train_start = time.time()
    for epoch in range(1, 181):

        model.train()
        with Join([model]):
            # torch.cuda.synchronize()
            start_time = time.time()
            logits = model(g, features)
            loss = criterion(logits[train_mask], labels[train_mask])
            torch.cuda.synchronize()
            end_time = time.time()
            comp_time = end_time - start_time
            time_log[epoch] = comp_time
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if local_rank == 0:
             if epoch % 10 == 0:
                 print(f'Epoch: {epoch}, loss: {loss.item() :3f}')

    print(f"compute time of {torch.cuda.get_device_name(device)}: {sum(time_log.values()) :.4f} s \n")
    torch.cuda.synchronize()
    train_end = time.time()
    if local_rank == 0:
        print(f'end-to-end time: {train_end - train_start : .4f} \n')

    # print(f'end-to-end time of {torch.cuda.get_device_name(device)}: {train_time} \n')




def run(local_rank, nprocs, devices, partition_res, args):
    device = devices[local_rank]
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",  # Use NCCL backend for distributed GPU training
        init_method='env://',
        world_size=nprocs,
        rank=local_rank,
    )
    g = partition_res[local_rank]
    print(f"{torch.cuda.get_device_name(device)} 训练子图{local_rank}，其边数量为{g.num_edges()}..")
    g = g.to(device)
    if args.model == "GCN":
        model = GCN(g.ndata['feat'].shape[1], 256, g.ndata['label'].max().item()+1).to(device)
    elif args.model == "GAT":
        if args.dataset == 'Reddit':
            model = GAT(g.ndata['feat'].shape[1], 128, g.ndata['label'].max().item() + 1, 4).to(device)
        elif args.dataset == 'ogbn-products':
            model = GAT(g.ndata['feat'].shape[1], 32, g.ndata['label'].max().item() + 1, 4).to(device)
        else:
            model = GAT(g.ndata['feat'].shape[1], 256, g.ndata['label'].max().item() + 1, 8).to(device)
    else:
        raise ValueError(f"Undefined model: {args.model}")
    if local_rank == 0:
        print(f"training {args.model} model...")
    model = DDP(model, device_ids=[device], output_device=device)
    train(model, g, local_rank, device)
    if local_rank == 0:
        # print(f"testing {args.model} model...")
        print(f"saving {args.model} model...")
        torch.save(model.state_dict(), f'GALA_trainres/{args.model}_{args.dataset}_combo{args.gpus}_noCASE.pth' )
    # test(local_rank, nprocs, g, model, device)
    dist.destroy_process_group()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--edgelimit", type=str, default='0.25')
    parser.add_argument("--dataset", type=str, default="Flickr")
    parser.add_argument('--model', type=str, default="GCN")
    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
    gpus = ['5,0', '5,4', '4,0', '5,1'][args.gpus]
    num_gpus = len(gpus.split(','))
    os.environ['WORLD_SIZE'] = str(num_gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    devices = []
    for i in range(num_gpus):
        devices.append(i)
    print(devices)
    # sg0 = dgl.load_graphs(f"NESA_2part/{args.dataset}_sg0.bin")[0][0]
    # sg1 = dgl.load_graphs(f"NESA_2part/{args.dataset}_sg1.bin")[0][0]
    sg0 = dgl.load_graphs(f"NESA_2part/{args.dataset}_sg0_{args.edgelimit}.bin")[0][0]
    sg1 = dgl.load_graphs(f"NESA_2part/{args.dataset}_sg1_{args.edgelimit}.bin")[0][0]
    partition_res = [sg0, sg1]
    mp.spawn(run, args=(num_gpus, devices, partition_res, args, ), nprocs=num_gpus, join=True,)
    print('pretrain finished!')

if __name__ == "__main__":
    main()
