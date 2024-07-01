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


    def forward(self, blocks, features):
        x = F.relu(self.conv1(blocks[0], features))
        x = self.conv2(blocks[1], x)
        return x

class GAT(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, num_heads):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feats, h_feats, num_heads, allow_zero_in_degree=True)
        self.conv2 = GATConv(h_feats * num_heads, num_classes, 1, allow_zero_in_degree=True)
    def forward(self, blocks, features):
        x = F.elu(self.conv1(blocks[0], features).flatten(1))
        x = self.conv2(blocks[1], x).mean(1)
        return x

@torch.no_grad()
def test(local_rank, model, g, device, nprocs):
    model.eval()
    ys = []
    y_hats = []
    test_idx = (g.ndata["test_mask"] == True).nonzero(as_tuple=True)[0]
    test_idx.to(device)
    sampler = dgl.dataloading.MultiLayerNeighborSampler([10, 10])
    dataloader = dgl.dataloading.DataLoader(
        g, test_idx, sampler, device=device, batch_size=1024, use_uva=True, use_ddp=True,
    )
    for input_nodes, output_nodes, blocks in dataloader:
        blocks = [b.to(device) for b in blocks]
        input_feat = blocks[0].srcdata["feat"]
        ys.append(blocks[-1].dstdata["label"])
        y_hats.append(model(blocks, input_feat))
    test_acc = MF.accuracy(torch.cat(y_hats), torch.cat(ys), task='multiclass', num_classes=g.ndata['label'].max().item()+1) / nprocs
    dist.reduce(tensor=test_acc, dst=0)
    dist.barrier()
    if local_rank == 0:
        print(f"test accuracy为{test_acc.item():.4f}")


def train(local_rank, device, g, model):
    train_nid = (g.ndata["train_mask"] == True).nonzero(as_tuple=True)[0]
    train_nid = train_nid.to(device)
    sampler = dgl.dataloading.MultiLayerNeighborSampler([10, 10])
    train_dataloader = dgl.dataloading.DataLoader(g, train_nid, sampler, device=device, batch_size=1024, shuffle=True, use_uva=True, use_ddp=True)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    train_start = time.time()
    for epoch in range(1, 24):
        model.train()
        total_loss = 0
        with Join([model]):
            for step, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
                blocks = [b.to(device) for b in blocks]
                input_feat = blocks[0].srcdata["feat"]
                output_label = blocks[-1].dstdata["label"]
                output_logit = model(blocks, input_feat)
                loss = criterion(output_logit, output_label)
                torch.cuda.synchronize()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        if local_rank == 0:
            if epoch % 10 == 0:
                print(f'Epoch: {epoch}, loss: {total_loss / (step + 1) :3f}')
    train_end = time.time()
    if local_rank == 0:
        print(f'end-to-end time: {train_end - train_start : .4f} s \n')

def run(local_rank, nprocs, devices, g, args):
    device = devices[local_rank]
    torch.cuda.set_device(device)
    print(f'device:{torch.cuda.get_device_name(device)}')
    dist.init_process_group(
        backend="nccl",  # Use NCCL backend for distributed GPU training
        init_method='env://',
        world_size=nprocs,
        rank=local_rank,
    )
    if args.model == "GCN":
        model = GCN(g.ndata['feat'].shape[1], 256, g.ndata['label'].max().item() + 1).to(device)
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
    train(local_rank, device, g, model)
    if local_rank == 0:
        # print(f"saving {args.model} model...")
        # torch.save(model.state_dict(), f'DGL_trainres/{args.model}_{args.dataset}_combo{args.gpus}.pth')
        print(f"testing {args.model} model...")
    test(local_rank, model, g, device, nprocs)
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=9)
    parser.add_argument("--dataset", type=str, default="Reddit")
    parser.add_argument('--model', type=str, default="GAT")
    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
    gpus = ['5,0', '5,4', '0,4', '5,1', '0,4,5', '0,2,5', '4,0,2', '5,4,6', '4,0,5,2', '4,0,6,2'][args.gpus]
    num_gpus = len(gpus.split(','))
    os.environ['WORLD_SIZE'] = str(num_gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    devices = []
    for i in range(num_gpus):
        devices.append(i)
    print(devices)
    print(f'{args.dataset}')
    g = dgl.load_graphs(f"self_graph/{args.dataset}.bin")[0][0]
    mp.spawn(run, args=(num_gpus, devices, g, args, ), nprocs=num_gpus, join=True, )
    print('finished!')

if __name__ == "__main__":
    main()
