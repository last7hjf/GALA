import os
import random

import torch
os.environ['DGLBACKEND'] = 'pytorch'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import dgl
import time
import argparse
import random
def vertex_neighbor(nx_g, vid):
    neighbor = set(nx_g.neighbors(vid))
    return neighbor

def perpare():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="Flickr")
    parser.add_argument('--model', type=str, default="GCN")
    parser.add_argument("--edgelimit", type=str, default='0.25')
    args = parser.parse_args()
    return args

def main():
    args = perpare()
    start_time = time.time()
    print(f"dataset: {args.dataset}")
    print(f'gpu combo: {args.gpus}, edgelimit: {args.edgelimit}, model: {args.model}')
    # cut_vertex = torch.load(f"CASE/{args.dataset}_cutv.pth")
    # sg0 = dgl.load_graphs(f"NESA_2part/{args.dataset}_sg0.bin")[0][0]
    # sg1 = dgl.load_graphs(f"NESA_2part/{args.dataset}_sg1.bin")[0][0]
    sg0 = dgl.load_graphs(f"NESA_2part/{args.dataset}_sg0_{args.edgelimit}.bin")[0][0]
    sg1 = dgl.load_graphs(f"NESA_2part/{args.dataset}_sg1_{args.edgelimit}.bin")[0][0]
    cut_vertex = set(sg0.ndata[dgl.NID].numpy()) & set(sg1.ndata[dgl.NID].numpy())
    print(len(cut_vertex))
    g = dgl.load_graphs(f"self_graph/{args.dataset}.bin")[0][0]
    nx_g = dgl.to_networkx(g)
    sg0_miss_eids = set()
    sg1_miss_eids = set()

    for v in cut_vertex:
        original_in_edges = g.in_edges(v)
        predecessor_N = set(original_in_edges[0].tolist())
        sg0_vidx = sg0.nodes()[torch.nonzero(sg0.ndata[dgl.NID] == v).item()]
        sg0_in_edges = sg0.in_edges(sg0_vidx)
        sg0_predecessor_N = set()
        for pNidx in sg0_in_edges[0].tolist():
            pNid = sg0.ndata[dgl.NID][torch.nonzero(sg0.nodes() == pNidx).item()]
            sg0_predecessor_N.add(pNid.tolist())
        sg0_miss_pN = predecessor_N - sg0_predecessor_N
        # print(f'sg0 miss neighbor: {len(sg0_miss_pN)}')
        for pN in sg0_miss_pN:
            eid = g.edge_ids(pN, v)
            sg0_miss_eids.add(eid)

        sg1_vidx = sg1.nodes()[torch.nonzero(sg1.ndata[dgl.NID] == v).item()]
        sg1_in_edges = sg1.in_edges(sg1_vidx)
        sg1_predecessor_N = set()
        for pNidx in sg1_in_edges[0].tolist():
            pNid = sg1.ndata[dgl.NID][torch.nonzero(sg1.nodes() == pNidx).item()]
            sg1_predecessor_N.add(pNid.tolist())
        sg1_miss_pN = predecessor_N - sg1_predecessor_N
        # print(f'sg1 miss neighbor: {len(sg1_miss_pN)}')
        for pN in sg1_miss_pN:
            eid = g.edge_ids(pN, v)
            sg1_miss_eids.add(eid)

    #sg0_eid = list(g.edge_ids(sg0.all_edges()[0], sg0.all_edges()[1]).numpy())
    print(f'sg0 miss neighbor: {len(sg0_miss_eids)}')
    print(f'sg1 miss neighbor: {len(sg1_miss_eids)}')
    sg0_eid = list(sg0.edata[dgl.EID].numpy())
    sg1_eid = list(sg1.edata[dgl.EID].numpy())
    # sg0_num_edges = sg0.num_edges()

    # sg0_num_edges = sg0.num_edges()
    # sg0_num_add_edges = round(sg0_num_edges / 1.39 * 0.14 * 0.4)
    # sg0_misseI = sg0_miss_eids
    # sg0_add_eid = random.sample(sg0_misseI, sg0_num_add_edges)
    # print(f'sg0 add edges: {len(sg0_add_eid)}')

    sg1_num_edges = sg1.num_edges()
    sg1_num_add_edges = round(sg1_num_edges / 0.86 * 0.19 * 0.6)
    sg1_misseI = []
    for e in sg1_miss_eids:
        sg1_misseI.append(e)
    random.shuffle(sg1_misseI)
    sg1_add_eid = sg1_misseI[:sg1_num_add_edges]
    print(f'sg1 add edges: {len(sg1_add_eid)}')

    # final_sg0_eid = sg0_eid + sg0_add_eid
    # final_sg1_eid = sg1_eid
    final_sg0_eid = sg0_eid
    final_sg1_eid = sg1_eid + sg1_add_eid

    final_sg0 = dgl.edge_subgraph(g, final_sg0_eid, relabel_nodes=True)
    final_sg1 = dgl.edge_subgraph(g, final_sg1_eid, relabel_nodes=True)

    # sg0_NID = set(sg0.ndata[dgl.NID].numpy())
    # final_sg0_NID = set(final_sg0.ndata[dgl.NID].numpy())
    # sg0_add_NID = final_sg0_NID - sg0_NID
    # for add_v in sg0_add_NID:
    #     add_vidx = final_sg0.nodes()[torch.nonzero(final_sg0.ndata[dgl.NID] == add_v).item()]
    #     final_sg0.ndata['train_mask'][torch.nonzero(final_sg0.nodes() == add_vidx).item()] = False

    sg1_NID = set(sg1.ndata[dgl.NID].numpy())
    final_sg1_NID = set(final_sg1.ndata[dgl.NID].numpy())
    sg1_add_NID = final_sg1_NID - sg1_NID
    for add_v in sg1_add_NID:
        add_vidx = final_sg1.nodes()[torch.nonzero(final_sg1.ndata[dgl.NID] == add_v).item()]
        final_sg1.ndata['train_mask'][torch.nonzero(final_sg1.nodes() == add_vidx).item()] = False

    dgl.save_graphs(f"NESA_2part/{args.dataset}_{args.model}_absg0_combo{args.gpus}_{args.edgelimit}.bin", [final_sg0])
    dgl.save_graphs(f"NESA_2part/{args.dataset}_{args.model}_absg1_combo{args.gpus}_{args.edgelimit}.bin", [final_sg1])
    end_time = time.time()
    case_time = end_time - start_time
    print("CASE time:{0:.3f}s".format(case_time))
    print("CASE finished!")

if __name__ == "__main__":
    main()