import os
os.environ["DGLBACKEND"] = "pytorch"
import numpy as np
import random
import dgl
import torch
import time
import argparse
import networkx as nx

def perpare():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Flickr")
    parser.add_argument("--edgelimit", type=float, default="0.5")
    parser.add_argument("--T", type=float, default="800")
    parser.add_argument("--alpha", type=float, default="0.8")
    args = parser.parse_args()
    return args


def vertex_edge(nx_g, vid):
    in_edges = list(nx_g.in_edges(vid))
    out_edges = list(nx_g.out_edges(vid))
    edges = in_edges + out_edges
    return set(edges)

def vertex_neighbor(nx_g, vid):
    neighbor = set(nx_g.neighbors(vid))
    return neighbor

def heuristic_partition(g, args, delta):
    nx_g = dgl.to_networkx(g)
    start_time = time.time()


    nodes = set(nx_g.nodes())
    N = nx_g.number_of_nodes()
    E = nx_g.number_of_edges()
    average_degree = sum(nx_g.degree(v) for v in nx_g.nodes) / len(nx_g.nodes)
    candidate_vertices = [v for v in nx_g.nodes if abs(nx_g.degree(v) - average_degree) < 0.3 * average_degree]
    # candidate_vertices = [v for v in nx_g.nodes]
    first_seed_vertex = random.choice(candidate_vertices)
    print(nx_g.degree(first_seed_vertex))
    seed_vertex_set = set()
    seed_vertex_set.add(first_seed_vertex)
    expandable_vertex_set = vertex_neighbor(nx_g, first_seed_vertex)
    p0_vertex = expandable_vertex_set
    p0_vertex.update(seed_vertex_set)
    p0_edge_set = vertex_edge(nx_g, first_seed_vertex)
    limit = E * delta


    t = args.T
    while len(p0_edge_set) < limit:
        max_score = float('-inf')
        new_seed_vertex = None
        if p0_vertex - seed_vertex_set != 0:
            for vertex in (p0_vertex - seed_vertex_set):
                difference = vertex_neighbor(nx_g, vertex) - p0_vertex
                num_neighbor = len(vertex_neighbor(nx_g, vertex))
                score = -len(difference) + t * num_neighbor
                if score > max_score:
                    max_score = score
                    new_seed_vertex = vertex
        else:
            new_seed_vertex = random.choice(list(nodes - p0_vertex))
        seed_vertex_set.add(new_seed_vertex)
        new_seed_edge = vertex_edge(nx_g, new_seed_vertex)
        p0_edge_set.update(new_seed_edge)
        p0_vertex.update(vertex_neighbor(nx_g, new_seed_vertex))
        t = args.alpha * t


    p1_edge_set = set(nx_g.edges()) - p0_edge_set
    p1_vertex = set()
    for edge in p1_edge_set:
        p1_vertex.add(edge[0])
        p1_vertex.add(edge[1])
    end_time = time.time()
    part_time = end_time - start_time
    print("partition time:{0:.3f}s".format(part_time))
    # print(p0_edge_set)
    src0, dst0 = zip(*p0_edge_set)
    src0 = [int(v) for v in src0]
    dst0 = [int(v) for v in dst0]

    src1, dst1 = zip(*p1_edge_set)
    src1 = [int(v) for v in src1]
    dst1 = [int(v) for v in dst1]

    eid0 = g.edge_ids(src0, dst0)


    eid1 = g.edge_ids(src1, dst1)
    # print(set(eid0.numpy()))
    # print(set(eid1.numpy()))
    # print(set(eid0.numpy()) & set(eid1.numpy()))

    subgraph0 = dgl.edge_subgraph(g, eid0, relabel_nodes=True)
    # subgraph0 = dgl.remove_self_loop(subgraph0)
    # subgraph0 = dgl.add_self_loop(subgraph0)

    subgraph1 = dgl.edge_subgraph(g, eid1, relabel_nodes=True)
    # subgraph1 = dgl.remove_self_loop(subgraph1)
    # subgraph1 = dgl.add_self_loop(subgraph1)

    cut_vertex = p1_vertex & p0_vertex
    cut_vertex = {int(v) for v in cut_vertex}
    print(len(cut_vertex))
    # torch.save(cut_vertex, f'CASE/{args.dataset}_cutv.pth')

    dgl.save_graphs(f"NESA_2part/{args.dataset}_sg0_{delta}.bin", [subgraph0])
    dgl.save_graphs(f"NESA_2part/{args.dataset}_sg1_{delta}.bin", [subgraph1])


def main(delta):
    args = perpare()
    if args.dataset == "cora":
        dataset = dgl.load_graphs("self_graph/cora.bin")
    elif args.dataset == "Pubmed":
        dataset = dgl.load_graphs("self_graph/Pubmed.bin")
    elif args.dataset == "Flickr":
        dataset = dgl.load_graphs("self_graph/Flickr.bin")
    elif args.dataset == "cophy":
        dataset = dgl.load_graphs("self_graph/cophy.bin")
    elif args.dataset == "corafull":
        dataset = dgl.load_graphs("self_graph/corafull.bin")
    elif args.dataset == "Reddit":
        dataset = dgl.load_graphs("self_graph/Reddit.bin")
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))

    g = dataset[0][0]
    print(args.dataset)
    print(g)
    heuristic_partition(g, args, delta)

if __name__ == "__main__":
    main(0.25)
    main(0.2)
    main(0.7)
    main(0.75)


