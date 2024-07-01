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

def heuristic_partition(g, args, delta1, delta2):
    nx_g = dgl.to_networkx(g)
    start_time = time.time()


    nodes = set(nx_g.nodes())
    # N = nx_g.number_of_nodes()
    unassigned_edge_num = nx_g.number_of_edges()
    unassigned_edge = set(nx_g.edges())
    average_degree = sum(nx_g.degree(v) for v in nx_g.nodes) / len(nx_g.nodes)
    candidate_vertices = [v for v in nx_g.nodes if abs(nx_g.degree(v) - average_degree) < 0.3 * average_degree]
    # candidate_vertices = [v for v in nx_g.nodes]
    first_seed_vertex = random.choice(candidate_vertices)
    print(f'sg0 first seed degree: {nx_g.degree(first_seed_vertex)}')
    seed_vertex_set = set()
    seed_vertex_set.add(first_seed_vertex)
    expandable_vertex_set = vertex_neighbor(nx_g, first_seed_vertex)
    p0_vertex = expandable_vertex_set
    p0_vertex.update(seed_vertex_set)
    p0_edge_set = vertex_edge(nx_g, first_seed_vertex)
    limit1 = unassigned_edge_num * delta1

    t = args.T
    while len(p0_edge_set) < limit1:
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

    src0, dst0 = zip(*p0_edge_set)
    src0 = [int(v) for v in src0]
    dst0 = [int(v) for v in dst0]
    eid0 = g.edge_ids(src0, dst0)
    subgraph0 = dgl.edge_subgraph(g, eid0, relabel_nodes=True)

    unassigned_g = dgl.remove_edges(g, eid0)
    unassigned_nxg = dgl.to_networkx(unassigned_g)
    nodes = set(unassigned_nxg.nodes())
    unassigned_edge_num = unassigned_nxg.number_of_edges()
    unassigned_edge = set(unassigned_nxg.edges())
    average_degree = sum(unassigned_nxg.degree(v) for v in unassigned_nxg.nodes) / len(unassigned_nxg.nodes)
    candidate_vertices = [v for v in unassigned_nxg.nodes if abs(unassigned_nxg.degree(v) - average_degree) < 0.3 * average_degree]
    # candidate_vertices = [v for v in nx_g.nodes]
    first_seed_vertex = random.choice(candidate_vertices)
    print(f'sg1 first seed degree: {unassigned_nxg.degree(first_seed_vertex)}')
    seed_vertex_set = set()
    seed_vertex_set.add(first_seed_vertex)
    expandable_vertex_set = vertex_neighbor(unassigned_nxg, first_seed_vertex)
    p1_vertex = expandable_vertex_set
    p1_vertex.update(seed_vertex_set)
    p1_edge_set = vertex_edge(unassigned_nxg, first_seed_vertex)
    limit2 = unassigned_edge_num * delta2

    t = args.T
    while len(p1_edge_set) < limit2:
        max_score = float('-inf')
        new_seed_vertex = None
        if p1_vertex - seed_vertex_set != 0:
            for vertex in (p1_vertex - seed_vertex_set):
                difference = vertex_neighbor(unassigned_nxg, vertex) - p1_vertex
                num_neighbor = len(vertex_neighbor(unassigned_nxg, vertex))
                score = -len(difference) + t * num_neighbor
                if score > max_score:
                    max_score = score
                    new_seed_vertex = vertex
        else:
            new_seed_vertex = random.choice(list(nodes - p1_vertex))
        seed_vertex_set.add(new_seed_vertex)
        new_seed_edge = vertex_edge(unassigned_nxg, new_seed_vertex)
        p1_edge_set.update(new_seed_edge)
        p1_vertex.update(vertex_neighbor(unassigned_nxg, new_seed_vertex))
        t = args.alpha * t

    p2_edge_set = unassigned_edge - p1_edge_set
    p2_vertex = set()
    for edge in p2_edge_set:
        p2_vertex.add(edge[0])
        p2_vertex.add(edge[1])

    end_time = time.time()
    part_time = end_time - start_time
    print("partition time:{0:.3f}s".format(part_time))
    # print(p0_edge_set)
    src1, dst1 = zip(*p1_edge_set)
    src1 = [int(v) for v in src1]
    dst1 = [int(v) for v in dst1]

    src2, dst2 = zip(*p2_edge_set)
    src2 = [int(v) for v in src2]
    dst2 = [int(v) for v in dst2]

    eid1 = g.edge_ids(src1, dst1)

    eid2 = g.edge_ids(src2, dst2)


    subgraph1 = dgl.edge_subgraph(g, eid1, relabel_nodes=True)
    # subgraph0 = dgl.remove_self_loop(subgraph0)
    # subgraph0 = dgl.add_self_loop(subgraph0)

    subgraph2 = dgl.edge_subgraph(g, eid2, relabel_nodes=True)
    # subgraph1 = dgl.remove_self_loop(subgraph1)
    # subgraph1 = dgl.add_self_loop(subgraph1)

    # cut_vertex_01 = p1_vertex & p0_vertex
    # cut_vertex_01 = {int(v) for v in cut_vertex_01}
    # print(len(cut_vertex_01))
    # cut_vertex_02 = p2_vertex & p0_vertex
    # cut_vertex_02 = {int(v) for v in cut_vertex_02}
    # print(len(cut_vertex_02))
    # cut_vertex_12 = p1_vertex & p2_vertex
    # cut_vertex_12 = {int(v) for v in cut_vertex_12}
    # print(len(cut_vertex_12))
    # torch.save(cut_vertex, f'CASE/{args.dataset}_cutv.pth')
    print(f'num of edges in sg0: {subgraph0.num_edges()}')
    print(f'num of edges in sg1: {subgraph1.num_edges()}')
    print(f'num of edges in sg2: {subgraph2.num_edges()}')

    dgl.save_graphs(f"NESA_3part/{args.dataset}_sg0_{delta1}+{delta2}.bin", [subgraph0])
    dgl.save_graphs(f"NESA_3part/{args.dataset}_sg1_{delta1}+{delta2}.bin", [subgraph1])
    dgl.save_graphs(f"NESA_3part/{args.dataset}_sg2_{delta1}+{delta2}.bin", [subgraph2])


def main(delta1, delta2):
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
    heuristic_partition(g, args, delta1, delta2)

if __name__ == "__main__":
    main(0.33, 0.5)
    main(0.2, 0.5)
    main(0.17, 0.4)
    main(0.14, 0.33)
    main(0.1, 0.33)
    main(0.11, 0.25)