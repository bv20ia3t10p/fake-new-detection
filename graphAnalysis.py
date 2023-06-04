# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 20:50:29 2023

@author: bvtp1

"""
from collections import deque
from heapq import heappop, heappush
from itertools import count
import functools
import math
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
import community as community_louvain
from networkx.algorithms.shortest_paths.weighted import _weight_function

class network_analysis:
    def __init__(self, graph):
        self.G = to_networkx(graph,to_undirected=True)
    def community_layout(self,g, partition):
        pos_communities = self._position_communities(g, partition, scale=3.)
        pos_nodes = self._position_nodes(g, partition, scale=1.)
        pos = dict()
        for node in g.nodes():
            pos[node] = pos_communities[node] + pos_nodes[node]
    
        return pos
    def _position_communities(self,g, partition, **kwargs):
        between_community_edges = self._find_between_community_edges(g, partition)
        communities = set(partition.values())
        hypergraph = nx.DiGraph()
        hypergraph.add_nodes_from(communities)
        for (ci, cj), edges in between_community_edges.items():
            hypergraph.add_edge(ci, cj, weight=len(edges))
        pos_communities = nx.spring_layout(hypergraph, **kwargs)
        pos = dict()
        for node, community in partition.items():
            pos[node] = pos_communities[community]
        return pos
    def _find_between_community_edges(self,g, partition):
        edges = dict()
        for (ni, nj) in g.edges():
            ci = partition[ni]
            cj = partition[nj]
            if ci != cj:
                try:
                    edges[(ci, cj)] += [(ni, nj)]
                except KeyError:
                    edges[(ci, cj)] = [(ni, nj)]
    
        return edges
    
    def _position_nodes(self,g, partition, **kwargs):
        communities = dict()
        for node, community in partition.items():
            try:
                communities[community] += [node]
            except KeyError:
                communities[community] = [node]
    
        pos = dict()
        for ci, nodes in communities.items():
            subgraph = g.subgraph(nodes)
            pos_subgraph = nx.spring_layout(subgraph, **kwargs)
            pos.update(pos_subgraph)
    
        return pos
    def draw_network(self):
        g = self.G
        # compute the best partition
        plt.figure(figsize=(12,8))
        # draw the graph
        partition = community_louvain.best_partition(g)
        pos = self.community_layout(g, partition)
        nx.draw_networkx(g, pos, node_color=list(partition.values())); plt.show()
    def degree_centrality(self):
        G = self.G
        if len(G) <= 1:
            return {n: 1 for n in G}
    
        s = 1.0 / (len(G) - 1.0)
        centrality = {n: d * s for n, d in G.degree()}
        return centrality
    def eigenvector_centrality(self, max_iter=100, tol=1.0e-6, nstart=None, weight=None):
        G = self.G
        if len(G) == 0:
            raise nx.NetworkXPointlessConcept(
                "cannot compute centrality for the null graph"
            )
    # If no initial vector is provided, start with the all-ones vector.
        if nstart is None:
            nstart = {v: 1 for v in G}
        if all(v == 0 for v in nstart.values()):
            raise nx.NetworkXError("initial vector cannot have all zero values")
        # Normalize the initial vector so that each entry is in [0, 1]. This is
        # guaranteed to never have a divide-by-zero error by the previous line.
        nstart_sum = sum(nstart.values())
        x = {k: v / nstart_sum for k, v in nstart.items()}
        nnodes = G.number_of_nodes()
        # make up to max_iter iterations
        for _ in range(max_iter):
            xlast = x
            x = xlast.copy()  # Start with xlast times I to iterate with (A+I)
            # do the multiplication y^T = x^T A (left eigenvector)
            for n in x:
                for nbr in G[n]:
                    w = G[n][nbr].get(weight, 1) if weight else 1
                    x[nbr] += xlast[n] * w
            # Normalize the vector. The normalization denominator `norm`
            # should never be zero by the Perron--Frobenius
            # theorem. However, in case it is due to numerical error, we
            # assume the norm to be one instead.
            norm = math.hypot(*x.values()) or 1
            x = {k: v / norm for k, v in x.items()}
            # Check for convergence (in the L_1 norm).
            if sum(abs(x[n] - xlast[n]) for n in x) < nnodes * tol:
                return x
        raise nx.PowerIterationFailedConvergence(max_iter)
    def closeness_centrality(self, u=None, distance=None, wf_improved=True):
        G = self.G
        if G.is_directed():
            G = G.reverse()  # create a reversed graph view
    
        if distance is not None:
            # use Dijkstra's algorithm with specified attribute as edge weight
            path_length = functools.partial(
                nx.single_source_dijkstra_path_length, weight=distance
            )
        else:
            path_length = nx.single_source_shortest_path_length
    
        if u is None:
            nodes = G.nodes
        else:
            nodes = [u]
        closeness_dict = {}
        for n in nodes:
            sp = path_length(G, n)
            totsp = sum(sp.values())
            len_G = len(G)
            _closeness_centrality = 0.0
            if totsp > 0.0 and len_G > 1:
                _closeness_centrality = (len(sp) - 1.0) / totsp
                # normalize to number of nodes-1 in connected part
                if wf_improved:
                    s = (len(sp) - 1.0) / (len_G - 1)
                    _closeness_centrality *= s
            closeness_dict[n] = _closeness_centrality
        if u is not None:
            return closeness_dict[u]
        return closeness_dict
    def betweenness_centrality(self,
     k=None, normalized=True, weight=None, endpoints=False, seed=None):
        G = self.G
        betweenness = dict.fromkeys(G, 0.0)  # b[v]=0 for v in G
        if k is None:
            nodes = G
        else:
            nodes = seed.sample(list(G.nodes()), k)
        for s in nodes:
            # single source shortest paths
            if weight is None:  # use BFS
                S, P, sigma, _ = self._single_source_shortest_path_basic(G, s)
            else:  # use Dijkstra's algorithm
                S, P, sigma, _ = self._single_source_dijkstra_path_basic(G, s, weight)
            # accumulation
            if endpoints:
                betweenness, _ = self._accumulate_endpoints(betweenness, S, P, sigma, s)
            else:
                betweenness, _ = self._accumulate_basic(betweenness, S, P, sigma, s)
        # rescaling
        betweenness = self._rescale(
            betweenness,
            len(G),
            normalized=normalized,
            directed=G.is_directed(),
            k=k,
            endpoints=endpoints,
        )
        return betweenness
    def _single_source_shortest_path_basic(self,G, s):
        S = []
        P = {}
        for v in G:
            P[v] = []
        sigma = dict.fromkeys(G, 0.0)  # sigma[v]=0 for v in G
        D = {}
        sigma[s] = 1.0
        D[s] = 0
        Q = deque([s])
        while Q:  # use BFS to find shortest paths
            v = Q.popleft()
            S.append(v)
            Dv = D[v]
            sigmav = sigma[v]
            for w in G[v]:
                if w not in D:
                    Q.append(w)
                    D[w] = Dv + 1
                if D[w] == Dv + 1:  # this is a shortest path, count paths
                    sigma[w] += sigmav
                P[w].append(v)  # predecessors
        return S, P, sigma, D

    def _single_source_dijkstra_path_basic(self, G, s, weight):
        weight = _weight_function(G, weight)
        # modified from Eppstein
        S = []
        P = {}
        for v in G:
            P[v] = []
        sigma = dict.fromkeys(G, 0.0)  # sigma[v]=0 for v in G
        D = {}
        sigma[s] = 1.0
        push = heappush
        pop = heappop
        seen = {s: 0}
        c = count()
        Q = []  # use Q as heap with (distance,node id) tuples
        push(Q, (0, next(c), s, s))
        while Q:
            (dist, _, pred, v) = pop(Q)
            if v in D:
                continue  # already searched this node.
            sigma[v] += sigma[pred]  # count paths
            S.append(v)
            D[v] = dist
            for w, edgedata in G[v].items():
                vw_dist = dist + weight(v, w, edgedata)
                if w not in D and (w not in seen or vw_dist < seen[w]):
                    seen[w] = vw_dist
                    push(Q, (vw_dist, next(c), v, w))
                    sigma[w] = 0.0
                    P[w] = [v]
                elif vw_dist == seen[w]:  # handle equal paths
                    sigma[w] += sigma[v]
                    P[w].append(v)
        return S, P, sigma, D


    def _accumulate_basic(self,betweenness, S, P, sigma, s):
        delta = dict.fromkeys(S, 0)
        while S:
            w = S.pop()
            coeff = (1 + delta[w]) / sigma[w]
            for v in P[w]:
                delta[v] += sigma[v] * coeff
            if w != s:
                betweenness[w] += delta[w]
        return betweenness, delta
        
        
    def _accumulate_endpoints(self,betweenness, S, P, sigma, s):
        betweenness[s] += len(S) - 1
        delta = dict.fromkeys(S, 0)
        while S:
            w = S.pop()
            coeff = (1 + delta[w]) / sigma[w]
            for v in P[w]:
                delta[v] += sigma[v] * coeff
            if w != s:
                betweenness[w] += delta[w] + 1
        return betweenness, delta
        
        
    def _accumulate_edges(self,betweenness, S, P, sigma, s):
        delta = dict.fromkeys(S, 0)
        while S:
            w = S.pop()
            coeff = (1 + delta[w]) / sigma[w]
            for v in P[w]:
                c = sigma[v] * coeff
                if (v, w) not in betweenness:
                    betweenness[(w, v)] += c
                else:
                    betweenness[(v, w)] += c
                delta[v] += c
            if w != s:
                betweenness[w] += delta[w]
        return betweenness
    def _rescale(self,betweenness, n, normalized, directed=False, k=None, endpoints=False):
        if normalized:
            if endpoints:
                if n < 2:
                    scale = None  # no normalization
                else:
                    # Scale factor should include endpoint nodes
                    scale = 1 / (n * (n - 1))
            elif n <= 2:
                scale = None  # no normalization b=0 for all nodes
            else:
                scale = 1 / ((n - 1) * (n - 2))
        else:  # rescale by 2 for undirected graphs
            if not directed:
                scale = 0.5
            else:
                scale = None
        if scale is not None:
            if k is not None:
                scale = scale * n / k
            for v in betweenness:
                betweenness[v] *= scale
        return betweenness
    