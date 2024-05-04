import torch
import torch.fx as fx
from typing import Dict, Any, List, cast

class Recomputation:
    def __init__(self, node_info, intermediate_nodes):
        self.node_info: Dict[fx.Node, NodeInfo] = node_info
        self.intermediate_nodes: List[fx.Node] = intermediate_nodes
        self.memory_size = 0 
        self.recomp_srcs = None
        self.recomp_graph = None
        self.recomp_cnt = None
        self.recomp_time = 0
        self.total_recomp_time = 0 
        self.recompute_ratio = 0


    def recomputation_policy(self, candidate_set, mem_limit, max_peak_memory):

        mem_comsumption = max_peak_memory
        initialization(candidate_set)
        recomps = set()

        while len(candidate_set) != 0:
            r_cand = max_recomp_candidate(candidate_set)
            recomps.add(r_cand)
            cand = r_cand
            candidates.remove(cand)
            recomp_cnt = update_recomps(cand, recomps)
            update_candidates(cand, recomp_cnt, candidates)
            mem_consumption -= cand.memory_size
            if (mem_consumption - mem_limit) <= 0:
                break

        