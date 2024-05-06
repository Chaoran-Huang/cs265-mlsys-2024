import torch.fx as fx
from typing import Dict, Any, List, cast, Set
from graph_prof import NodeInfo


class Candidate:
    def __init__(self, node=None, memory_size=0, recomp_srcs=None, recomp_graph=None, recomp_cnt=0.0, recomp_time=0.0, total_recomp_time=0.0, recompute_ratio=0.0):
        self.node = node
        self.memory_size = memory_size
        self.recomp_srcs = recomp_srcs if recomp_srcs is not None else set()
        self.recomp_graph = recomp_graph
        self.recomp_cnt = recomp_cnt
        self.recomp_time = recomp_time
        self.total_recomp_time = total_recomp_time
        self.recompute_ratio = recompute_ratio


class Recomputation:
    def __init__(self, node_info, intermediate_nodes):
        self.node_info: Dict[fx.Node, NodeInfo] = node_info
        self.intermediate_nodes: List[fx.Node] = intermediate_nodes
        self.candidate_set: Set[Candidate] = {Candidate(node=n) for n in set(intermediate_nodes)}

    def recomputation_policy(self, mem_limit, max_peak_memory):

        mem_consumption = max_peak_memory
        self.initialization()
        recomps = set()

        while len(self.candidate_set) != 0:
            r_cand = self.max_recomp_candidate(self.candidate_set)
            recomps.add(r_cand)
            cand = r_cand
            self.candidate_set.remove(cand)
            recomp_cnt = self.update_existing_recomputatuions(cand, recomps)
            self.update_candidates(recomps, cand, recomp_cnt, self.candidate_set)
            mem_consumption -= cand.memory_size
            if (mem_consumption - mem_limit) <= 0:
                break
        return recomps

    def initialization(self):
        for cand in self.candidate_set:
            cand.recomp_srcs = self.find_srcs(cand.node, set())
            cand.recomp_time = sum([self.node_info[src].run_time for src in cand.recomp_srcs])
            cand.memory_size = self.node_info[cand.node].memory_size
            cand.total_recomp_time = cand.recomp_time
            cand.recompute_ratio = float(
                'inf') if cand.total_recomp_time == 0 else cand.memory_size / cand.total_recomp_time

    def max_recomp_candidate(self, candidate_set: Set[Candidate]):
        max_candidate = None
        for cand in candidate_set:
            if max_candidate is None:
                max_candidate = cand
            elif max_candidate.recompute_ratio < cand.recompute_ratio:
                max_candidate = cand
        return max_candidate

    def update_recompute_ratio(self, candidate_set: Set[Candidate]):
        for cand in candidate_set:
            cand.recompute_ratio = cand.memory_size / cand.total_recomp_time

    def update_existing_recomputatuions(self, cand, recomps):
        recomp_cnt = 1
        for rp in recomps:
            rp.recomp_srcs.remove(cand)
            rp.recomp_srcs.add(cand.recomp_srcs)
            rp.recomp_time += cand.recomp_time
            recomp_cnt += 1
        return recomp_cnt

    def update_candidates(self, recomps, t, recomp_cnt, candidates):
        for cand in candidates:
            if t in cand.recomp_srcs:
                cand.recomp_srcs.remove(t)
                cand.recomp_srcs.add(t.recomp_srcs)
                cand.total_recomp_time = cand.recomp_time
                for rp in recomps:
                    if cand in rp.recomp_srcs:
                        cand.total_recomp_time += cand.recomp_time

            if cand in t.recomp_srcs:
                cand.total_recomp_time = recomp_cnt * cand.recomp_time
        self.update_recompute_ratio(candidates)

    def find_srcs(self, node: fx.Node, srcs: Set[fx.Node]):
        # compute the source recursively
        input_nodes: List[fx.Node] = node.all_input_nodes
        for input_node in input_nodes:
            if (input_node.op == "placeholder") or (input_node in self.intermediate_nodes):
                srcs.add(input_node)
            else:
                self.find_srcs(input_node, srcs)
        return srcs
