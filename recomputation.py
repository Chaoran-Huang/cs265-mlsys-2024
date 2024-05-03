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


    def get_recomputation_nodes(self):
        return "imported"