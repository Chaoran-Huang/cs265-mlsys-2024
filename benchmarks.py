import importlib
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.fx as fx
from torchbenchmark.models import hf_Bert, resnet18, resnet50
from torchbenchmark.util.model import BenchmarkModel

import recomputation
from activation_checkpoint import activation_checkpointing
from statistics import mean
from graph_prof import GraphProfiler
from graph_tracer import SEPFunction, compile

model_names: List[str] = [
    "torchbenchmark.models.hf_Bert.Model",
    "torchbenchmark.models.resnet18.Model",
    "torchbenchmark.models.resnet50.Model",
]

actual_model_names: List[str] = [
    "hf_Bert",
    "resnet18",
    "resnet50",
]

model_batch_sizes: Dict[str, int] = {
    "torchbenchmark.models.hf_Bert.Model": 16,
    "torchbenchmark.models.resnet18.Model": 128,
    "torchbenchmark.models.resnet50.Model": 64,
}


class Experiment:
    def __init__(self, model_name: str, batch_size: int, extra_args=[]):
        pos = model_name.rfind(".")
        module = importlib.import_module(model_name[:pos])
        model_class = getattr(module, model_name[(pos + 1) :])

        model: BenchmarkModel = model_class(
            "train", "cuda", batch_size=batch_size, extra_args=extra_args
        )
        self.model: nn.Module = model.model
        self.model_type = type(model)

        self.batch_size = batch_size
        self.example_inputs = model.example_inputs

        if self.model_type == hf_Bert.Model:

            def bert_train_step(
                model: nn.Module, optim: optim.Optimizer, example_inputs: Any
            ):
                loss = model(**example_inputs).loss
                loss = SEPFunction.apply(loss)
                loss.backward()
                optim.step()
                optim.zero_grad()

            self.train_step = bert_train_step
            self.optimizer: optim.Optimizer = model.optimizer

        elif self.model_type in (resnet18.Model, resnet50.Model):
            self.loss_fn = model.loss_fn
            self.example_inputs = model.example_inputs[0]

            def resnet_train_step(
                model: nn.Module, optim: optim.Optimizer, example_inputs: Any
            ):
                output = model(example_inputs)
                target = torch.rand_like(output)
                loss = self.loss_fn(output, target)
                loss = SEPFunction.apply(loss)
                loss.backward()
                optim.step()
                optim.zero_grad()

            self.optimizer: optim.Optimizer = model.opt
            self.train_step = resnet_train_step

    def init_opt_states(self):
        for param in self.model.parameters():
            if param.requires_grad:
                param.grad = torch.rand_like(param)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def graph_transformation(self, gm: fx.GraphModule, args: Any) -> fx.GraphModule:
        print(gm.graph)
        warm_up_iters, profile_iters = 2, 3
        graph_profiler = GraphProfiler(gm)

        with torch.no_grad():
            for _ in range(warm_up_iters):
                graph_profiler.run(*args)
            graph_profiler.reset_stats()

            for _ in range(profile_iters):
                graph_profiler.run(*args)
            graph_profiler.aggregate_stats()
            graph_profiler.print_stats()

        recomputation_node = recomputation.Recomputation(
            node_info=graph_profiler.node_info, intermediate_nodes=graph_profiler.intermediate_nodes)

        peak_mem = max([mem.peak_total_mem for mem in graph_profiler.node_info.values()])
        recomps = recomputation_node.recomputation_policy(
            mem_limit=torch.cuda.get_device_properties(0).total_memory / 2,
            max_peak_memory=peak_mem)

        new_graph_module = activation_checkpointing(gm, graph_profiler.node_info, recomps)

        return new_graph_module

    def run(self):
        self.train_step(self.model, self.optimizer, self.example_inputs)
        print("Successful.")


# if __name__ == "__main__":
#     exp = Experiment(model_names[0], model_batch_sizes[model_names[0]])
#     exp.init_opt_states()
#     compiled_fn = compile(exp.train_step, exp.graph_transformation)
#     compiled_fn(exp.model, exp.optimizer, exp.example_inputs)

if __name__ == "__main__":

    for model_name in model_names:
        for batch_size in model_batch_sizes[model_name]:

            exp = Experiment(model_name, batch_size)
            exp.init_opt_states()
            compiled_fn = compile(exp.train_step, exp.graph_transformation)
            compiled_fn(exp.model, exp.optimizer, exp.example_inputs)
            torch.cuda.synchronize()

            num_iters = 3
            run_times = []
            torch.cuda.reset_peak_memory_stats()
            for _ in range(num_iters):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                compiled_fn(exp.model, exp.optimizer, exp.example_inputs)
                end_event.record()
                torch.cuda.synchronize()
                run_times.append(start_event.elapsed_time(end_event))
            run_time = mean(run_times)
            peak_memory = torch.cuda.max_memory_allocated()