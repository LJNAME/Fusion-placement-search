"""Continuous input semantics without modifying historical frozen executors."""

from .gpu_holdout import CanonicalReferenceExecutor
from .reference import _torch_dtype


class ContinuousDecoderExecutor(CanonicalReferenceExecutor):
    """Root input is generated once; connected input nodes alias predecessors.

    The inherited run loop routes every predecessor to its consumer device
    before dispatch. There is no new transfer or scheduling policy here.
    """

    def __init__(self, placed, devices, seed=1234, templates=None, *, input_tensor=None):
        self.input_tensor = input_tensor
        super().__init__(placed, devices, seed, {} if templates is None else templates)
        if input_tensor is not None:
            self._check_input(self.graph.extra["input_node"], input_tensor)

    def _validate(self):
        super()._validate()
        root = self.graph.extra.get("input_node")
        if root not in self.graph.nodes or self.graph.nodes[root].op_type != "input":
            raise ValueError("continuous decoder requires an explicit root input")
        for node in self.graph.nodes.values():
            if node.op_type == "input":
                expected = 0 if node.id == root else 1
                if len(self.graph.predecessors(node.id)) != expected:
                    raise ValueError(f"input {node.id} requires {expected} predecessors")

    def _check_input(self, node_id, value):
        node = self.graph.nodes[node_id]
        expected = tuple(int(dim) for dim in node.extra["output_shape"])
        if tuple(value.shape) != expected:
            raise ValueError(f"input {node_id} shape {tuple(value.shape)} differs from {expected}")
        if value.dtype != _torch_dtype(str(node.extra["dtype"])):
            raise ValueError(f"input {node_id} dtype differs from graph metadata")

    def _input(self, node_id, device):
        if node_id != self.graph.extra["input_node"]:
            raise RuntimeError("only the root input may be generated")
        if self.input_tensor is None:
            return super()._input(node_id, device)
        key = (node_id, str(device))
        if key not in self._inputs:
            self._inputs[key] = self.input_tensor.detach().to(device)
        return self._inputs[key]

    def _execute_node(self, node_id, inputs, device):
        if self.graph.nodes[node_id].op_type != "input" or not inputs:
            return super()._execute_node(node_id, inputs, device)
        if len(inputs) != 1:
            raise ValueError("a connected decoder input requires exactly one tensor")
        value = inputs[0]
        self._check_input(node_id, value)
        if value.device != device:
            raise ValueError("decoder input was not routed to its consumer device")
        return value
