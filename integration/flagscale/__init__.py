"""FlagScale 适配层：把 FlagScale/Megatron 的上下文翻译成本项目自己的数据结构。"""

from .context import HeteroExecutionContext

__all__ = ["HeteroExecutionContext"]