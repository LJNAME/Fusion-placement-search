"""FlagScale 集成层。

本目录是项目中**唯一**允许知道 FlagScale / Megatron API 的地方。
其余所有子包（simulator / cost_model / search / runtime）都不允许 import megatron。
"""