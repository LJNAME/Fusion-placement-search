# LightOP 第二批：依赖准备失败

回执 SHA256：`80cd23089ae77e9493f94abd9037ee773864739680f7da6b3d67f1812ae2c225`。冻结计划匹配，只有 preparation 目录，没有 measurements。

`resolve_host_dependencies.log` 两行报错：

```text
ERROR: Could not find a version that satisfies the requirement pytest==8.3.5 (from versions: none)
ERROR: No matching distribution found for pytest==8.3.5
```

失败发生在 pip dry-run，没有进入依赖安装、基准导入预检或 16 项 GPU 测量。before.json 与 after_failure.json 字节一致，包版本未变；Torch/Triton/LightOP/NumPy 保持原值。

pytest 8.3.5 已在官方 PyPI 确认存在。当前回执未保存 pip 环境变量或配置，也没有索引请求详情，不能仅凭 versions:none 判断是禁用索引、配置覆盖、网络访问还是候选筛选问题。

本次修复：pip 子进程使用 --isolated、禁用配置文件并移除继承的 PIP_* 设置；代理与 CA 等非 PIP 环境变量保留。新增离线 wheelhouse 模式，依赖解析和安装都使用本地文件，逐文件 SHA 校验，不访问包索引。原数值库保护及基准导入预检继续保留。

离线包按 Python3.10/Linux x86_64 构建，额外检查 Linux/Python3.10 的环境标记，包含 pytest 的 exceptiongroup/tomli 和 aiohttp 的 async-timeout 条件依赖。包不包含 Torch、Triton、LightOP、NumPy 的替换版本，厂商 vLLM/lmslim wheel 仍使用原固定 SHA。

原始回执保留在工作区 transfer/lightop-vendor-suite-received-20261103-02；项目记录 data/lightop_vendor_suite02_review/report.json。本次没有新增时延、正确性或正式 Profile 证据。
