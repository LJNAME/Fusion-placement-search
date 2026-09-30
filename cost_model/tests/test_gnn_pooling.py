"""GNN 融合组池化 batch 偏移回归测试（需要 torch，服务器上运行）。

运行：python -m unittest discover -s cost_model/tests -v
"""

from __future__ import annotations

import unittest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "需要 torch（服务器上运行）")
class TestGNNPoolingBatchOffset(unittest.TestCase):
    """回归测试：batch 内不同样本的融合组池化必须按 b*G 偏移，不能串扰。

    背景：曾因池化时未加 batch 偏移，导致不同样本的同名融合组
    被累加到同一行，设备/组特征在样本之间互相污染，
    表现为 GNN val MAPE ~570%、Spearman ~0.2（对比 Ridge 0.99）。
    """

    def _fixed_model(self):
        from cost_model.gnn import CostGNN

        m = CostGNN(vocab_size=2, n_devs=2, hidden=4, num_layers=0)
        m.eval()
        with torch.no_grad():
            # node_enc：节点特征 dim = vocab(2) + 3 = 5；
            # [1,0,0,0,0] -> [1,0,0,0]，[0,1,0,0,0] -> [0,1,0,0]
            m.node_enc[0].weight.copy_(
                torch.tensor(
                    [
                        [1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ],
                    dtype=torch.float32,
                )
            )
            m.node_enc[0].bias.zero_()
            # group_mlp：只看 mean 池化后的节点嵌入（max 与组特征权重置零），第二层恒等
            group_feat_dim = 2 + 2  # n_devs + 2
            m.group_mlp[0].weight.copy_(
                torch.cat(
                    [
                        torch.eye(4),
                        torch.zeros(4, 4),
                        torch.zeros(4, group_feat_dim),
                    ],
                    dim=1,
                )
            )
            m.group_mlp[0].bias.zero_()
            # head：makespan 头 = dim0 - dim1（样本0 输出 1、样本1 输出 -1，可区分；
            # 若发生串扰两者都会变成 0，测试即可捕获），peak 头置零
            m.head_mk.weight.copy_(torch.tensor([[1.0, -1.0, 0.0, 0.0]]))
            m.head_mk.bias.zero_()
            m.head_pk.weight.zero_()
            m.head_pk.bias.zero_()
        return m

    def test_samples_do_not_crosstalk(self):
        from cost_model.gnn import build_sample, collate

        m = self._fixed_model()
        graph = {
            "name": "g",
            "nodes": [
                {"id": "a", "op_type": "op", "flops": 1.0, "out_bytes": 1.0, "param_bytes": 0.0},
                {"id": "b", "op_type": "op", "flops": 1.0, "out_bytes": 1.0, "param_bytes": 0.0},
            ],
            "edges": [{"src": "a", "dst": "b", "bytes": 1.0}],
        }
        vocab = ["op", "UNK"]
        plan0 = {"name": "p0", "groups": [{"node_ids": ["a"], "device": 0}, {"node_ids": ["b"], "device": 0}]}
        plan1 = {"name": "p1", "groups": [{"node_ids": ["a"], "device": 1}, {"node_ids": ["b"], "device": 1}]}
        rec0 = {"graph": graph, "plan": plan0, "makespan": 1.0, "peak_mem": {"0": 1.0}}
        rec1 = {"graph": graph, "plan": plan1, "makespan": 1.0, "peak_mem": {"0": 1.0}}
        s0 = build_sample(rec0, vocab, 2, include_target=False)
        s1 = build_sample(rec1, vocab, 2, include_target=False)
        # 两个样本的节点嵌入刻意不同：样本0 -> [1,0,0,0]，样本1 -> [0,1,0,0]
        s0["node_feats"] = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0]]
        )
        s1["node_feats"] = torch.tensor(
            [[0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]]
        )
        batch = collate([s0, s1])
        out = m(
            batch["node_feats"],
            batch["node_mask"],
            batch["edge_index"],
            batch["edge_attr"],
            batch["group_assign"],
            batch["group_feats"],
            batch["group_mask"],
            batch["dev_bytes"],
        )
        # 修复后：样本0 的组池化只来自自己的节点 -> makespan 头 1；
        # 样本1 -> -1；peak 头被置零为 0。串扰时两行都会变成 0
        expected = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        self.assertTrue(
            torch.allclose(out, expected, atol=1e-5),
            f"样本间组池化串扰: {out.tolist()}",
        )


if __name__ == "__main__":
    unittest.main()
