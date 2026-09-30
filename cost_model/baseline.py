"""Ridge 线性回归基线（纯 numpy，双目标：log makespan / log 峰值显存）。"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np


class RidgeLogRegressor:
    """在标准化特征 + 中心化目标上的岭回归（对数空间训练）。"""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.mu_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.coef_: Optional[np.ndarray] = None
        self.ymean_: Optional[np.ndarray] = None

    def fit(self, X, y) -> "RidgeLogRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mu_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0) + 1e-9
        Xs = (X - self.mu_) / self.scale_
        self.ymean_ = y.mean(axis=0)
        ys = y - self.ymean_
        n, d = Xs.shape
        a = Xs.T @ Xs + self.alpha * np.eye(d)
        self.coef_ = np.linalg.solve(a, Xs.T @ ys)
        return self

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Xs = (X - self.mu_) / self.scale_
        return Xs @ self.coef_ + self.ymean_

    def save(self, path: str, extra: Optional[dict] = None) -> None:
        payload = {
            "alpha": self.alpha,
            "mu": self.mu_.tolist(),
            "scale": self.scale_.tolist(),
            "coef": self.coef_.tolist(),
            "ymean": self.ymean_.tolist(),
        }
        if extra:
            payload.update(extra)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "RidgeLogRegressor":
        with open(path, "r", encoding="utf-8") as f:
            p = json.load(f)
        m = cls(alpha=p["alpha"])
        m.mu_ = np.asarray(p["mu"], dtype=float)
        m.scale_ = np.asarray(p["scale"], dtype=float)
        m.coef_ = np.asarray(p["coef"], dtype=float)
        m.ymean_ = np.asarray(p["ymean"], dtype=float)
        return m
