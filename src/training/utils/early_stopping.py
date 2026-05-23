"""
早停机制模块

监控训练过程中的验证困惑度(ppl)，当连续多轮ppl不再降低时提前终止训练。
"""

from typing import Dict, Any, Optional


class EarlyStopping:
    """
    早停控制器。

    监控验证ppl，当连续多轮未降低时触发停止信号。

    Args:
        patience: 容忍轮数，超过此值ppl未降低则触发早停。
        enabled: 是否启用早停。
    """

    def __init__(
        self,
        patience: int = 5,
        enabled: bool = True,
    ):
        self.patience = patience
        self.enabled = enabled

        self.best_ppl: Optional[float] = None
        self.counter = 0
        self.early_stop = False

    def step(self, val_ppl: float) -> bool:
        """
        执行早停检查。

        Args:
            val_ppl: 当前轮次的验证困惑度。

        Returns:
            bool: 是否触发早停。
        """
        if not self.enabled:
            return False

        if self.best_ppl is None:
            self.best_ppl = val_ppl
            return False

        if val_ppl < self.best_ppl:
            self.best_ppl = val_ppl
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop

    def state_dict(self) -> Dict[str, Any]:
        """返回早停状态，用于保存检查点。"""
        return {
            "best_ppl": self.best_ppl,
            "counter": self.counter,
            "early_stop": self.early_stop,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """从检查点恢复早停状态。"""
        self.best_ppl = state.get("best_ppl")
        self.counter = state.get("counter", 0)
        self.early_stop = state.get("early_stop", False)
