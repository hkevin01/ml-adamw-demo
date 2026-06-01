"""
ID: TEST-SCHEDULER-001
Purpose: Unit tests for _build_scheduler factory in src/train.py.
Requirement: All four scheduler modes return the correct type and step-timing flag.
             Unsupported names raise ValueError.
Verification: pytest -v tests/test_scheduler.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

# Add src/ to path so imports resolve without package install.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from train import TrainConfig, _build_scheduler  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_optimizer(lr: float = 1e-3) -> torch.optim.Optimizer:
    """
    Purpose: Build a minimal AdamW optimizer for scheduler construction tests.
    Outputs: AdamW optimizer over a single Linear layer.
    """
    model = nn.Linear(4, 2)
    return torch.optim.AdamW(model.parameters(), lr=lr)


# ---------------------------------------------------------------------------
# Tests - none scheduler
# ---------------------------------------------------------------------------

class TestNoneScheduler:
    """Tests for scheduler='none' mode."""

    def test_returns_none(self) -> None:
        """No scheduler object should be returned."""
        opt = _make_optimizer()
        config = TrainConfig(scheduler="none")
        sched, step_on_batch = _build_scheduler(opt, config, steps_per_epoch=10)
        assert sched is None

    def test_step_on_batch_false(self) -> None:
        """Step-timing flag must be False when there is no scheduler."""
        opt = _make_optimizer()
        config = TrainConfig(scheduler="none")
        _, step_on_batch = _build_scheduler(opt, config, steps_per_epoch=10)
        assert step_on_batch is False


# ---------------------------------------------------------------------------
# Tests - cosine scheduler
# ---------------------------------------------------------------------------

class TestCosineScheduler:
    """Tests for scheduler='cosine' mode (CosineAnnealingLR)."""

    def test_returns_cosine_scheduler(self) -> None:
        """Should return a CosineAnnealingLR instance."""
        opt = _make_optimizer()
        config = TrainConfig(scheduler="cosine", scheduler_t_max=10)
        sched, _ = _build_scheduler(opt, config, steps_per_epoch=10)
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_step_on_batch_false(self) -> None:
        """Cosine is epoch-level; step_on_batch must be False."""
        opt = _make_optimizer()
        config = TrainConfig(scheduler="cosine", scheduler_t_max=10)
        _, step_on_batch = _build_scheduler(opt, config, steps_per_epoch=10)
        assert step_on_batch is False

    def test_eta_min_respected(self) -> None:
        """eta_min should be propagated to the scheduler."""
        opt = _make_optimizer(lr=1e-2)
        config = TrainConfig(scheduler="cosine", scheduler_t_max=5, min_lr=1e-6)
        sched, _ = _build_scheduler(opt, config, steps_per_epoch=5)
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)
        assert sched.eta_min == pytest.approx(1e-6)

    def test_lr_decreases_over_epochs(self) -> None:
        """LR should decrease after each step."""
        opt = _make_optimizer(lr=1e-2)
        config = TrainConfig(scheduler="cosine", scheduler_t_max=10, min_lr=0.0)
        sched, _ = _build_scheduler(opt, config, steps_per_epoch=10)
        initial_lr = opt.param_groups[0]["lr"]
        sched.step()
        after_step_lr = opt.param_groups[0]["lr"]
        assert after_step_lr < initial_lr


# ---------------------------------------------------------------------------
# Tests - onecycle scheduler
# ---------------------------------------------------------------------------

class TestOneCycleScheduler:
    """Tests for scheduler='onecycle' mode (OneCycleLR)."""

    def test_returns_onecycle_scheduler(self) -> None:
        """Should return an OneCycleLR instance."""
        opt = _make_optimizer()
        config = TrainConfig(scheduler="onecycle", epochs=5)
        sched, _ = _build_scheduler(opt, config, steps_per_epoch=10)
        assert isinstance(sched, torch.optim.lr_scheduler.OneCycleLR)

    def test_step_on_batch_true(self) -> None:
        """OneCycleLR requires per-batch stepping; flag must be True."""
        opt = _make_optimizer()
        config = TrainConfig(scheduler="onecycle", epochs=5)
        _, step_on_batch = _build_scheduler(opt, config, steps_per_epoch=10)
        assert step_on_batch is True

    def test_total_steps_matches_epochs_times_steps(self) -> None:
        """OneCycleLR total_steps = epochs * steps_per_epoch."""
        opt = _make_optimizer()
        config = TrainConfig(scheduler="onecycle", epochs=4)
        sched, _ = _build_scheduler(opt, config, steps_per_epoch=8)
        assert isinstance(sched, torch.optim.lr_scheduler.OneCycleLR)
        assert sched.total_steps == 4 * 8

    def test_can_step_total_steps_times(self) -> None:
        """Stepping exactly total_steps times must not raise."""
        opt = _make_optimizer(lr=1e-3)
        epochs, spe = 3, 5
        config = TrainConfig(scheduler="onecycle", epochs=epochs, lr=1e-3)
        sched, _ = _build_scheduler(opt, config, steps_per_epoch=spe)
        for _ in range(epochs * spe):
            sched.step()  # must not raise


# ---------------------------------------------------------------------------
# Tests - linear scheduler
# ---------------------------------------------------------------------------

class TestLinearScheduler:
    """Tests for scheduler='linear' mode (LinearLR)."""

    def test_returns_linear_scheduler(self) -> None:
        """Should return a LinearLR instance."""
        opt = _make_optimizer()
        config = TrainConfig(scheduler="linear", epochs=10, min_lr=1e-5)
        sched, _ = _build_scheduler(opt, config, steps_per_epoch=10)
        assert isinstance(sched, torch.optim.lr_scheduler.LinearLR)

    def test_step_on_batch_false(self) -> None:
        """LinearLR is epoch-level; step_on_batch must be False."""
        opt = _make_optimizer()
        config = TrainConfig(scheduler="linear", epochs=10, min_lr=1e-5)
        _, step_on_batch = _build_scheduler(opt, config, steps_per_epoch=10)
        assert step_on_batch is False

    def test_lr_strictly_decreasing(self) -> None:
        """LR must not increase during linear decay."""
        opt = _make_optimizer(lr=1e-2)
        config = TrainConfig(scheduler="linear", epochs=5, min_lr=1e-4, lr=1e-2)
        sched, _ = _build_scheduler(opt, config, steps_per_epoch=5)
        lrs = []
        for _ in range(5):
            sched.step()
            lrs.append(opt.param_groups[0]["lr"])
        assert all(lrs[i] >= lrs[i + 1] for i in range(len(lrs) - 1))


# ---------------------------------------------------------------------------
# Tests - error handling
# ---------------------------------------------------------------------------

class TestSchedulerErrorHandling:
    """Tests for unsupported and edge-case scheduler names."""

    def test_unsupported_name_raises_value_error(self) -> None:
        """An unknown scheduler name must raise ValueError."""
        opt = _make_optimizer()
        config = TrainConfig(scheduler="sgdr")
        with pytest.raises(ValueError, match="Unsupported scheduler"):
            _build_scheduler(opt, config, steps_per_epoch=10)

    def test_empty_name_raises_value_error(self) -> None:
        """An empty scheduler name must raise ValueError."""
        opt = _make_optimizer()
        config = TrainConfig(scheduler="")
        with pytest.raises(ValueError):
            _build_scheduler(opt, config, steps_per_epoch=10)

    @pytest.mark.parametrize("name", ["COSINE", "Cosine", "NONE", "OneCycle"])
    def test_case_insensitive_names_accepted(self, name: str) -> None:
        """Scheduler names must be case-insensitive."""
        opt = _make_optimizer()
        config = TrainConfig(scheduler=name, epochs=5)
        sched, _ = _build_scheduler(opt, config, steps_per_epoch=5)
        # Should not raise; return type varies by name


# ---------------------------------------------------------------------------
# Tests - boundary inputs
# ---------------------------------------------------------------------------

class TestSchedulerBoundaryInputs:
    """Tests for edge-case step counts and epoch values."""

    def test_cosine_t_max_one(self) -> None:
        """T_max=1 is a valid edge case and must not raise."""
        opt = _make_optimizer()
        config = TrainConfig(scheduler="cosine", scheduler_t_max=1)
        sched, _ = _build_scheduler(opt, config, steps_per_epoch=1)
        sched.step()  # must not raise

    def test_onecycle_single_step(self) -> None:
        """One epoch with one step per epoch is the minimum valid onecycle run."""
        opt = _make_optimizer(lr=1e-3)
        config = TrainConfig(scheduler="onecycle", epochs=1, lr=1e-3)
        sched, _ = _build_scheduler(opt, config, steps_per_epoch=1)
        sched.step()  # must not raise
