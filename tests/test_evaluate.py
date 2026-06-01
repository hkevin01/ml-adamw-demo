"""
ID: TEST-EVALUATE-001
Purpose: Unit tests for validate() and accuracy_from_logits() in src/evaluate.py
         and src/utils.py.
Requirement: validate() returns correct (loss, acc) scalars; accuracy_from_logits
             handles perfect, zero, and partial predictions; model is restored to
             train mode after validation.
Verification: pytest -v tests/test_evaluate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluate import validate          # noqa: E402
from utils import accuracy_from_logits  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _tiny_loader(n: int = 32, n_features: int = 8, n_classes: int = 3,
                 batch_size: int = 16) -> DataLoader:
    """
    Purpose: Build a deterministic tiny DataLoader for test isolation.
    Outputs: DataLoader with (float32 features, long labels).
    """
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, n_features, generator=g)
    y = torch.randint(0, n_classes, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def _identity_model(n_features: int = 8, n_classes: int = 3) -> nn.Module:
    """
    Purpose: A single Linear layer with predictable output for test assertions.
    """
    m = nn.Linear(n_features, n_classes, bias=False)
    nn.init.zeros_(m.weight)
    return m


# ---------------------------------------------------------------------------
# Tests - accuracy_from_logits
# ---------------------------------------------------------------------------

class TestAccuracyFromLogits:
    """Tests for the accuracy_from_logits utility."""

    def test_perfect_accuracy(self) -> None:
        """When argmax matches all targets, accuracy should be 1.0."""
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        targets = torch.tensor([0, 1])
        assert accuracy_from_logits(logits, targets) == pytest.approx(1.0)

    def test_zero_accuracy(self) -> None:
        """When argmax matches no targets, accuracy should be 0.0."""
        logits = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
        targets = torch.tensor([1, 1])
        assert accuracy_from_logits(logits, targets) == pytest.approx(0.0)

    def test_partial_accuracy(self) -> None:
        """Half-correct predictions should return 0.5."""
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [2.0, 0.0]])
        targets = torch.tensor([0, 1, 1, 1])
        assert accuracy_from_logits(logits, targets) == pytest.approx(0.5)

    def test_returns_python_float(self) -> None:
        """Return type must be a Python float, not a tensor."""
        logits = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))
        result = accuracy_from_logits(logits, targets)
        assert isinstance(result, float)

    def test_single_sample(self) -> None:
        """Single-sample batch must not raise and must return valid accuracy."""
        logits = torch.tensor([[0.0, 1.0, 0.0]])
        targets = torch.tensor([1])
        acc = accuracy_from_logits(logits, targets)
        assert acc == pytest.approx(1.0)

    def test_many_classes(self) -> None:
        """Should work correctly with 10+ classes."""
        n_classes = 10
        n = 50
        logits = torch.eye(n_classes).repeat(n // n_classes, 1)
        targets = torch.arange(n_classes).repeat(n // n_classes)
        acc = accuracy_from_logits(logits, targets)
        assert acc == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tests - validate function
# ---------------------------------------------------------------------------

class TestValidate:
    """Tests for the validate() function in evaluate.py."""

    def test_returns_two_floats(self) -> None:
        """validate() must return a tuple of two Python floats."""
        loader = _tiny_loader()
        model = _identity_model()
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")
        result = validate(model, loader, criterion, device, use_amp=False)
        assert isinstance(result, tuple)
        assert len(result) == 2
        loss, acc = result
        assert isinstance(loss, float)
        assert isinstance(acc, float)

    def test_loss_is_non_negative(self) -> None:
        """Cross-entropy loss must always be non-negative."""
        loader = _tiny_loader()
        model = _identity_model()
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")
        loss, _ = validate(model, loader, criterion, device, use_amp=False)
        assert loss >= 0.0

    def test_accuracy_in_valid_range(self) -> None:
        """Accuracy must be in [0.0, 1.0]."""
        loader = _tiny_loader()
        model = _identity_model()
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")
        _, acc = validate(model, loader, criterion, device, use_amp=False)
        assert 0.0 <= acc <= 1.0

    def test_model_restored_to_train_mode(self) -> None:
        """
        Postcondition: model must be back in training mode after validate().
        Failure mode: forgetting model.train() at the end of validate().
        """
        loader = _tiny_loader()
        model = _identity_model()
        model.train()
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")
        validate(model, loader, criterion, device, use_amp=False)
        assert model.training

    def test_no_gradients_computed(self) -> None:
        """validate() must not accumulate gradients on model parameters."""
        loader = _tiny_loader()
        model = _identity_model()
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")
        validate(model, loader, criterion, device, use_amp=False)
        for param in model.parameters():
            assert param.grad is None

    def test_perfect_model_high_accuracy(self) -> None:
        """
        A model that always outputs the correct class should return acc near 1.0.
        We construct a dataset where class 0 always has the highest logit.
        """
        n, n_features, n_classes = 64, 4, 3
        # All labels are class 0
        x = torch.randn(n, n_features)
        y = torch.zeros(n, dtype=torch.long)
        loader = DataLoader(TensorDataset(x, y), batch_size=32)

        # Model that always scores class 0 highest
        model = nn.Linear(n_features, n_classes, bias=True)
        with torch.no_grad():
            nn.init.zeros_(model.weight)
            model.bias.data = torch.tensor([10.0, 0.0, 0.0])

        criterion = nn.CrossEntropyLoss()
        _, acc = validate(model, loader, criterion, torch.device("cpu"), use_amp=False)
        assert acc == pytest.approx(1.0)

    def test_empty_loader_returns_zero_zero(self) -> None:
        """An empty DataLoader must return (0.0, 0.0) without raising."""
        empty_loader = DataLoader(TensorDataset(
            torch.empty(0, 8), torch.empty(0, dtype=torch.long)
        ), batch_size=16)
        model = _identity_model()
        criterion = nn.CrossEntropyLoss()
        result = validate(model, empty_loader, criterion, torch.device("cpu"), use_amp=False)
        assert result == (0.0, 0.0)

    def test_deterministic_output(self) -> None:
        """Same model and loader must produce identical results on repeated calls."""
        loader = _tiny_loader()
        model = _identity_model()
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")
        r1 = validate(model, loader, criterion, device, use_amp=False)
        r2 = validate(model, loader, criterion, device, use_amp=False)
        assert r1 == r2
