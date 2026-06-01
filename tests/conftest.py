"""
ID: CONFTEST-001
Purpose: Shared pytest configuration for the ml-adamw-demo test suite.
Notes:
  - Filters the expected PyTorch UserWarning that fires when scheduler.step()
    is called without a preceding optimizer.step(). The scheduler factory tests
    intentionally call scheduler.step() in isolation to verify return types and
    that no exceptions are raised; they do not need a full training step.
"""
import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "filterwarnings",
        "ignore:Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`"
        ":UserWarning",
    )
