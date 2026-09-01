import json

import pytest

from ml_tools import (
    ToolInputError,
    compare_models,
    load_dataset_summary,
    train_pytorch_classifier,
    train_sklearn_model,
)


def test_dataset_summary_is_structured() -> None:
    result = json.loads(load_dataset_summary("iris"))
    assert result["ok"] is True
    assert result["n_samples"] == 150
    assert result["n_features"] == 4
    assert result["missing_values"] == 0


def test_sklearn_training_has_cv_metrics() -> None:
    result = json.loads(train_sklearn_model("iris", "decision_tree", cv=3))
    assert result["ok"] is True
    assert result["cv_folds"] == 3
    assert 0.0 <= result["test_accuracy"] <= 1.0
    assert result["cv_std_accuracy"] >= 0.0


def test_invalid_dropout_is_recoverable() -> None:
    with pytest.raises(ToolInputError, match="dropout"):
        train_pytorch_classifier("iris", dropout=1.5, epochs=1)


def test_shape_mismatch_is_recoverable() -> None:
    with pytest.raises(ToolInputError, match="shape mismatch"):
        train_pytorch_classifier("iris", input_dim_override=999, epochs=1)


def test_comparison_contains_markdown_table() -> None:
    result = json.loads(
        compare_models(["iris"], ["decision_tree", "logistic_regression"], cv=3)
    )
    assert result["ok"] is True
    assert len(result["rows"]) == 2
    assert "| Dataset | Model |" in result["markdown_table"]

