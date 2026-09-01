"""Machine-learning tools exposed to the local ReAct agent.

The module keeps tool outputs JSON serializable so an Ollama-hosted model can
read observations reliably. All experiments use deterministic seeds and
stratified validation where possible.
"""

from __future__ import annotations

import json
import math
import random
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
DATASETS: dict[str, Callable[..., Any]] = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
}


class ToolInputError(ValueError):
    """Raised when an agent supplies a recoverable invalid tool parameter."""


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, allow_nan=False)


def _seed_everything(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dataset(name: str):
    clean_name = str(name).lower().strip()
    if clean_name not in DATASETS:
        raise ToolInputError(
            f"Unknown dataset '{clean_name}'. Valid options: {sorted(DATASETS)}"
        )
    return clean_name, DATASETS[clean_name]()


def _validate_test_size(test_size: float) -> float:
    try:
        value = float(test_size)
    except (TypeError, ValueError) as exc:
        raise ToolInputError("test_size must be a decimal number") from exc
    if not 0.1 <= value <= 0.5:
        raise ToolInputError("test_size must be between 0.1 and 0.5")
    return value


def _safe_cv(y: np.ndarray, requested: int) -> StratifiedKFold:
    try:
        requested = int(requested)
    except (TypeError, ValueError) as exc:
        raise ToolInputError("cv must be an integer") from exc
    min_class_count = int(np.bincount(np.asarray(y, dtype=int)).min())
    folds = min(requested, min_class_count)
    if folds < 2:
        raise ToolInputError("At least two samples per class are required for CV")
    return StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=RANDOM_STATE
    )


def load_dataset_summary(dataset_name: str) -> str:
    """Return dimensions, class balance, missingness, and feature names."""
    name, data = _dataset(dataset_name)
    frame = pd.DataFrame(data.data, columns=data.feature_names)
    class_values, class_counts = np.unique(data.target, return_counts=True)
    return _json(
        {
            "ok": True,
            "tool": "load_dataset_summary",
            "dataset": name,
            "n_samples": int(frame.shape[0]),
            "n_features": int(frame.shape[1]),
            "feature_names": [str(item) for item in data.feature_names],
            "classes": [str(item) for item in class_values],
            "class_distribution": {
                str(label): int(count)
                for label, count in zip(class_values, class_counts)
            },
            "missing_values": int(frame.isna().sum().sum()),
        }
    )


def _sklearn_estimator(model_type: str) -> BaseEstimator:
    model = str(model_type).lower().strip()
    if model == "decision_tree":
        return DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE)
    if model == "logistic_regression":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
                ),
            ]
        )
    if model == "random_forest":
        return RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        )
    if model == "svc":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale")),
            ]
        )
    raise ToolInputError(
        "Unsupported model. Valid options: decision_tree, logistic_regression, "
        "random_forest, svc"
    )


def train_sklearn_model(
    dataset_name: str,
    model_type: str,
    test_size: float = 0.2,
    cv: int = 5,
) -> str:
    """Train and evaluate a classical classifier with stratified CV."""
    _seed_everything()
    name, data = _dataset(dataset_name)
    test_size = _validate_test_size(test_size)
    estimator = _sklearn_estimator(model_type)
    cv_splitter = _safe_cv(data.target, cv)

    x_train, x_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=data.target,
    )
    started = time.perf_counter()
    estimator.fit(x_train, y_train)
    predictions = estimator.predict(x_test)
    scores = cross_validate(
        estimator,
        data.data,
        data.target,
        cv=cv_splitter,
        scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
        n_jobs=-1,
    )
    elapsed = time.perf_counter() - started

    return _json(
        {
            "ok": True,
            "tool": "train_sklearn_model",
            "dataset": name,
            "model": str(model_type).lower().strip(),
            "test_size": test_size,
            "test_accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "test_macro_f1": round(float(f1_score(y_test, predictions, average="macro")), 4),
            "cv_folds": int(cv_splitter.n_splits),
            "cv_mean_accuracy": round(float(scores["test_accuracy"].mean()), 4),
            "cv_std_accuracy": round(float(scores["test_accuracy"].std()), 4),
            "cv_mean_macro_f1": round(float(scores["test_macro_f1"].mean()), 4),
            "elapsed_seconds": round(elapsed, 4),
        }
    )


def tune_hyperparameters(
    dataset_name: str,
    model_type: str,
    search_type: str = "grid",
    cv: int = 5,
    test_size: float = 0.2,
) -> str:
    """Tune SVC or Decision Tree parameters using Grid/RandomizedSearchCV."""
    _seed_everything()
    name, data = _dataset(dataset_name)
    model_name = str(model_type).lower().strip()
    search_name = str(search_type).lower().strip()
    test_size = _validate_test_size(test_size)
    cv_splitter = _safe_cv(data.target, cv)

    if model_name == "svc":
        estimator: BaseEstimator = Pipeline(
            [("scale", StandardScaler()), ("classifier", SVC())]
        )
        parameters = {
            "classifier__C": [0.1, 1.0, 10.0, 100.0],
            "classifier__kernel": ["linear", "rbf"],
            "classifier__gamma": ["scale", "auto"],
        }
    elif model_name == "decision_tree":
        estimator = DecisionTreeClassifier(random_state=RANDOM_STATE)
        parameters = {
            "max_depth": [None, 3, 5, 8],
            "min_samples_split": [2, 5, 10],
            "criterion": ["gini", "entropy", "log_loss"],
        }
    else:
        raise ToolInputError("model_type must be 'svc' or 'decision_tree'")

    x_train, x_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=data.target,
    )
    if search_name == "grid":
        search = GridSearchCV(
            estimator,
            parameters,
            cv=cv_splitter,
            scoring="accuracy",
            n_jobs=-1,
            refit=True,
        )
    elif search_name == "randomized":
        search = RandomizedSearchCV(
            estimator,
            parameters,
            n_iter=12,
            cv=cv_splitter,
            scoring="accuracy",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            refit=True,
        )
    else:
        raise ToolInputError("search_type must be 'grid' or 'randomized'")

    started = time.perf_counter()
    search.fit(x_train, y_train)
    predictions = search.predict(x_test)
    elapsed = time.perf_counter() - started
    best_std = float(search.cv_results_["std_test_score"][search.best_index_])

    return _json(
        {
            "ok": True,
            "tool": "tune_hyperparameters",
            "dataset": name,
            "model": model_name,
            "search_type": search_name,
            "candidates_evaluated": int(len(search.cv_results_["params"])),
            "best_parameters": search.best_params_,
            "best_cv_accuracy": round(float(search.best_score_), 4),
            "best_cv_std": round(best_std, 4),
            "test_accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "test_macro_f1": round(float(f1_score(y_test, predictions, average="macro")), 4),
            "elapsed_seconds": round(elapsed, 4),
        }
    )


def select_or_reduce_features(
    dataset_name: str,
    method: str,
    n_components: float | int = 2,
    n_features_to_select: int = 2,
    direction: str = "forward",
    cv: int = 5,
) -> str:
    """Evaluate PCA or sequential feature selection in a leakage-safe pipeline."""
    _seed_everything()
    name, data = _dataset(dataset_name)
    method_name = str(method).lower().strip()
    cv_splitter = _safe_cv(data.target, cv)

    if method_name == "pca":
        try:
            component_value: float | int = float(n_components)
            if component_value >= 1 and component_value.is_integer():
                component_value = int(component_value)
        except (TypeError, ValueError) as exc:
            raise ToolInputError("n_components must be an integer or variance ratio") from exc
        if isinstance(component_value, int):
            if not 1 <= component_value <= data.data.shape[1]:
                raise ToolInputError(
                    f"n_components must be 1..{data.data.shape[1]} for {name}"
                )
        elif not 0.0 < component_value < 1.0:
            raise ToolInputError("A float n_components must be between 0 and 1")

        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                ("transform", PCA(n_components=component_value, random_state=RANDOM_STATE)),
                (
                    "classifier",
                    LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
                ),
            ]
        )
        scores = cross_validate(
            pipeline,
            data.data,
            data.target,
            cv=cv_splitter,
            scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
            n_jobs=-1,
        )
        pipeline.fit(data.data, data.target)
        pca: PCA = pipeline.named_steps["transform"]
        return _json(
            {
                "ok": True,
                "tool": "select_or_reduce_features",
                "dataset": name,
                "method": "pca",
                "selected_dimensions": int(pca.n_components_),
                "explained_variance_ratio_sum": round(
                    float(pca.explained_variance_ratio_.sum()), 4
                ),
                "cv_mean_accuracy": round(float(scores["test_accuracy"].mean()), 4),
                "cv_std_accuracy": round(float(scores["test_accuracy"].std()), 4),
                "cv_mean_macro_f1": round(float(scores["test_macro_f1"].mean()), 4),
            }
        )

    if method_name == "sequential":
        try:
            selected_count = int(n_features_to_select)
        except (TypeError, ValueError) as exc:
            raise ToolInputError("n_features_to_select must be an integer") from exc
        if not 1 <= selected_count < data.data.shape[1]:
            raise ToolInputError(
                f"n_features_to_select must be 1..{data.data.shape[1] - 1} for {name}"
            )
        direction_name = str(direction).lower().strip()
        if direction_name not in {"forward", "backward"}:
            raise ToolInputError("direction must be 'forward' or 'backward'")

        selector = SequentialFeatureSelector(
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            n_features_to_select=selected_count,
            direction=direction_name,
            scoring="accuracy",
            cv=cv_splitter,
            n_jobs=-1,
        )
        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                ("selector", selector),
                (
                    "classifier",
                    LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
                ),
            ]
        )
        scores = cross_validate(
            pipeline,
            data.data,
            data.target,
            cv=cv_splitter,
            scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
            n_jobs=-1,
        )
        pipeline.fit(data.data, data.target)
        mask = pipeline.named_steps["selector"].get_support()
        selected_names = [
            str(feature) for feature, selected in zip(data.feature_names, mask) if selected
        ]
        return _json(
            {
                "ok": True,
                "tool": "select_or_reduce_features",
                "dataset": name,
                "method": "sequential",
                "direction": direction_name,
                "selected_features": selected_names,
                "cv_mean_accuracy": round(float(scores["test_accuracy"].mean()), 4),
                "cv_std_accuracy": round(float(scores["test_accuracy"].std()), 4),
                "cv_mean_macro_f1": round(float(scores["test_macro_f1"].mean()), 4),
            }
        )

    raise ToolInputError("method must be 'pca' or 'sequential'")


class RegularizedMLP(nn.Module):
    """Configurable MLP with BatchNorm and Dropout regularization."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float,
        batch_norm: bool,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers.append(nn.Linear(previous, width))
            if batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous = width
        layers.append(nn.Linear(previous, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def train_pytorch_classifier(
    dataset_name: str,
    hidden_dims: list[int] | None = None,
    dropout: float = 0.3,
    batch_norm: bool = True,
    scheduler: str = "step",
    epochs: int = 60,
    lr: float = 0.01,
    weight_decay: float = 0.0001,
    test_size: float = 0.2,
    device: str = "auto",
    input_dim_override: int | None = None,
) -> str:
    """Train a regularized PyTorch classifier with scheduler and NaN checks.

    ``input_dim_override`` exists only to create a controlled shape-mismatch
    recovery trace. Normal calls should leave it as null/None.
    """
    _seed_everything()
    name, data = _dataset(dataset_name)
    test_size = _validate_test_size(test_size)
    hidden_dims = [64, 32] if hidden_dims is None else hidden_dims
    if not isinstance(hidden_dims, list) or not hidden_dims:
        raise ToolInputError("hidden_dims must be a non-empty JSON list, e.g. [64, 32]")
    try:
        hidden_dims = [int(width) for width in hidden_dims]
    except (TypeError, ValueError) as exc:
        raise ToolInputError("Every hidden dimension must be an integer") from exc
    if any(width < 2 or width > 1024 for width in hidden_dims):
        raise ToolInputError("Each hidden dimension must be between 2 and 1024")

    try:
        dropout = float(dropout)
        lr = float(lr)
        weight_decay = float(weight_decay)
        epochs = int(epochs)
    except (TypeError, ValueError) as exc:
        raise ToolInputError("dropout, lr, weight_decay, and epochs must be numeric") from exc
    if not 0.0 <= dropout < 1.0:
        raise ToolInputError("dropout must satisfy 0 <= dropout < 1")
    if not math.isfinite(lr) or lr <= 0:
        raise ToolInputError("lr must be a finite positive number")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ToolInputError("weight_decay must be finite and non-negative")
    if not 1 <= epochs <= 500:
        raise ToolInputError("epochs must be between 1 and 500")

    scheduler_name = str(scheduler).lower().strip()
    if scheduler_name not in {"none", "step", "cosine", "plateau"}:
        raise ToolInputError("scheduler must be one of: none, step, cosine, plateau")

    actual_input_dim = int(data.data.shape[1])
    if input_dim_override is not None and int(input_dim_override) != actual_input_dim:
        raise ToolInputError(
            "shape mismatch: input_dim_override="
            f"{input_dim_override}, but dataset '{name}' has {actual_input_dim} features. "
            "Retry with input_dim_override=null or the exact feature count."
        )

    requested_device = str(device).lower().strip()
    if requested_device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested_device in {"cpu", "cuda"}:
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise ToolInputError("CUDA was requested but torch.cuda.is_available() is False")
        selected_device = requested_device
    else:
        raise ToolInputError("device must be auto, cpu, or cuda")
    torch_device = torch.device(selected_device)

    x_train, x_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=data.target,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32, device=torch_device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=torch_device)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32, device=torch_device)

    model = RegularizedMLP(
        input_dim=actual_input_dim,
        hidden_dims=hidden_dims,
        output_dim=int(len(np.unique(data.target))),
        dropout=dropout,
        batch_norm=bool(batch_norm),
    ).to(torch_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    lr_scheduler: Any = None
    if scheduler_name == "step":
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, epochs // 3), gamma=0.5
        )
    elif scheduler_name == "cosine":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == "plateau":
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=max(2, epochs // 10)
        )

    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train_tensor)
        loss = criterion(logits, y_train_tensor)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "NaN/Inf loss detected. Retry with a smaller finite learning rate."
            )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if scheduler_name == "plateau":
            lr_scheduler.step(loss.detach())
        elif lr_scheduler is not None:
            lr_scheduler.step()
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 5) == 0:
            history.append(
                {
                    "epoch": epoch,
                    "loss": round(float(loss.detach().cpu().item()), 5),
                    "lr": round(float(optimizer.param_groups[0]["lr"]), 8),
                }
            )

    if torch_device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    model.eval()
    with torch.no_grad():
        test_logits = model(x_test_tensor)
        predicted = torch.argmax(test_logits, dim=1).cpu().numpy()

    return _json(
        {
            "ok": True,
            "tool": "train_pytorch_classifier",
            "framework": "PyTorch",
            "dataset": name,
            "device": selected_device,
            "gpu_name": torch.cuda.get_device_name(0)
            if selected_device == "cuda"
            else None,
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "batch_norm": bool(batch_norm),
            "scheduler": scheduler_name,
            "epochs": epochs,
            "final_loss": round(float(loss.detach().cpu().item()), 5),
            "test_accuracy": round(float(accuracy_score(y_test, predicted)), 4),
            "test_macro_f1": round(float(f1_score(y_test, predicted, average="macro")), 4),
            "final_learning_rate": round(float(optimizer.param_groups[0]["lr"]), 8),
            "elapsed_seconds": round(elapsed, 4),
            "training_history": history,
        }
    )


def train_pytorch_mlp(
    dataset_name: str,
    hidden_dim: int = 32,
    epochs: int = 50,
    lr: float = 0.01,
) -> str:
    """Baseline one-hidden-layer PyTorch MLP required by Task 1."""
    result = json.loads(
        train_pytorch_classifier(
            dataset_name=dataset_name,
            hidden_dims=[int(hidden_dim)],
            dropout=0.0,
            batch_norm=False,
            scheduler="none",
            epochs=epochs,
            lr=lr,
            weight_decay=0.0,
        )
    )
    result["tool"] = "train_pytorch_mlp"
    result["baseline"] = True
    return _json(result)


def compare_models(
    datasets: list[str],
    models: list[str],
    cv: int = 5,
    test_size: float = 0.2,
) -> str:
    """Benchmark multiple classical models and return a Markdown summary table."""
    if not isinstance(datasets, list) or len(datasets) < 1:
        raise ToolInputError("datasets must be a non-empty JSON list")
    if not isinstance(models, list) or len(models) < 1:
        raise ToolInputError("models must be a non-empty JSON list")
    if len(datasets) > 3 or len(models) > 5:
        raise ToolInputError("Benchmark is limited to 3 datasets and 5 models per call")

    rows: list[dict[str, Any]] = []
    for dataset_name in datasets:
        for model_type in models:
            result = json.loads(
                train_sklearn_model(
                    dataset_name=dataset_name,
                    model_type=model_type,
                    test_size=test_size,
                    cv=cv,
                )
            )
            rows.append(
                {
                    "dataset": result["dataset"],
                    "model": result["model"],
                    "test_accuracy": result["test_accuracy"],
                    "cv_mean_accuracy": result["cv_mean_accuracy"],
                    "cv_std_accuracy": result["cv_std_accuracy"],
                    "cv_mean_macro_f1": result["cv_mean_macro_f1"],
                }
            )

    header = (
        "| Dataset | Model | Test accuracy | CV mean | CV std | CV macro-F1 |\n"
        "|---|---|---:|---:|---:|---:|"
    )
    lines = [header]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['test_accuracy']:.4f} "
            f"| {row['cv_mean_accuracy']:.4f} | {row['cv_std_accuracy']:.4f} "
            f"| {row['cv_mean_macro_f1']:.4f} |"
        )
    best = max(rows, key=lambda row: (row["cv_mean_accuracy"], -row["cv_std_accuracy"]))
    return _json(
        {
            "ok": True,
            "tool": "compare_models",
            "cv_folds": int(cv),
            "rows": rows,
            "best_by_cv_mean_then_stability": best,
            "markdown_table": "\n".join(lines),
        }
    )


AVAILABLE_TOOLS: dict[str, Callable[..., str]] = {
    "load_dataset_summary": load_dataset_summary,
    "train_sklearn_model": train_sklearn_model,
    "train_pytorch_mlp": train_pytorch_mlp,
    "tune_hyperparameters": tune_hyperparameters,
    "select_or_reduce_features": select_or_reduce_features,
    "train_pytorch_classifier": train_pytorch_classifier,
    "compare_models": compare_models,
}

