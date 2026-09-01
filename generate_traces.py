"""Generate three genuine multi-step agent logs for the submission."""

from __future__ import annotations

from react_agent import run_agent_loop


TRACE_TASKS = [
    (
        "logs/trace_01_tuning.txt",
        "Inspect the wine dataset, then tune an SVC with grid search and 5-fold CV. "
        "Explain the best parameters, mean validation accuracy, variation, and test accuracy.",
    ),
    (
        "logs/trace_02_feature_and_deep_model.txt",
        "Inspect the iris dataset. Evaluate PCA with 2 components, then train a regularized "
        "PyTorch classifier with hidden_dims [64, 32], dropout 0.3, BatchNorm enabled, "
        "cosine scheduling, 60 epochs, and lr 0.01. Compare the evidence and mention the GPU.",
    ),
    (
        "logs/trace_03_self_correction.txt",
        "Demonstrate self-correction on the breast_cancer dataset. Your FIRST action must call "
        "train_pytorch_classifier with hidden_dims [32], epochs 30, lr 0.01 and the deliberately "
        "wrong input_dim_override 999. After the expected shape-mismatch observation, reason "
        "about the error and retry with input_dim_override null. Report only the successful metrics.",
    ),
]


def main() -> None:
    for log_path, query in TRACE_TASKS:
        print(f"\nGenerating {log_path}\n")
        run_agent_loop(query, max_iterations=12, log_path=log_path)


if __name__ == "__main__":
    main()

