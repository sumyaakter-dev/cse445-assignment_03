"""Run the two-dataset, three-algorithm benchmark required by Task 3."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from ml_tools import compare_models
from react_agent import MODEL_NAME, query_local_llm, run_agent_loop


DATASETS = ["iris", "breast_cancer"]
MODELS = ["decision_tree", "logistic_regression", "random_forest"]

AGENT_BENCHMARK_PROMPT = """
Complete a reproducible benchmark using exactly two datasets (iris and
breast_cancer) and exactly three algorithms (decision_tree,
logistic_regression, random_forest). First inspect each dataset with two
separate load_dataset_summary actions. Then call compare_models once with both
datasets, all three models, and cv=5. Finish with the returned Markdown table.
Interpret CV mean as expected performance and CV standard deviation as
stability. Recommend a model per dataset without inventing any values.
""".strip()


def measure_ollama_latency(repeats: int = 3) -> dict[str, float | int | str]:
    query_local_llm("Reply with exactly WARM.", model_name=MODEL_NAME)
    durations: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        query_local_llm(
            "Reply with exactly one word: READY.", model_name=MODEL_NAME
        )
        durations.append(time.perf_counter() - started)
    return {
        "model": MODEL_NAME,
        "repeats": repeats,
        "mean_seconds": round(statistics.mean(durations), 4),
        "std_seconds": round(statistics.pstdev(durations), 4),
        "min_seconds": round(min(durations), 4),
        "max_seconds": round(max(durations), 4),
    }


def build_markdown(results: dict, latency: dict | None) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    sections = [
        "# CSE445 Model Comparison Benchmark",
        "",
        f"Generated: {timestamp}",
        "",
        "## Configuration",
        "",
        f"- Datasets: {', '.join(DATASETS)}",
        f"- Models: {', '.join(MODELS)}",
        "- Validation: stratified 5-fold cross-validation",
        "- Reproducibility seed: 42",
        "",
        "## Results",
        "",
        results["markdown_table"],
        "",
        "## Selection rule",
        "",
        "Models are compared primarily by mean cross-validation accuracy. When means "
        "are close, lower CV standard deviation is preferred because it indicates "
        "more stable performance across folds.",
        "",
        "## Best overall row",
        "",
        "```json",
        json.dumps(results["best_by_cv_mean_then_stability"], indent=2),
        "```",
    ]
    if latency is not None:
        sections.extend(
            [
                "",
                "## Local LLM latency",
                "",
                f"- Ollama model: {latency['model']}",
                f"- Mean latency: {latency['mean_seconds']:.4f} seconds",
                f"- Standard deviation: {latency['std_seconds']:.4f} seconds",
                f"- Range: {latency['min_seconds']:.4f} to {latency['max_seconds']:.4f} seconds",
            ]
        )
    return "\n".join(sections) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CSE445 benchmark")
    parser.add_argument("--output", default="results/benchmark_results.md")
    parser.add_argument("--log", default="logs/trace_04_benchmark_agent.txt")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--direct-only", action="store_true")
    args = parser.parse_args()

    results = json.loads(compare_models(DATASETS, MODELS, cv=5, test_size=0.2))
    latency = None if args.skip_latency else measure_ollama_latency()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_markdown(results, latency), encoding="utf-8")
    print(results["markdown_table"])
    print(f"\nBenchmark report saved to: {output_path}")

    if not args.direct_only:
        run_agent_loop(
            AGENT_BENCHMARK_PROMPT,
            max_iterations=12,
            log_path=args.log,
            model_name=MODEL_NAME,
        )


if __name__ == "__main__":
    main()

