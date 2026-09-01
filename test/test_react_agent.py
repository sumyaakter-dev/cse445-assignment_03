from pathlib import Path

import react_agent


def test_explicit_tools_trigger_final_synthesis(monkeypatch, tmp_path: Path) -> None:
    responses = iter(
        [
            "Thought: inspect data\nAction: load_dataset_summary\n"
            'Action Input: {"dataset_name": "iris"}',
            "Thought: train model\nAction: train_sklearn_model\n"
            'Action Input: {"dataset_name": "iris", "model_type": "decision_tree", "cv": 3}',
            "Thought: I have gathered all necessary experimental data.\n"
            "Final Answer: The verified experiment is complete.",
        ]
    )

    monkeypatch.setattr(
        react_agent,
        "query_local_llm",
        lambda *args, **kwargs: next(responses),
    )
    answer = react_agent.run_agent_loop(
        "Use load_dataset_summary and train_sklearn_model for iris.",
        max_iterations=5,
        log_path=tmp_path / "trace.txt",
    )
    assert answer == "The verified experiment is complete."
    assert "Task completed successfully" in (tmp_path / "trace.txt").read_text()

