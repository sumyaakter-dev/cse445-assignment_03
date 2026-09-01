"""Local Ollama ReAct controller with structured parsing and self-correction."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests

from ml_tools import AVAILABLE_TOOLS, ToolInputError


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

TOOL_CATALOG = """
1. load_dataset_summary
   Input: {"dataset_name": "iris|wine|breast_cancer"}
2. train_sklearn_model
   Input: {"dataset_name": "...", "model_type": "decision_tree|logistic_regression|random_forest|svc", "test_size": 0.2, "cv": 5}
3. train_pytorch_mlp
   Input: {"dataset_name": "...", "hidden_dim": 32, "epochs": 50, "lr": 0.01}
4. tune_hyperparameters
   Input: {"dataset_name": "...", "model_type": "svc|decision_tree", "search_type": "grid|randomized", "cv": 5}
5. select_or_reduce_features
   Input: {"dataset_name": "...", "method": "pca|sequential", "n_components": 2, "n_features_to_select": 2, "direction": "forward|backward", "cv": 5}
6. train_pytorch_classifier
   Input: {"dataset_name": "...", "hidden_dims": [64, 32], "dropout": 0.3, "batch_norm": true, "scheduler": "none|step|cosine|plateau", "epochs": 60, "lr": 0.01, "weight_decay": 0.0001, "device": "auto", "input_dim_override": null}
7. compare_models
   Input: {"datasets": ["iris", "breast_cancer"], "models": ["decision_tree", "logistic_regression", "random_forest"], "cv": 5}
""".strip()

SYSTEM_PROMPT = f"""You are an autonomous local Machine Learning Agent.
You must base every numerical claim on observations returned by tools.
Use one tool per turn and continue until every requested experiment is complete.
Never invent a metric. Keep Thought concise and focused on the next operation.
CRITICAL: Emit exactly one Action per response and stop immediately after the
closing brace of Action Input. Never simulate an Observation, never write
"Waiting for output", and never emit a second Thought or Action in the same turn.

Available tools:
{TOOL_CATALOG}

For a tool call, output exactly:
Thought: <brief reason for the next operation>
Action: <exact tool name>
Action Input: <one valid JSON object>

After an error observation, explain the cause briefly and retry with corrected
parameters. Do not abandon a recoverable task. When all required observations
have been collected, output:
Thought: I have gathered all necessary experimental data.
Final Answer: <evidence-based answer; include a Markdown table when requested>
"""


class Transcript:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def emit(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("\n".join(self.lines).rstrip() + "\n", encoding="utf-8")


def query_local_llm(
    prompt: str,
    model_name: str = MODEL_NAME,
    timeout: int = 240,
    attempts: int = 2,
) -> str:
    """Query Ollama with bounded retries for transient connection failures."""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "seed": 42,
            "num_ctx": 8192,
            "stop": [
                "\nObservation:",
                "\nWaiting for",
                "\n( Output:",
                "\n\nThought:",
            ],
        },
    }
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            response.raise_for_status()
            output = response.json().get("response", "").strip()
            if not output:
                raise RuntimeError("Ollama returned an empty response")
            return output
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(1.5)
    raise RuntimeError(f"Unable to query Ollama after {attempts} attempts: {last_error}")


def _extract_json_object(text: str) -> dict[str, Any]:
    marker = re.search(r"Action Input:\s*", text, flags=re.IGNORECASE)
    if not marker:
        raise ValueError("Missing 'Action Input:' line")
    remainder = text[marker.end() :].lstrip()
    if remainder.startswith("```json"):
        remainder = remainder[7:].lstrip()
    elif remainder.startswith("```"):
        remainder = remainder[3:].lstrip()
    opening = remainder.find("{")
    if opening < 0:
        raise ValueError("Action Input does not contain a JSON object")
    decoder = json.JSONDecoder()
    value, _ = decoder.raw_decode(remainder[opening:])
    if not isinstance(value, dict):
        raise ValueError("Action Input must decode to a JSON object")
    return value


def parse_action(text: str) -> tuple[str, dict[str, Any]]:
    match = re.search(r"Action:\s*([A-Za-z_][A-Za-z0-9_]*)", text)
    if not match:
        raise ValueError("Missing or invalid 'Action:' line")
    return match.group(1), _extract_json_object(text)


def _recovery_hint(error: Exception, tool_name: str) -> str:
    message = str(error).lower()
    if "shape mismatch" in message or "shapes cannot be multiplied" in message:
        return "Remove input_dim_override or set it to the dataset's reported feature count."
    if "dropout" in message:
        return "Use a dropout value from 0.0 up to, but not including, 1.0."
    if "nan" in message or "inf" in message or "finite" in message:
        return "Retry with a smaller finite learning rate such as 0.001."
    if "dataset" in message:
        return "Use one of: iris, wine, breast_cancer."
    if isinstance(error, TypeError):
        return f"Check the documented JSON parameter names for {tool_name}."
    return "Read the error message, correct only the invalid parameter, and retry."


def execute_tool(tool_name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
    if tool_name not in AVAILABLE_TOOLS:
        payload = {
            "ok": False,
            "error_type": "UnknownTool",
            "message": f"Tool '{tool_name}' is not registered.",
            "correction_hint": f"Choose one of: {sorted(AVAILABLE_TOOLS)}",
        }
        return json.dumps(payload, indent=2), False
    try:
        result = AVAILABLE_TOOLS[tool_name](**arguments)
        parsed = json.loads(result)
        return result, bool(parsed.get("ok", True))
    except (ToolInputError, TypeError, ValueError, RuntimeError, FloatingPointError) as exc:
        payload = {
            "ok": False,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "failed_tool": tool_name,
            "failed_input": arguments,
            "correction_hint": _recovery_hint(exc, tool_name),
        }
        return json.dumps(payload, indent=2, allow_nan=False), False
    except Exception as exc:  # Defensive boundary around third-party libraries.
        payload = {
            "ok": False,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "failed_tool": tool_name,
            "correction_hint": _recovery_hint(exc, tool_name),
        }
        return json.dumps(payload, indent=2, allow_nan=False), False


def synthesize_final_answer(
    user_query: str,
    evidence: list[dict[str, Any]],
    model_name: str,
) -> str:
    """Ask the local model to report results after tool use is complete.

    The synthesis prompt intentionally contains no tool catalog. This keeps a
    small local model from repeating an already completed action.
    """
    prompt = f"""You are the final reporting stage of a machine-learning agent.
All requested computations have already succeeded. You cannot call tools now.
Use only the JSON evidence below; never invent, estimate, or omit a requested
metric. Interpret cross-validation mean as expected performance and its standard
deviation as stability.

Original user query:
{user_query}

Verified tool evidence:
{json.dumps(evidence, indent=2, allow_nan=False)}

Respond exactly in this format and do not output Action or Action Input:
Thought: I have gathered all necessary experimental data.
Final Answer: <concise evidence-based answer>
"""
    output = query_local_llm(prompt, model_name=model_name)
    if "Final Answer:" in output and "Action:" not in output:
        return output

    correction_prompt = f"""{prompt}

Your previous formatting attempt was invalid:
{output}

Do not call a tool. Return only one Thought line followed by Final Answer.
"""
    return query_local_llm(correction_prompt, model_name=model_name)


def run_agent_loop(
    user_query: str,
    max_iterations: int = 12,
    log_path: str | Path | None = None,
    model_name: str = MODEL_NAME,
) -> str:
    """Run the Thought/Action/Observation cycle and optionally export a log."""
    transcript = Transcript()
    transcript.emit("=" * 72)
    transcript.emit(f"MODEL: {model_name}")
    transcript.emit(f"USER QUERY: {user_query}")
    transcript.emit("=" * 72)

    prompt = f"{SYSTEM_PROMPT}\n\nUser Query: {user_query}\n"
    final_answer = ""
    completed_tools: set[str] = set()
    successful_calls: set[str] = set()
    explicitly_required_tools = {
        tool_name for tool_name in AVAILABLE_TOOLS if tool_name in user_query
    }
    successful_evidence: list[dict[str, Any]] = []
    for step in range(1, max_iterations + 1):
        transcript.emit(f"\n--- ReAct Step {step} ---")
        if (
            explicitly_required_tools
            and explicitly_required_tools.issubset(completed_tools)
        ):
            try:
                llm_output = synthesize_final_answer(
                    user_query=user_query,
                    evidence=successful_evidence,
                    model_name=model_name,
                )
            except RuntimeError as exc:
                transcript.emit(f"Controller Error: {exc}")
                final_answer = f"Controller Error: {exc}"
                break
            transcript.emit(llm_output)
            if "Final Answer:" in llm_output and "Action:" not in llm_output:
                final_answer = llm_output.split("Final Answer:", 1)[1].strip()
                transcript.emit("\n>>> Task completed successfully.")
                break
            observation = "Observation: " + json.dumps(
                {
                    "ok": False,
                    "error_type": "FinalSynthesisFormatError",
                    "message": "The reporting stage did not return the required Final Answer format.",
                    "correction_hint": "Return a Thought line and Final Answer only; no Action.",
                },
                indent=2,
            )
            transcript.emit(observation)
            continue
        try:
            llm_output = query_local_llm(prompt, model_name=model_name)
        except RuntimeError as exc:
            transcript.emit(f"Controller Error: {exc}")
            final_answer = f"Controller Error: {exc}"
            break
        transcript.emit(llm_output)
        prompt += f"\n{llm_output}\n"

        has_action = re.search(r"Action:\s*[A-Za-z_]", llm_output) is not None
        if "Final Answer:" in llm_output and not has_action:
            missing_tools = sorted(explicitly_required_tools - completed_tools)
            if missing_tools:
                observation = "Observation: " + json.dumps(
                    {
                        "ok": False,
                        "error_type": "IncompleteTask",
                        "message": "A Final Answer was attempted before all explicitly requested tools succeeded.",
                        "missing_tools": missing_tools,
                        "correction_hint": "Call each missing tool before producing the Final Answer.",
                    },
                    indent=2,
                )
                transcript.emit(observation)
                prompt += f"{observation}\n"
                continue
            final_answer = llm_output.split("Final Answer:", 1)[1].strip()
            transcript.emit("\n>>> Task completed successfully.")
            break

        try:
            tool_name, arguments = parse_action(llm_output)
            call_key = json.dumps(
                {"tool": tool_name, "arguments": arguments}, sort_keys=True
            )
            if call_key in successful_calls:
                succeeded = False
                observation_body = json.dumps(
                    {
                        "ok": False,
                        "error_type": "DuplicateSuccessfulCall",
                        "message": "This exact tool call already succeeded and was not executed again.",
                        "completed_tool": tool_name,
                        "correction_hint": "Proceed to the next unfinished part of the user query.",
                    },
                    indent=2,
                )
            else:
                observation_body, succeeded = execute_tool(tool_name, arguments)
                if succeeded:
                    successful_calls.add(call_key)
                    completed_tools.add(tool_name)
                    successful_evidence.append(json.loads(observation_body))
        except (ValueError, json.JSONDecodeError) as exc:
            succeeded = False
            observation_body = json.dumps(
                {
                    "ok": False,
                    "error_type": "ActionParseError",
                    "message": str(exc),
                    "correction_hint": (
                        "Output one Action and one Action Input containing strict JSON."
                    ),
                },
                indent=2,
            )

        observation = f"Observation: {observation_body}"
        transcript.emit(observation)
        prompt += f"{observation}\n"
        if not succeeded:
            recovery = (
                "Recovery Directive: The previous operation failed. Briefly identify "
                "the invalid value, then retry the same goal with corrected parameters."
            )
            transcript.emit(recovery)
            prompt += f"{recovery}\n"
    else:
        final_answer = "Maximum iterations reached before a Final Answer."
        transcript.emit(f"\n>>> {final_answer}")

    if log_path is not None:
        transcript.save(log_path)
        transcript.emit(f"Log saved to: {log_path}")
    return final_answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local CSE445 ReAct ML agent")
    parser.add_argument(
        "--query",
        default=(
            "Analyze the breast_cancer dataset, train a Random Forest and a "
            "PyTorch MLP, compare their accuracies, and recommend the better model."
        ),
    )
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--log", default="logs/manual_agent_trace.txt")
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()
    run_agent_loop(
        user_query=args.query,
        max_iterations=args.max_iterations,
        log_path=args.log,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
