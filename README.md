# CSE445 Autonomous Local LLM Machine Learning Agent

This repository implements a privacy-preserving ReAct agent in Windows WSL2.
The reasoning model (`llama3.2:3b`) runs locally through Ollama and invokes
deterministic Scikit-learn and GPU-enabled PyTorch tools.

## Assignment coverage

| Requirement | Implementation |
|---|---|
| WSL2 and local quantized LLM | Ubuntu 22.04, Ollama REST API, Llama 3.2 3B |
| Baseline ReAct engine | `react_agent.py` action parser and iterative controller |
| Baseline ML tools | Dataset summary, three Scikit-learn classifiers, PyTorch MLP |
| Hyperparameter tuning | GridSearchCV/RandomizedSearchCV for SVC and Decision Tree |
| Feature methods | Leakage-safe PCA and sequential feature selection pipelines |
| Regularized deep model | Configurable hidden layers, Dropout, BatchNorm, AdamW, schedulers |
| Self-correction | Structured errors, correction hints, retry loop, NaN/shape checks |
| Benchmark | Three algorithms, two datasets, stratified CV, Markdown table |
| Execution evidence | `generate_traces.py` exports three multi-step logs |

## Project structure

```text
.
|-- ml_tools.py
|-- react_agent.py
|-- benchmark_runner.py
|-- generate_traces.py
|-- setup_check.py
|-- requirements.txt
|-- REPORT_GUIDE.md
|-- tests/
|-- logs/
`-- results/
```

## WSL installation

These commands assume Ubuntu 22.04 and an active virtual environment.

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv curl build-essential git zstd unzip
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

Install Ollama and pull the local model:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b
ollama run llama3.2:3b "Reply with exactly: LOCAL LLM READY"
```

For an NVIDIA GPU and a compatible WSL driver, install the CUDA wheel first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

## Verification

```bash
python setup_check.py
python -m pytest -q
```

Expected checks include `CUDA ready: True`, the NVIDIA GPU name, a reachable
Ollama API, and `llama3.2:3b` in the model list.

## Task 1: baseline ReAct run

```bash
python react_agent.py --query "Inspect the iris dataset, train a decision tree with 5-fold CV, and interpret its accuracy and stability." --log logs/baseline_trace.txt
```

The controller accepts only a registered action and strict JSON input, executes
the selected function, appends the observation, and continues until the model
emits `Final Answer:`.

## Task 2: advanced tools

The local model can autonomously call:

- `tune_hyperparameters` for SVC and Decision Tree search.
- `select_or_reduce_features` for PCA or sequential feature selection.
- `train_pytorch_classifier` for Dropout, BatchNorm, AdamW, and four scheduler
  choices (`none`, `step`, `cosine`, `plateau`).

## Task 3: self-correction and benchmark

Generate the three required multi-step logs:

```bash
python generate_traces.py
```

The third trace deliberately supplies an incorrect neural-network input
dimension. The tool returns a structured shape-mismatch observation; the agent
must reason about it and retry with a corrected value.

Run the complete two-dataset, three-model benchmark and local latency test:

```bash
python benchmark_runner.py
```

Outputs:

- `results/benchmark_results.md`
- `logs/trace_04_benchmark_agent.txt`

## Reproducibility and statistical interpretation

All splits and algorithms use seed 42. Classical model estimates use stratified
5-fold cross-validation. Mean CV accuracy estimates expected performance; CV
standard deviation measures fold-to-fold instability. Macro-F1 weights every
class equally and therefore complements accuracy when class frequencies differ.

## Submission checklist

- [ ] Confirm all tests pass.
- [ ] Review each generated log and retain at least three successful multi-step traces.
- [ ] Include `results/benchmark_results.md` with genuine machine results.
- [ ] Add the 3-5 page PDF technical report based on `REPORT_GUIDE.md`.
- [ ] Push code, selected logs, benchmark output, and report to GitHub.
- [ ] Exclude `venv/`, caches, and editor files.
