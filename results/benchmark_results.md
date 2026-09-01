# CSE445 Model Comparison Benchmark

Generated: 2026-08-30 19:33:57 UTC

## Configuration

- Datasets: iris, breast_cancer
- Models: decision_tree, logistic_regression, random_forest
- Validation: stratified 5-fold cross-validation
- Reproducibility seed: 42

## Results

| Dataset | Model | Test accuracy | CV mean | CV std | CV macro-F1 |
|---|---|---:|---:|---:|---:|
| iris | decision_tree | 0.9333 | 0.9533 | 0.0340 | 0.9531 |
| iris | logistic_regression | 0.9333 | 0.9533 | 0.0452 | 0.9532 |
| iris | random_forest | 0.9000 | 0.9467 | 0.0267 | 0.9464 |
| breast_cancer | decision_tree | 0.9386 | 0.9227 | 0.0295 | 0.9161 |
| breast_cancer | logistic_regression | 0.9825 | 0.9737 | 0.0166 | 0.9714 |
| breast_cancer | random_forest | 0.9561 | 0.9561 | 0.0123 | 0.9529 |

## Selection rule

Models are compared primarily by mean cross-validation accuracy. When means are close, lower CV standard deviation is preferred because it indicates more stable performance across folds.

## Best overall row

```json
{
  "dataset": "breast_cancer",
  "model": "logistic_regression",
  "test_accuracy": 0.9825,
  "cv_mean_accuracy": 0.9737,
  "cv_std_accuracy": 0.0166,
  "cv_mean_macro_f1": 0.9714
}
```

## Local LLM latency

- Ollama model: llama3.2:3b
- Mean latency: 0.0530 seconds
- Standard deviation: 0.0040 seconds
- Range: 0.0490 to 0.0585 seconds
