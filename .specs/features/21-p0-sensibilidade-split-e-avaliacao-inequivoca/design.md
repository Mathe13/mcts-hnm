# F21: P0 — Sensibilidade com Subsampling Equivalente e Política de Avaliação Inequívoca Design

## System Architecture Overview

Esta feature harmoniza o runner de sensibilidade de hiperparâmetros (`run_sensitivity_ablation`) com o runner de benchmark do Axis 1 (`run_benchmark`), eliminando a discrepância metodológica de subsampling e formalizando uma política inequívoca de avaliação com 4 quadrantes de métricas.

```
                          ┌─────────────────────────────────────────┐
                          │    CIFAR-100 Embeddings & Labels        │
                          │   (train_features, train_labels, ...)   │
                          └────────────────────┬────────────────────┘
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       │                                               │
                       ▼                                               ▼
         ┌───────────────────────────┐                   ┌───────────────────────────┐
         │      run_benchmark()      │                   │ run_sensitivity_ablation()│
         │         (Axis 1)          │                   │  (Ablação c e alpha)      │
         └─────────────┬─────────────┘                   └─────────────┬─────────────┘
                       │                                               │
                       │           SHARED SEED & SPLIT LOGIC           │
                       ├───────────────────────────────────────────────┤
                       │  split = build_scenario1_split(               │
                       │      fine_labels=train_lbl_np,                │
                       │      target_class=target_class,               │
                       │      pos_fraction=IMBALANCE_RATIOS["1:50"],   │
                       │      seed=seed,                               │
                       │  )                                            │
                       │  positive_indices: ~9 instâncias (1:50)       │
                       │  negative_indices: 44.550 instâncias          │
                       │  effective pool ratio: ~1:4.950               │
                       └───────────────────────┬───────────────────────┘
                                               │
                                               ▼
                                ┌─────────────────────────────┐
                                │      train_scenario1()      │
                                │   (dual checkpoint state)   │
                                └──────────────┬──────────────┘
                                               │
                                               ▼
                                ┌─────────────────────────────┐
                                │     4-Quadrant Evaluation   │
                                ├─────────────────────────────┤
                                │ 1. Last, Thresh 0.5         │
                                │ 2. Last, Thresh Calibrated  │
                                │ 3. Best Val, Thresh 0.5     │
                                │ 4. Best Val, Thresh Calibr. │
                                └──────────────┬──────────────┘
                                               │
                                               ▼
                                ┌─────────────────────────────┐
                                │   Export JSON / CSV Artifact│
                                │   with Provenance Sidecar   │
                                └─────────────────────────────┘
```

---

## Detailed Component Design

### 1. Parametrização e Subsampling em `run_sensitivity_ablation()` (`src/training/benchmark_runner.py`)

- **Assinatura e Parâmetros**:
  ```python
  def run_sensitivity_ablation(
      train_features: torch.Tensor,
      train_labels: np.ndarray | torch.Tensor,
      val_features: torch.Tensor,
      val_labels: np.ndarray | torch.Tensor,
      test_features: torch.Tensor,
      test_labels: np.ndarray | torch.Tensor,
      c_values: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
      alpha_values: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7),
      output_dir: str | Path = "results",
      benchmark_name: str = "tier1_hyperparameter_sensitivity",
      imbalance_ratio: str = "1:50",
      num_steps: int = 1000,
      eval_every: int = 100,
      seeds: tuple[int, ...] = (42,),
      target_class_scenario1: int = 0,
      config: ExperimentConfig | None = None,
      device: torch.device | None = None,
  ) -> dict[str, Any]:
  ```
- **Lógica de Particionamento**:
  - Para cada `seed` em `seeds`, invoca:
    ```python
    set_global_seed(seed)
    pos_frac = IMBALANCE_RATIOS[imbalance_ratio]
    split = build_scenario1_split(
        fine_labels=train_lbl_np,
        target_class=target_class_scenario1,
        pos_fraction=pos_frac,
        seed=seed,
    )
    pos_idx = split.positive_indices
    neg_idx = split.negative_indices
    binary_train_lbl = split.binary_labels
    class_to_indices = split.negative_class_to_indices
    ```
  - Isso garante identidade bit a bit com o conjunto de treino do `run_benchmark(scenario="scenario1")` sob a mesma seed.

### 2. Política Inequívoca de Avaliação nos 4 Quadrantes

No treinamento de cada combinação $(c, \alpha, \text{seed})$:
- A cada avaliação (e no final do treino), avalia-se tanto no checkpoint final (`last`) quanto no melhor modelo de validação (`best_validation`).
- Para cada checkpoint, avalia-se:
  1. `thresh_0.5`: métricas padrão com corte fixo em logit $0.0$ (probabilidade $0.5$).
  2. `calibrated_val`: métricas usando $\tau_{\text{val}} = \arg\max_\tau \text{F1}_{\text{val}}(\tau)$ determinado estritamente no conjunto de validação.
- O resultado de cada célula do grid registra explicitamente:
  ```json
  {
    "c": 2.0,
    "alpha": 0.3,
    "seed": 42,
    "imbalance_ratio": "1:50",
    "num_positives_train": 9,
    "num_negatives_train": 44550,
    "test_f1_last_thresh05": 0.65,
    "test_precision_last_thresh05": 0.60,
    "test_recall_last_thresh05": 0.71,
    "test_f1_last_calibrated": 0.68,
    "test_precision_last_calibrated": 0.66,
    "test_recall_last_calibrated": 0.70,
    "best_tau_val_last": 0.42,
    "test_f1_best_val_thresh05": 0.67,
    "test_precision_best_val_thresh05": 0.62,
    "test_recall_best_val_thresh05": 0.73,
    "test_f1_best_val_calibrated": 0.70,
    "test_precision_best_val_calibrated": 0.68,
    "test_recall_best_val_calibrated": 0.72,
    "best_tau_val": 0.45,
    "test_auprc": 0.69
  }
  ```
- Para compatibilidade retroativa, as chaves canônicas `test_f1`, `test_precision`, `test_recall` espelham `test_f1_best_val_calibrated` (ou o modo padrão documentado com clareza).

### 3. Teste Pareado de Equivalência (`tests/test_sensitivity_runner_equivalence.py`)

- Um teste dedicado `test_sensitivity_runner_equivalence.py` instancia synthetic ou smoke splits e executa:
  1. `build_scenario1_split` direto vs índices extraídos no mock/hook do `run_sensitivity_ablation`.
  2. Execução comparativa de `run_benchmark` e `run_sensitivity_ablation` sob a mesma semente, conferindo que:
     - `pos_idx_bench == pos_idx_sens`
     - `neg_idx_bench == neg_idx_sens`
     - `len(pos_idx_sens) == len(pos_idx_bench)`
     - Parâmetros `imbalance_ratio`, `pos_batch_size`, `neg_batch_size` e `tree_type` são rigorosamente conservados.

---

## Traceability to Requirements

| Requirement | Implementation Component | Verification Target |
| ----------- | ------------------------ | ------------------- |
| `SENS-01` | `src/training/benchmark_runner.py` (`run_sensitivity_ablation`) | Adota `build_scenario1_split` por semente |
| `SENS-02` | `src/training/benchmark_runner.py` (`run_sensitivity_ablation`) | Salva metadados de razão e tamanhos do pool |
| `EVAL-01` | `src/training/trainer.py` & `src/training/benchmark_runner.py` | Computa os 4 quadrantes (`last`/`best_val` $\times$ `0.5`/`calibrated`) |
| `EVAL-02` | `src/training/benchmark_runner.py` | Estrutura JSON/CSV com campos inequívocos |
| `EQUIV-01` | `tests/test_sensitivity_runner_equivalence.py` | Teste automatizado atestando identidade de índices |
