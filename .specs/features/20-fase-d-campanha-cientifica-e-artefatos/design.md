# F20: Fase D — Campanha Científica Final e Artefatos Design

## System Architecture Overview

A Fase D unifica a execução da matriz experimental congelada, a análise estatística inferencial pareada e a geração rigorosa dos artefatos de publicação científica.

```
                    ┌────────────────────────────────────────────────────────┐
                    │               Scientific Campaign Runner               │
                    │         (Frozen Protocol Matrix: Seeds, Splits)        │
                    └───────────────────────────┬────────────────────────────┘
                                                │
         ┌──────────────────────────────────────┼──────────────────────────────────────┐
         │                                      │                                      │
         ▼                                      ▼                                      ▼
┌──────────────────┐                  ┌──────────────────┐                  ┌──────────────────┐
│  Scenario 1 Grid │                  │  Scenario 2 Grid │                  │ Computational    │
├──────────────────┤                  ├──────────────────┤                  │ Cost Telemetry   │
│• 1:50, 100, 500  │                  │• Uniform         │                  ├──────────────────┤
│• 4 Methods       │                  │• Uniform Grouped │                  │• Step updates    │
│• 5 Paired Seeds  │                  │• 4x Factorial    │                  │• Gradient samples│
│• AUPRC & Tau_val │                  │• 5 Paired Seeds  │                  │• Mining forwards │
│• Last & Best_Val │                  │• Macro-F1 & W10  │                  │• Wall-clock time │
└──────────────────┘                  └──────────────────┘                  └──────────────────┘
         │                                      │                                      │
         └──────────────────────────────────────┼──────────────────────────────────────┘
                                                │
                                                ▼
                               ┌──────────────────────────────────┐
                               │     Inferential Statistics       │
                               │  (Paired t, Cohen's d, Method ×  │
                               │   Ratio Interaction Analysis)    │
                               └────────────────┬─────────────────┘
                                                │
                                                ▼
                               ┌──────────────────────────────────┐
                               │   Publication Paper Artifacts    │
                               │  (Val Curves, Exposure Heatmaps, │
                               │   2x2 Ablation, LaTeX Tables)    │
                               └────────────────┬─────────────────┘
                                                │
                                                ▼
                               ┌──────────────────────────────────┐
                               │          Gate D Verifier         │
                               │  (Integrity audit, provenance,   │
                               │   campaign_audit_manifest.json)  │
                               └──────────────────────────────────┘
```

---

## Detailed Component Design

### 1. Scientific Campaign Runner (`src/training/campaign_runner.py`)

Orquestra a matriz experimental completa sob o protocolo rigorosamente congelado:

- **Configurações Congeladas**:
  - Seeds pareadas: `(42, 43, 44, 45, 46)`
  - Classe-alvo do Cenário 1: `target_class = 0` com os mesmos 80 positivos únicos em `train_fit` para todas as razões
  - Razões do Cenário 1: `("1:50", "1:100", "1:500")`
  - Métodos do Cenário 1: `("uniform", "ohem", "mcts_strict", "mcts_enhanced")`
  - Métodos do Cenário 2: `("uniform", "uniform_grouped", "mcts_strict", "mcts_qnorm_only", "mcts_dualstream_only", "mcts_enhanced")`
  - Checkpoints: `CheckpointManager` persiste `last` e `best_validation` para todos os treinamentos
- **Saídas Atômicas**:
  - Cada execução é serializada em `results/{mode}/campaign/{run_id}.json` acompanhada de seu `.meta.json` de proveniência.
  - Um manifesto consolidado `campaign_manifest.json` registra o inventário completo de execuções.

### 2. Inferência Estatística Pareada (`src/utils/statistics.py`)

Implementa testes de hipótese e tamanhos de efeito respeitando o desenho amostral:

- **Diferenças Pareadas por Seed**:
  $$d_s = \text{Metric}_A(s) - \text{Metric}_B(s), \quad \bar{d} = \frac{1}{S} \sum_{s=1}^S d_s$$
  $$\text{SE} = \frac{s_d}{\sqrt{S}}, \quad d_{\text{Cohen}} = \frac{\bar{d}}{s_d}$$
  $$95\% \text{ CI} = \left[ \bar{d} - t_{S-1, 0.975} \text{SE}, \quad \bar{d} + t_{S-1, 0.975} \text{SE} \right]$$
- **Interação Método $\times$ Razão**:
  Para avaliar se a vantagem do MCTS sobre o Uniform varia com o grau de desbalanceamento ($r_1$ vs. $r_2$):
  $$\Delta_s = \left( M_{\text{MCTS}}(s, r_2) - M_{\text{Uni}}(s, r_2) \right) - \left( M_{\text{MCTS}}(s, r_1) - M_{\text{Uni}}(s, r_1) \right)$$
  Permite testar se a interação é estatisticamente diferente de zero.

### 3. Gerador de Artefatos Científicos e Tabelas (`src/utils/paper_artifacts.py`)

- **Curvas de Validação com Banda de Incerteza**:
  - Média e desvio-padrão por passo sobre as 5 seeds para cada método.
- **Heatmap de Exposição por Classe**:
  - Matriz visual de seleção de classes ao longo do tempo.
- **Matriz de Ablação $2 \times 2$ (QNorm $\times$ DualStream)**:
  - Gráfico de barras agrupadas comparando os 4 quadrantes da ablação em acurácia, macro-F1 e Worst-10 recall.
- **Gráficos de Trade-off Desempenho vs. Custo**:
  - AUPRC / Acurácia em função de wall-clock seconds e forwards adicionais de mineração.
- **Tabelas LaTeX Consolidadas**:
  - Diferenciação explícita entre `best_validation` e `last`.
  - Inclusão de AUPRC, F1 minoritário calibrado ($\tau_{\text{val}}$) e precision/recall.

### 4. Verificador do Gate D (`src/utils/gate_d_verifier.py`)

- **Função**: `verify_gate_d(campaign_dir: Path | str, mode: str = "smoke") -> dict[str, Any]`
- **Critérios de Validação**:
  1. **Completude da Matriz**: Confirma presença de todas as combinações de método $\times$ razão $\times$ seed.
  2. **Ausência de Dados Sintéticos**: Em modo REAL, rejeita qualquer artefato sintético ou cache não rastreável.
  3. **Validade Numérica**: Inexistência de `NaN`, `Inf` ou chaves ausentes.
  4. **Emissão do Manifesto Final**: Gera `campaign_audit_manifest.json` com resumo e digest SHA-256 de todas as saídas.
