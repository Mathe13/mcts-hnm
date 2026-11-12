# MCTS-HNM — Fase 1: recuperação conservadora de evidências

Escopo: corrigir o notebook 05 **sem repetir os benchmarks completos** e preparar exportação fiel para rodadas futuras. Executar sobre o código que já recebeu o primeiro patch de integridade científica.

## 1. O que é recuperável imediatamente

- **Entropia de visitação, Scenario 2**: os arquivos `results/real/runs/*.json` já armazenam os históricos por passo. O novo arquivo `fig_measured_visitation_entropy` usa somente esses valores, média e desvio padrão entre seeds. Ele **não** é o gráfico completo de entropia *e* cobertura: não inventamos históricos de cobertura.
- **Confusão de validação**: o notebook 02 calcula `confusion_mat_100`, mas historicamente só salvava as árvores. A célula agora persiste `embeddings/real/trees/validation_confusion_matrix_100.npz` com sidecar de proveniência. **Se a variável ainda existir no kernel do notebook 02**, basta executar a célula modificada. Caso contrário, use a recuperação explícita abaixo (reajusta apenas a regressão logística de referência, NÃO repete Tier-1/Tier-2).
- **Resumo de exposição/recall do Scenario 2**: existe no JSON agregado, mas é **apenas o último snapshot por estratégia**, originalmente sem identificação de seed. É possível inspecioná-lo; não convertê-lo em cinco observações pareadas para uma figura publicável.
- **Quatro quadrantes**: não podem ser refeitos a partir de `final_metrics` somente. O treinamento tinha os estados do melhor checkpoint em memória, mas os checkpoints **não foram serializados** nas execuções históricas. O patch passa a guardar as quatro métricas já medidas nos runs **futuros**, sem exigir nova inferência durante a exportação.

## 2. Aplicação remota

```bash
cd ~/workspace/mcts-hnm
python3 apply_mcts_hnm_phase1_patch.py --check
python3 apply_mcts_hnm_phase1_patch.py --emit-patch /tmp/mcts_phase1.patch
git apply --check /tmp/mcts_phase1.patch
git apply /tmp/mcts_phase1.patch
python3 -m pytest tests/test_phase1_evidence.py -q
git diff --check
```

## 3. Inspecionar o material existente

```bash
python3 -m src.utils.phase1_evidence inspect --mode real --repo-root .
```

Se a matriz de confusão ainda não tiver sido salva, existem **duas opções**:

1. No kernel do notebook 02 que contenha `confusion_mat_100`, executar a célula que chama `save_validation_confusion_matrix(...)` — sem ajustar novos modelos.
2. Reconstruir explicitamente a matriz usando somente os embeddings de treinamento e validação cacheados, sem relançar benchmarks:

```bash
python3 -m src.utils.phase1_evidence rebuild-confusion --mode real --repo-root .
```

A segunda opção executa um ajuste determinístico da regressão logística usada para a matriz de confusão e grava `reconstructed_from_cached_embeddings=true`. Ela exige o cache real e seu sidecar de proveniência. **Não é a recuperação exata de uma matriz anteriormente não salva**, embora reproduza o método original com os mesmos dados, seed e configuração.

## 4. Executar Notebook 05

Reinicie o kernel e execute todas as células em `ExecutionMode.REAL`. Verifique `Artifact generation status` e `paper/artifact_manifest.json`. O status pode continuar `partial`: isso é correto caso persistam ausências de dados.

Esperado sem novos benchmarks:

- Gerar `fig_measured_visitation_entropy` usando históricos existentes, se >=2 seeds tiverem checkpoints compatíveis.
- Gerar o dendrograma somente após existir uma matriz de confusão validada.
- **Continuar omitindo** a figura de exposição × delta recall se não houver dados por seed.
- **Continuar omitindo** a figura dos quatro quadrantes se os atomic runs históricos não contiverem essas quatro avaliações.
- **Continuar omitindo** curvas precision-recall quando não houver pontos PR ou scores reais e interação de razões quando só houver 1:50.

## 5. Limitações e experimentos futuros

O patch acrescenta ao `benchmark_runner.py` a persistência de `best_validation_metrics`, `last_metrics`, `exposure_diagnostics`, `per_class_recall`, `evaluation_quadrants` e `computational_costs` nos JSONs **de novas execuções**. Não modifica os resultados existentes.

Quando uma futura rodada REAL for executada, as figuras de exposição/recall e quatro quadrantes poderão ser geradas automaticamente *se* todos os métodos e as mesmas seeds estiverem disponíveis. Isso não implica autorização para executar novos experimentos apenas para preencher figuras.

## 6. Segurança, diferenças científicas e checagens

- Não altera o algoritmo, modelos, seeds, hiperparâmetros ou os resultados históricos.
- Não aceita artefatos SMOKE para publicação REAL nem arquivos de confusão sem sidecar/hash coerente.
- Não aceita um snapshot agregado como resultados multisseed.
- Não cria valores ilustrativos em REAL.
- O dendrograma continua utilizando a implementação Ward histórica. Validar sua adequação geométrica antes de fazer inferências metodológicas sobre Ward.
- O artigo deve descrever figuras ausentes como ainda não medidas, não como falha do MCTS.
- A compilação LaTeX ainda exige uma instalação funcional de `pdflatex` no servidor.
