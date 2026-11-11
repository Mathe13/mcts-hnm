# MCTS-HNM — Roadmap revisado para a próxima campanha experimental

**Escopo:** Tier 1 (CIFAR-100, embeddings existentes), implementação do MCTS-HNM, baselines, diagnóstico de amostragem, avaliação estatística e artefatos.  
**Objetivo:** produzir comparações corretas e interpretáveis; identificar por que o Enhanced perde desempenho em parte dos cenários antes de propor novas políticas.  
**Fonte de evidência:** código, configurações, notebooks executados e resultados persistidos. O conteúdo de `paper/` é um **template** e não deve servir como evidência ou premissa científica.  
**Regra operacional:** preparar e testar as mudanças, mas **não executar automaticamente benchmarks REAL extensos**.
**Extensão de desbalanceamento:** planejar um estudo controlado de **1:50, 1:100 e 1:500**, com 1:500 como *stress test* condicionado ao piloto e ao orçamento; não interpretar `pos_fraction` como razão real.

## 0. Regras gerais para o agente

1. Inspecionar as implementações reais antes de alterar arquivos; os caminhos abaixo são pontos de partida, não contratos imutáveis.
2. Preservar o código e os resultados históricos; criar variantes com nomes próprios, sem sobrescrever `strict` ou `enhanced`.
3. Classificar cada achado como **defeito confirmado**, **limitação de desenho**, **hipótese** ou **resultado negativo legítimo**.
4. Alterar uma variável algorítmica por vez, ou adotar um fatorial explicitamente definido.
5. Não usar o teste oficial para construir árvores, selecionar configurações, ajustar limiares, escolher checkpoints ou decidir quando interromper.
6. Preservar isolamento entre SMOKE/REAL, seeds, splits, caches, checkpoints e versões de árvore.
7. Não regenerar embeddings ou alterar datasets sem justificativa metodológica concreta.
8. Não fabricar séries, métricas ausentes, visitas ou resultados; marcar dados faltantes explicitamente.
9. Medir orçamento de otimização **e** custo computacional total, incluindo seleção de candidatos.
10. Toda correção com impacto no comportamento deve incluir testes e descrição dos resultados que precisam ser reexecutados.

## Questões científicas prioritárias

- **RQ1 — Corretude:** runners diferentes implementam o mesmo protocolo quando recebem a mesma configuração?
- **RQ2 — Multiclasse:** a perda de desempenho vem da escolha de classes pelo UCB, da homogeneidade do minibatch ou de ambos?
- **RQ3 — Enhanced:** qual o efeito isolado da normalização Q, do Dual-Stream e da interação entre eles?
- **RQ4 — Hierarquia:** o efeito de Ward decorre da informação da matriz de confusão ou de propriedades estruturais da árvore?
- **RQ5 — Eficiência:** o eventual ganho se mantém quando contabilizado o custo de amostragem e treinamento?
- **RQ6 — Desbalanceamento:** sob quantidade fixa de positivos únicos e orçamento de treinamento equivalente, a vantagem relativa do HNM muda entre razões **1:50, 1:100 e 1:500**?

---

## Fase A — Auditoria e integridade do protocolo (P0; bloqueante)

### A1. Congelar e inventariar a versão histórica

Inspecionar `src/mcts/`, `src/trees/`, `src/data/splits.py`, `src/training/` e notebooks `03` e `05`. Criar `docs/experimental_audit.md` e um manifesto por execução com commit, data, seed, configuração resolvida, versões de dependências, hardware, identidade/hash dos embeddings, hashes dos splits, árvore e resultados. Os históricos ficam imutáveis.

**Aceite:** qualquer número existente pode ser rastreado até uma execução e suas condições; resultados de diferentes campanhas não são combinados silenciosamente.

### A2. Corrigir a definição operacional do desbalanceamento

Distinguir três variáveis: (i) razão real `n_negative/n_positive` no conjunto de treinamento, (ii) número de positivos **únicos** disponíveis e (iii) fração positiva por minibatch. Não tratar `pos_fraction=0.02` como equivalente a razão final `1:50`.

Implementar um construtor de splits estratificado com `target_neg_pos_ratio` e `target_positive_count` separados. Se não houver exemplos suficientes para satisfazer a configuração, falhar explicitamente; nunca ajustar a razão em silêncio. Fixar os mesmos índices disponíveis para todos os métodos sob a mesma combinação `(seed, classe-alvo, razão)`.

Preparar **três razões reais de negativos por positivo no treino: 1:50, 1:100 e 1:500**, com a mesma quantidade de **positivos únicos** em todas elas. Exemplo de desenho condicionado à disponibilidade de dados em `train_fit`: fixar **80 positivos por classe-alvo/seed**, selecionar respectivamente **4.000 / 8.000 / 40.000 negativos**. As contagens são ilustrativas: validar limites após a separação treino/validação/calibração, sem reposição para definir as amostras disponíveis e sem mover exemplos de validação/teste para o treino. Se não for viável, reduzir **a mesma** quantidade de positivos em todos os cenários ou declarar o cenário inviável, sem ajustar em silêncio.

Selecionar os **mesmos positivos por seed e classe-alvo** em todas as razões. Amostrar negativos de modo estratificado entre classes negativas, preferencialmente com conjuntos aninhados (`N_50 ⊂ N_100 ⊂ N_500`) para reduzir variação de composição; registrar índices/hashes e diversidade de classes negativas. Reconhecer que aumentar a razão também aumenta o número/diversidade de negativos disponíveis, aspecto inerente ao desenho de quantidade fixa de positivos, e não atribuir toda diferença apenas à prevalência numérica.

Manter a **mesma composição do minibatch** (fração positiva e/ou política focal) para uma comparação primária que isole o efeito do conjunto disponível; registrar explicitamente que a prevalência dos minibatches difere da prevalência do treino. Eventual experimento com composição proporcional ao dataset é **ablação independente**, não uma alteração implícita entre razões. Registrar razão efetiva, prevalência e contagens únicas também em validação/teste, **sem artificialmente reamostrar o teste oficial**. O teste One-vs-Rest do CIFAR-100 tem razão natural próxima de 1:99, de modo que 1:500 caracteriza desbalanceamento do **treinamento**, não do teste.

**Aceite:** testes com contagens exatas, classes ausentes, ratios inviáveis, amostragem aninhada/estratificada, positivos compartilhados entre razões e invariância dos splits entre estratégias; as razões empíricas 50, 100 e 500 são verificadas antes de cada execução.

### A3. Separar dados por finalidade

Usar `train_fit` para gradientes e mineração; `validation` para checkpoints, hiperparâmetros e limiar de F1; `test` oficial apenas na avaliação final. Se a hierarquia de confusão for construída a partir de predições, usar dados de calibração internos ao treino, idealmente out-of-fold, **sem reutilizar o teste**. Documentar quando a calibração da árvore compartilha dados com o treinamento e quais vieses isso pode introduzir.

Remover fallbacks para dados de teste quando validação/calibração não forem fornecidas: falhar de forma explícita. Curvas durante o treinamento são de validação, nunca de teste.

**Aceite:** testes de integração detectam uso indevido do teste; nenhuma configuração é escolhida a partir de resultados finais.

### A4. Garantir equivalência entre runners

Revisar `run_benchmark()`, `run_sensitivity_ablation()`, `train_scenario1()` e funções intermediárias. Usar uma configuração compartilhada, com defaults resolvidos e salvos: split, classe, arquitetura, inicialização, otimizador, batch, atualizações, `c`, `alpha`, recompensa, política, avaliação e checkpoint.

Executar comparação determinística curta pelos dois caminhos **somente quando representam o mesmo experimento**. Confrontar índices, perdas, atualizações, estado da árvore, predições e métricas com tolerâncias adequadas; documentar divergências intencionais.

**Aceite:** ausência de diferenças acidentais entre protocolos equivalentes e propagação correta de `c`/`alpha`.

### A5. Corrigir granularidade da persistência e da estatística

Guardar uma linha/arquivo por **run_id** (`scenario`, classe, razão, árvore, estratégia, seed, commit), e históricos de cada passo ou janela. Não usar apenas `strategy` como chave de curvas/diagnósticos; não reter apenas a primeira seed ou sobrescrever execuções posteriores. Derivar médias e figuras a partir dos registros individuais.

Parear comparações por **mesma seed, mesmos índices/split, classe-alvo e orçamento**. Evitar duplicar Uniform/OHEM no laço de árvores quando esses métodos não usam árvore; se for necessário repetir para pareamento, explicitar que são execuções independentes.

**Aceite:** sumarização e pareamento falham quando chaves são incompatíveis; gráficos podem ser reconstruídos a partir dos resultados persistidos.

### A6. Validar a construção das hierarquias

Para Ward, não pressupor que `sqrt(2*distance)` transforme qualquer matriz de dissimilaridade em distância euclidiana. Verificar a estrutura da matriz (incluindo euclidianidade/semidefinição positiva da representação apropriada) ou aplicar Ward diretamente sobre vetores euclidianos válidos, como perfis de confusão normalizados. Incluir Average Linkage como alternativa para dissimilaridades não euclidianas, preservando a implementação histórica como controle.

Conferir alinhamento de IDs/rótulos, origem da matriz de confusão, profundidade, tamanhos de subárvores, balanceamento e custo de seleção. Identificar cada árvore por hash/configuração.

**Aceite:** testes com matrizes e árvores pequenas conhecidas; método de clustering matematicamente justificado.

**Gate A — Não avançar** se splits, teste, runner, persistência ou pareamento estiverem inconsistentes.

---

## Fase B — Instrumentação e piloto diagnóstico (P0)

### B1. Observabilidade uniforme por execução

Persistir por seed e intervalo: `train_loss`, `val_loss`, `val_F1`, `val_AUPRC`, precision, recall, macro-F1/accuracy quando aplicáveis, learning rate, número de updates, tempo decorrido e gradiente (se barato). Registrar as distribuições de classes no conjunto disponível, nos minibatches e entre os exemplos efetivamente usados para gradiente.

Distinguir **visitas a nós**, **seleções de folhas**, **instâncias selecionadas**, **instâncias únicas** e **exposição por classe nos gradientes**. Incluir entropia normalizada de exposição, classes nunca expostas, cobertura acumulada, participação de superclasses, correlação exposição–recall e Worst-10 Recall.

### B2. Diagnóstico da política MCTS

Salvar amostras periódicas de recompensa bruta, recompensa após suavização, valor Q, normalização aplicada, termo de exploração, score UCB e contadores por nó. Verificar propagação para ancestrais, estado mantido entre updates, tratamento de scores negativos e probabilidade não nula de exploração.

Testar árvores minúsculas e recompensas determinísticas. Diferenciar comportamento esperado de UCB de possível starvation resultante de política, topologia ou recompensa não estacionária.

### B3. Checkpoints e limiares

Salvar `last` e `best_validation` (por métrica predefinida), além de estado do otimizador, sampler, árvore e RNG para retomada. Comparar **mesmo orçamento fixo N** usando `last` versus `best_validation`; não introduzir early stopping como vantagem orçamentária oculta na comparação principal.

No binário, reportar F1 com limiar 0,5, F1 com limiar fixado por validação e AUPRC. Guardar o limiar escolhido em validação e aplicá-lo uma única vez ao teste final.

### B4. Piloto curto e falsificável

Realizar testes SMOKE e um piloto REAL reduzido (por exemplo, 2 seeds, 200–500 updates) **somente mediante execução explícita**. Incluir Uniform, Uniform Grouped e Enhanced no multiclasse. Para o binário, validar **1:50 e 1:100** no piloto; incluir **1:500** com pelo menos Uniform e Enhanced para verificar estabilidade, batches, custos, prevalência e seleção de checkpoint. Não interpretar o piloto como estimativa final de efeito.

**Gate de viabilidade 1:500:** verificar que os conjuntos contêm a quantidade estipulada de positivos/negativos, que há estimativas de validação suficientemente informativas, que a seleção não depende de reposição oculta fora dos minibatches, que nenhuma política produz resultados não finitos e que custos são compatíveis com o orçamento. Se falhar, registrar 1:500 como não viável nesta campanha ou manter somente como estudo exploratório com limitações explícitas.

**Gate B:** histórico completo por seed; política validada; causa da queda multiclasse ao menos distinguível entre cobertura de classes e composição do batch. Se a causa permanecer ambígua, manter apenas diagnóstico na próxima campanha, sem adicionar novas políticas.

---

## Fase C — Ablações de mecanismo, com variantes imutáveis (P1)

### C1. Controle Uniform Grouped — obrigatório

No Scenario 2, comparar sob os mesmos dados, arquitetura, inicialização, batch e passos:

| Estratégia | Classe focal | Batch (exemplo com 64) |
|---|---|---|
| Uniform | Nenhuma | 64 uniformes |
| Uniform Grouped | Classe uniforme | 32 focais + 32 uniformes |
| Enhanced | Classe pelo MCTS | 32 focais + 32 uniformes |

Se Uniform Grouped também piorar substancialmente, investigar homogeneidade dos minibatches; se não, investigar seleção UCB/recompensa. Adicionar uma medição da diversidade de rótulos por batch e de cobertura acumulada por classe.

### C2. Ablar Q normalization × Dual-Stream

Primeiro confirmar no código todas as diferenças efetivas entre Strict e Enhanced. Se os dois fatores forem controláveis mantendo os demais componentes constantes, usar a grade:

| Variante | Q normalizado | Dual-Stream |
|---|---|---|
| Strict-equivalente | Não | Não |
| QNorm-only | Sim | Não |
| DualStream-only | Não | Sim |
| Enhanced-equivalente | Sim | Sim |

Não afirmar que o fatorial é uma comparação Strict × Enhanced perfeita se houver outras diferenças (recompensa, momentum, política de seleção, atualização de árvore). Nesse caso, isolar as diferenças adicionais e registrá-las antes de interpretar efeitos.

### C3. Controles simples de seleção — prioritários se houver orçamento

Adicionar pelo menos um seletor **greedy por EMA de loss por classe** (sem UCB), com mesma árvore ou composição do batch, e considerar um UCB sem hierarquia (classes como braços). Eles ajudam a separar o valor da hierarquia, da exploração e da dificuldade amostral. Hiperparâmetros são definidos em validação.

### C4. Políticas corretivas condicionais — não habilitar por padrão

**Somente se B confirmar baixa cobertura**, avaliar mistura Uniform–MCTS, exposição mínima por classe ou recompensa normalizada por classe. Testar uma variante por vez, preservando as originais; não ajustar políticas com base no teste. Se as falhas vierem do batch, priorizar menor fração focal ou composição multiclasse, não cotas de exploração.

### C5. Hierarquias

Comparar Semantic, Ward válido e Average Linkage quando apropriado; incluir árvore aleatória com topologia/folhas comparáveis **se houver orçamento**. Registrar estrutura e custo, além de F1/AUPRC/recall. Não concluir que a semântica da hierarquia foi responsável por uma diferença sem controlar topologia e disponibilidade de amostras.

**Gate C:** ablações permitem atribuir efeitos aos fatores, ou declarar explicitamente que os dados são insuficientes para isso.

---

## Fase D — Campanha científica final e artefatos (P1/P2)

### D1. Congelar protocolo antes da execução

Pré-registrar internamente configurações, classes, seeds, split, quantidade de updates, métricas primárias, checkpoints e hipóteses. Para limitar custo, priorizar:

1. **Scenario 1 binário — 1:50 e 1:100 (principal), 1:500 (stress test condicionado ao Gate B):** Uniform, OHEM, Strict e Enhanced; pelo menos **5 seeds pareadas** por razão, com **mesma classe-alvo e mesmos positivos únicos** em todas as razões. Congelar número de updates, batch, estratégia de checkpoint, validação e composição do minibatch. Priorizar a comparação completa em 1:50/1:100; incluir a grade completa de 1:500 após confirmar sua viabilidade. A matriz de 3 razões × 4 métodos × 5 seeds equivale a **60 treinamentos binários**, sem contar piloto/ablação.
2. **Scenario 2 multiclasse:** Uniform, Uniform Grouped, Strict, QNorm-only, DualStream-only, Enhanced; pelo menos 5 seeds pareadas. Se o orçamento restringir, executar primeiro Uniform, Uniform Grouped, Strict e Enhanced e depois as duas ablações.
3. **Hierarquias:** comparação Semantic × Ward usando uma política MCTS fixada; Average/Random conforme resultado do Gate C e orçamento.

Não misturar **1:500 de treinamento** com uma suposta **prevalência 1:500 no teste**. Usar teste oficial não reamostrado e a mesma população de avaliação para todas as razões; relatar que diferenças observadas se referem à robustez sob *desbalanceamento de treino*. Opcionalmente reportar o protocolo natural (~1:99, com todos os positivos de treino) como **referência separada**, sem combiná-lo causalmente com o desenho de 80 positivos fixos.

Estender para mais classes-alvo, razão 1:10, expansão dinâmica e Tier 2 **somente após** interpretar a campanha principal. Aumentar número de seeds se a incerteza observada justificar; não escolher seeds favoráveis.

### D2. Métricas e orçamento

**Binário:** F1 minoritária (com limiar documentado), **AUPRC como métrica primária recomendada para avaliar ranking sob desbalanceamento**, precision/recall, FPR@95 se estável e definido. Explicitar a prevalência do teste e o valor esperado da AUPRC aleatória (sua prevalência); como o teste é fixo, não inferir comportamento em uma população real 1:500 somente a partir dessas métricas. Reportar limiar escolhido em validação e, se relevante, calibração separada por razão.  
**Multiclasse:** accuracy, macro-F1, balanced accuracy, Worst-10 Recall, recall e exposição por classe.  
**Eficiência:** updates, exemplos usados no gradiente, forwards adicionais de seleção/OHEM, wall-clock, CPU/GPU e pico de memória, sempre com hardware documentado.

Comparar desempenho em função de updates **e** tempo/custo total. `best_validation` e `last` devem ser distinguidos nas tabelas; não comparar uma estratégia no melhor checkpoint com outra somente no último.

### D3. Estatística

Calcular diferenças **pareadas por seed e split** dentro de cada razão; para contrastes entre razões, preservar o pareamento de seed e dos positivos e identificar a interação **método × razão**, sem tratar as três razões como replicações independentes. Reportar efeitos por razão e interação com IC quando a amostragem permitir, além de intervalos de confiança e tamanho de efeito. Reportar valores individuais, média e desvio-padrão. Com apenas cinco seeds, interpretar intervalos/testes com cautela. Corrigir multiplicidade para famílias de comparações predefinidas. Ausência de significância não comprova equivalência. A unidade de inferência primária é a execução/seed, não cada amostra de teste tratada como replicação independente do método.

### D4. Artefatos e notebooks

O notebook `03` deve permitir execução isolada de cenários sem misturar caches e exibir proveniência. O `05` deve consumir somente resultados REAL validados e gerar: curvas de validação por seed e agregadas, banda de variabilidade identificada, heatmap de exposição/visitação com unidade definida, distribuição por classe, recall por classe, ablação 2×2 de QNorm × Dual-Stream e gráficos desempenho-versus-custo. Exportar tabelas com n, seeds, condições e métricas; não preencher lacunas com valores simulados. LaTeX é uma etapa de apresentação, não critério de sucesso científico.

**Gate D:** resultados finais rastreáveis; conclusões restritas às comparações efetivamente executadas; nenhuma afirmação derivada do template em `paper/`.

---

## Testes obrigatórios (transversais)

- **Unitários:** splits e ratios **1:50/1:100/1:500**, preservação dos mesmos positivos, aninhamento/estratificação dos negativos, falta de exemplos e ausência de reposição na construção do dataset, UCB, normalização, reward, momentum, atualização de ancestrais, amostragem, estados RNG, topologia/índices de árvore, seleção de checkpoint, métrica e limiar.
- **Integração:** equivalência de runners, isolamento SMOKE/REAL, proibição de teste no treino/calibração, consistência seed/split, retomada determinística quando suportada, proveniência, geração de artefatos.
- **Regressão:** usar configurações pequenas com saída conhecida; tolerâncias numéricas explícitas. Não exigir igualdade bit a bit de GPUs não determinísticas.

## Ordem, dependências e condições de parada

| Etapa | Prioridade | Depende de | Autoriza a próxima? |
|---|---|---|---|
| A1–A5, A6 | P0 | — | Gate A aprovado |
| B1–B3 | P0 | A | Instrumentação validada |
| B4 (piloto) | P0 | A+B | Gate B aprovado |
| C1–C2 | P1 | B | Ablações comparáveis |
| C3–C5 | P1 condicional | B, C1–C2 | Hipótese e orçamento justificados |
| D1–D4 | P1/P2 | A–C | Protocolo congelado |

**Não avançar** em caso de vazamento de teste, ratios incorretos, perda de histórico por seed, pareamento inválido ou instabilidade inexplicada de índices/classes. Diante de resultados ruins, registrar e investigar; não alterar múltiplos componentes para forçar melhoria.

## Entregáveis por fase

Em cada fase, informar: arquivos modificados; achados classificados por grau de evidência; decisões metodológicas; novos testes e resultados; variantes criadas; compatibilidade com baseline; artefatos gerados; experimentos que exigem reexecução; riscos ainda abertos. Manter `docs/experimental_audit.md`, `docs/experimental_protocol.md` e `docs/results_manifest.md` (ou formatos equivalentes) atualizados.

## Definition of Done

- [ ] Histórico imutável e proveniência íntegra.
- [ ] Desbalanceamento, positivos únicos e composição do batch explicitados para **1:50, 1:100 e 1:500**; 1:500 submetido a gate de viabilidade.
- [ ] Splits, runners, seeds e budgets equivalentes onde aplicável.
- [ ] Nenhum teste usado para tuning, árvore, checkpoints ou limiar.
- [ ] Diagnósticos e métricas preservados **por execução**.
- [ ] UCB/árvores testados em cenários determinísticos.
- [ ] Uniform Grouped e ablação QNorm × Dual-Stream executáveis e verificáveis.
- [ ] Hierarquia Ward matematicamente validada ou substituída por alternativa justificada.
- [ ] Custos computacionais registrados.
- [ ] Comparações estatísticas pareadas e reproduzíveis.
- [ ] Notebooks geram artefatos exclusivamente a partir de resultados REAL válidos.
- [ ] Nenhum benchmark REAL completo executado sem autorização explícita.

**Critério de sucesso científico:** entender, com comparações justas e repetíveis, quando o MCTS ajuda, não altera ou prejudica a classificação. Superar Uniform/OHEM não é pré-condição para concluir o trabalho.
