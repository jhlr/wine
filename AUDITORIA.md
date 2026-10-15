# Auditoria: requisitos do ENUNCIADO.md contra a entrega

Fora de `lab-02/` de propósito (não faz parte da entrega). "NB" = `lab-02/lab02.ipynb` (seção pelo título), "README" = `lab-02/README.md`, evidências em `lab-02/resultados/`. Estado: OK = atendido e conferido na execução salva; PARCIAL = atendido com ressalva descrita.

## Parte A
| ID | Requisito (enunciado) | NB / README | Evidência | Estado |
|---|---|---|---|---|
| A1 | Classe herdando de `Dataset`; recebe tabela de uma partição, características, média/desvio opcionais; docstrings de `__init__`, `__len__`, `__getitem__` | NB "A1" (docstrings impressas) / README A1 | `lab02lib/data.py` | OK |
| A1 | Estatísticas aprendidas só no treino; val/teste reutilizam; desvio zero tratado; convenção declarada (`ddof=0`) | NB "A1", "A2" (verificações) / README A1 | `a1_estatisticas.csv` | OK |
| A1 | X `[11]` float32, y escalar float32; lote `[B,11]`, `[B]`; sem NaN/inf; evidência de estatísticas do treino na val/teste; média da val não exigida zero | NB "A2" (`verificar_datasets`, linhas OK) | `a1_estatisticas.csv` | OK |
| A2 | Três carregadores; quatro parâmetros justificados com o efeito; avaliação sem descartar; ordem alinhada (`obter_escores`) | NB "A2" (tabela lida dos loaders + justificativas) / README A2 | `a2_loaders.csv` | OK |
| A2 | "Meça antes de alegar que workers aceleram" | README A2: declaro que **não medi**; `num_workers=0` sem alegar ganho | n/a | PARCIAL (medição opcional não feita, sem alegação) |
| A3 | Lotes 1, 8, 64, 512; época completa; mesmo modelo/dados/dispositivo; mesmo estado inicial e Adam novo; rede sem BatchNorm; nada descartado | NB "A3" / README A3 | `a3_tempos.csv` | OK |
| A3 | Aquecimento fora da medição; 3 repetições; mediana, exemplos, passos | NB "A3" | `a3_tempos.csv`, `a3_resumo.csv` | OK |
| A3 | Gráfico tempo × lote com unidades e equipamento; explicação (ganho, custo por passo, nº de atualizações), sem inventar retorno decrescente | NB "A3" (texto gerado do CSV) / README A3 | `a3_tempo_x_lote.png`, `a3_hardware.json` | OK (rep. 1 do lote 512 mais lenta; declarado) |
| A4 | Subconjunto só do treino com resto 1 (65/64) confirmado numericamente | NB "A4" | `a4_log.txt` | OK |
| A4 | BatchNorm1d em `train()`, `drop_last=False`; captura específica; mensagem original, tamanho do lote que falhou | NB "A4" | `a4_log.txt` | OK |
| A4 | Correção com custo explicado; sem `except` genérico; subconjunto só para demonstração | NB "A4" / README A4 | `a4_log.txt` | OK |

## Parte B
| ID | Requisito | NB / README | Evidência | Estado |
|---|---|---|---|---|
| B1 | Cinco cenários, 3 a 5 linhas, partição, uso futuro, informação que atravessa, hipóteses declaradas; cenário 5 com tempo e identidade | README B1 (NB aponta para o README) | README | OK |
| B2 | Pergunta de uso; 60/20/20 aproximado, semente fixa, perfis idênticos juntos (`GroupShuffleSplit`, duas chamadas) | NB "B2" / README B2 | `lab02lib/split.py` | OK |
| B2 | Auditoria: linhas por conjunto, `id_linha` disjunto, cobertura, 0 perfis compartilhados por par, ambas as classes e proporções | NB "B2" (asserts + tabelas) | `b2_auditoria.csv`, `b2_pares.csv` | OK |
| B2 | Semente não escolhida por métrica | NB "B2" / README B2 | semente 42 predeclarada | OK |
| B2 | Limites: índices sem sobreposição não provam ausência de características repetidas; sem tempo, produtores, outros domínios | README B2 | README | OK |

## Parte C
| ID | Requisito | NB / README | Evidência | Estado |
|---|---|---|---|---|
| C1 | Duas formas, ao menos uma real e uma entre 2/4/7 (formas 1 e 4, dados reais) | NB "C" / README C | `c_resultados.csv` | OK |
| C2 | Só treino+validação da B2; teste reservado; variáveis `c_`; assert de ausência do teste | NB "C" (asserts, checagem de nomes) | NB (saídas OK) | OK |
| C3 | Por forma: informação protegida, etapa onde atravessa, dev/avaliação/modelo/métrica declarados antes | NB "C" (Markdown anterior) / README C | NB | OK |
| C4 | Procedimento incorreto identificado; versão corrigida comparável (mesma semente/init/embaralhamento) | `lab02lib/leakage.py` (`executar_forma1`, `executar_forma4`) | `c_resultados.csv` (20 execuções) | OK |
| C5 | Dois valores e diferença com unidade (`100·(errado−correto)`) | NB "C" / README C (tabela) | `c_resumo.csv`, `c_tabela.csv` | OK |
| C6 | Explicação que sustenta ou limita; efeito pequeno não invalida; sem inflar nem trocar sementes | README C (um parágrafo por forma); NB texto gerado | `c_mecanismo_forma4.csv` | OK (forma 1 ≈ 0; forma 4 pequena; reportado como saiu) |
| C7 | Tabela e um parágrafo por forma; transformação declarada; 5 sementes predeclaradas, desvio ≠ erro-padrão | README C | `c_tabela.csv` | OK |
| C8 | Bônus: oitava forma (sobreamostragem antes de separar) | NB "Bônus" / README C (Bônus) | `bonus_resultados.csv`, `bonus_resumo.csv` | OK (efeito na AUC-ROC nulo e de sinal misto; mecanismo mostrado por contagem) |

## Parte D
| ID | Requisito | NB / README | Evidência | Estado |
|---|---|---|---|---|
| D7 | Rede 11→16→1, Adam, BCE sem ponderação; parada pela perda de validação (100/10/1e-4); sem loader de teste na função | NB "D7" (assinatura conferida) / README D7 | `lab02lib/train.py` | OK |
| D7 | Cópia independente dos melhores pesos; restauração; época de parada, melhor época, motivo verdadeiro; curvas | NB "D7" (prova `torch.equal` e perda recalculada) | `d7_historico.csv`, `d7_curvas_perda.png` | OK (parou por paciência na época 59; melhor 49) |
| D4 | Ação, FP/FN, `C_FP>0`, `C_FN>0`, unidade, hipótese explícita, justificativa da razão | NB "D4" / README D4 | `lab02lib/decision.py` | OK |
| D5 | Escores da validação do modelo restaurado; candidatos únicos e extremos; desempate declarado; comparação com 0,5 (total e por observação); validação reutilizada reconhecida | NB "D5 e D6" / README D5/D6 | `d5_busca_limiar.csv`, `d5_comparacao.csv`, `d5_curva_custo.png` | OK |
| D6 | Fórmula `C_FP/(C_FP+C_FN)`; investigação das diferenças; calibração e Brier sem provar calibração; sem ajuste após o teste | NB "D5 e D6" / README D5/D6 | `d6_calibracao.png`, `d6_calibracao.csv`, `d5_d6_meta.json` | OK |
| D-freeze | Regra, modelo, entradas, estatísticas e limiar fixados antes da avaliação final | NB "Congelando a decisão" | `decisao_congelada.json` (hash dos pesos, conferido na D1) | OK |
| D1/D2 | Teste só depois de fixar escolhas, escores uma vez; matriz rotulada (linhas real, colunas previsto, ordem 0/1), acurácia, precisão, revocação, F1, AUC-ROC dos escores, AUC-PR trapezoidal, AP rotulado, prevalência, convenção sem positivos | NB "D1, D2 e D3" / README D1/D2 | `d_tabela_final.csv`, `d_matrizes.csv`, `d_areas.json` | OK |
| D2 | Trivial com classe majoritária do treino; AUCs "não reportada" justificadas; tabela de 3 linhas; custo/observação | NB / README | `d_tabela_final.csv` | OK |
| D3 | ROC com diagonal, PR com prevalência do teste, eixos e áreas | NB "D1, D2 e D3" | `d_curvas_roc_pr.png` | OK |
| D8 | Parágrafo: ação, limiar, métricas e custos no teste (hipotéticos), erro que merece atenção, limitações; sem alegar fora do avaliado | README D8 | README | OK |
| D-extra | Comparar acurácia e revocação do trivial com a rede (FAQ) | README D1/D2 e D8 | `d_tabela_final.csv` | OK |

## Checklist da seção 9 e penalidades
| Item | Estado | Evidência |
|---|---|---|
| B2 completo | OK | `b2_auditoria.csv`, `b2_pares.csv` |
| A: estatísticas do treino, tipos/formas, 4 tempos, erro de BatchNorm | OK | `a1_estatisticas.csv`, `a3_tempos.csv`, `a4_log.txt` |
| C: duas formas, comparação com unidade, teste preservado | OK | `c_resultados.csv` |
| D: melhor estado restaurado, limiar na validação, teste depois, matriz/curvas/custos | OK | `decisao_congelada.json`, `d_*` |
| Relatório sem valor inventado; resultados pequenos explicados | OK | README gerado a partir dos CSV, com `assert` nas frases que dependem do sinal do resultado |
| Entrega: notebook executado do zero (kernel novo), README, USO-DE-IA, fonte e versões | OK (links e declaração pendentes do aluno) | `lab02.ipynb` salvo com saídas, `pendencias == []` |
| Nenhum marcador TODO / célula "pendente" | OK | busca no notebook, README, USO-DE-IA e lib |
| Vazamento na própria D | Nenhum encontrado | normalização do treino; limiar na validação; teste aberto uma vez após congelar (hash conferido) |
| Treino reportado como avaliação | Não | tabelas D1/D2 vêm do teste; D5 vem da validação e está rotulada como desenvolvimento |
| `USO-DE-IA.md` presente | OK | declaração final fica para o aluno |

## Pontos de atenção para a revisão
1. Na A1 a verificação de formas percorre o carregador de teste (só características e rótulos, sem modelo); nenhum escore ou métrica do teste existe antes da D1.
2. O efeito do bônus na AUC-ROC foi nulo e de sinal misto (reportado assim); a acurácia entre as variantes do bônus não é comparável (prevalências diferentes).
3. A diferença da forma 4 mistura o fim do vazamento com um treino menor.
4. Os tempos da A3 mudam de uma execução para outra; todos os demais números se repetiram.
