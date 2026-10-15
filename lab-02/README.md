# Lab 2 — Relatório individual

Nome: [preencher]  
Turma: [preencher]  
Link do repositório: [preencher]

Substitua os marcadores pelas suas medições e análises. Consulte ENUNCIADO.pdf para os requisitos integrais.

## Reprodução
Python, bibliotecas, sistema, dispositivo e semente: [preencher].
Comando ou sequência para executar: [preencher].
Modificações feitas em apoio.py: [declarar ou informar que não houve].

## Base e definição do alvo
Fonte, características, regra do alvo, unidade de observação e limitações: [preencher].

## B1 — Cinco cenários
1. Avaliações de produtos: [3–5 linhas].
2. Turbinas: [3–5 linhas].
3. Spam: [3–5 linhas].
4. Radiografias: [3–5 linhas].
5. Transações, tempo e clientes: [3–5 linhas].

## B2 — Auditoria da separação
Pergunta de uso e estratégia: [preencher].

| Conjunto | Linhas | Positivos | Proporção positiva | Perfis |
|---|---:|---:|---:|---:|
| Treino | | | | |
| Validação | | | | |
| Teste | | | | |

Cobertura e interseções de índices/perfis: [inserir verificações].

## A1 — Dataset
Formas, tipos, estatísticas e verificação: [preencher].

## A2 — DataLoader
| Conjunto | batch_size | shuffle | drop_last | num_workers | Justificativas |
|---|---|---|---|---|---|
| Treino | | | | | |
| Validação | | | | | |
| Teste | | | | | |

## A3 — Tempo por época
| Lote | Repetição 1 (s) | Repetição 2 (s) | Repetição 3 (s) | Mediana (s) | Passos |
|---|---|---|---|---|---|
| 1 | | | | | |
| 8 | | | | | |
| 64 | | | | | |
| 512 | | | | | |

Gráfico, método de medição e análise: [preencher].

## A4 — BatchNorm
Condição do erro, mensagem capturada, correção e consequência: [preencher].

## C — Duas formas de vazamento
| Forma | Dados e partições internas | Métrica | Incorreto | Corrigido | Diferença/unidade |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

Experimento 1 — mecanismo, comparação e limites: [preencher].
Experimento 2 — mecanismo, comparação e limites: [preencher].
Como preservei o teste principal: [preencher].

## D7 — Treinamento e parada
Arquitetura, otimizador, perda, máximo de épocas, paciência e melhora mínima: [preencher].
Melhor época, época de parada, motivo e restauração: [preencher].
Curvas de perda: [inserir].

## D4 — Matriz de custos
Ação, C_FP, C_FN, unidade e justificativa: [preencher].
Hipóteses e origem dos custos: [preencher].

## D5 e D6 — Escolha na validação
Candidatos, desempate, custo em 0,5, custo escolhido e diferença: [preencher].
Limiar da fórmula, comparação e investigação: [preencher].
Configuração congelada antes do teste: [preencher].

## D1 e D2 — Avaliação final
Proporção positiva, classe trivial e convenções: [preencher].
Matrizes de confusão e tabela de métricas/custos: [inserir].
Identifique AUC-PR trapezoidal e AP separadamente, caso reporte ambas.

## D3 — Curvas
ROC com diagonal aleatória e PR com prevalência do teste: [inserir].

## D8 — Recomendação
[Escreva o parágrafo final para quem decide.]

## Referências e limites
[Fontes, dependências de domínio, incertezas e o que não foi avaliado.]
