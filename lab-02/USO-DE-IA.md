# Uso de IA generativa no Lab 2

Nome: João Rietra

Usei IA generativa neste laboratório. O trabalho foi feito por um agente de código a partir de um prompt de especificação (SDD) que está anexado.

| Ferramenta/modelo | Finalidade e pedido | Trecho ou ideia aproveitada | Como verifiquei | O que descartei e por quê |
|---|---|---|---|---|
| Claude Code (modelo `claude-sonnet-5-5`, executado de forma autônoma) | Implementação do código do laboratório, execução do notebook, rascunho do README e do notebook, a partir do prompt anexado (`SDD-prompt.md`). | O código inteiro do pacote `lab02lib/`, o notebook `lab02.ipynb` e os rascunhos dos textos. Tudo isso ainda precisa da minha revisão (ver Declaração). | O agente verificou com: `assert` de guarda visíveis no notebook (cobertura da partição, zero perfis compartilhados, formas e tipos, estatísticas do treino, restauração da parada antecipada, hash dos pesos congelados, escores do teste obtidos uma vez); execução do notebook em kernel novo com `nbconvert` (sem erros, lista de pendências vazia); SHA-256 dos arquivos do curso conferidos contra o manifesto e `apoio.py` idêntico ao original; `mypy` sem erros em `lab02lib/`; segunda execução reproduzindo os mesmos números (só os tempos da A3 variaram); tabela de auditoria `AUDITORIA.md` (na raiz do repositório, fora de `lab-02/`) ligando cada requisito do enunciado a uma evidência. Os números dos textos foram lidos de `resultados/`. | Rascunhos que o agente trocou durante o trabalho: uma primeira versão da correção do BatchNorm com carregadores em passo travado (reescrita com um carregador espelho conferido tensor a tensor); uma configuração de logs do MLflow aplicada antes do `import mlflow`, que não surtiu efeito (movida para depois do import); uma linha `assert ... or True` sem efeito e um uso de `# noqa`, removidos antes de qualquer commit; o uso de `pyplot` com backend fixo, trocado pela API `Figure`; a legenda da curva PR, que cobria a curva, foi movida. Nenhum resultado numérico foi descartado: nada foi repetido para melhorar métrica. |

Anexo: o prompt usado está em `SDD-prompt.md` (nesta mesma pasta).

## Declaração
[preencher após revisar]
