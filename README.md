# Sistema de Recomendação de Livros por Filtragem Colaborativa

Trabalho da disciplina Oficina I. Sistema que recomenda itens a partir do histórico de avaliações de vários usuários (filtragem colaborativa), com interface web para consultar históricos, ver recomendações e avaliar itens.

## 1. Objetivo

Construir e avaliar um sistema de recomendação que:

- usa uma base de interações usuário–item;
- prepara os dados e implementa filtragem colaborativa;
- gera recomendações personalizadas **sem incluir itens que o usuário já conhece**;
- trata o usuário **sem histórico** (cold-start);
- oferece uma interface para consultar histórico, recomendações e avaliar itens;
- é avaliado com pelo menos duas métricas adequadas, com exemplos para cinco usuários.

## 2. Fundamentação

**Filtragem colaborativa** parte da ideia de que usuários que avaliaram itens de forma parecida no passado tendem a gostar dos mesmos itens no futuro. Não usa o conteúdo dos itens, só a matriz de interações usuário × item.

- **Item-based:** dois itens são parecidos se os mesmos usuários os avaliam de modo parecido. Um item é pontuado pelas notas que o usuário deu aos itens mais similares a ele. Como a similaridade entre itens muda pouco, o modelo é estável e permite **explicar** a recomendação ("porque você avaliou X").
- **User-based:** busca os usuários mais parecidos com o alvo e pontua cada item pela média ponderada das notas desses vizinhos.
- **Similaridade:** cosseno sobre notas **centradas na média do usuário**, o que corrige o fato de alguns usuários darem notas sistematicamente altas ou baixas. Com **shrinkage** `n/(n+λ)`, pares com poucos co-avaliadores pesam menos. Só os `k` vizinhos mais similares são mantidos.
- **Cold-start:** sem histórico não há similaridade a calcular. O sistema recomenda os itens mais populares, ordenados pela **média bayesiana** `(v·R + m·C)/(v + m)`. Ela evita que um item com 2 notas 10 supere um clássico com 300 notas de média 8,9.
- **Baselines:** popularidade e aleatório, para mostrar que a filtragem colaborativa agrega valor.

## 3. Dados

Base: **Book-Crossing** ([Kaggle · ruchi798/bookcrossing-dataset](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset)), com avaliações de livros feitas por leitores de uma comunidade online. Notas de 1 a 10. O catálogo (título, autor, ano, categoria) vem do mesmo dataset.

| | Bruto | Após tratamento |
|---|---|---|
| Interações | 1.031.175 | 82.948 |
| Usuários | — | 5.431 |
| Itens (livros) | 271.379 no catálogo | 4.368 |
| Esparsidade | — | 99,65 % |
| Nota média | — | 7,85 |

**Tratamento** (`src/recsys/data.py`):

1. remoção de nulos e conversão de tipos;
2. descarte de notas fora da escala 1–10. No Book-Crossing a nota **0 significa "leu, mas não avaliou"** (feedback implícito, 63 % das interações), não uma nota baixa. Foram descartadas para não contaminar o gosto do usuário;
3. remoção de duplicatas usuário–item;
4. filtro iterativo de esparsidade: usuários com < 5 avaliações e livros com < 8 avaliações saem;
5. mapeamento dos ids para índices contíguos e construção da matriz esparsa (CSR).

O funil do tratamento aparece na aba **Visão geral** da interface. O ISBN é o id do livro e é lido como texto (há zeros à esquerda e o dígito `X`).

`scripts/prepare_data.py` baixa o dataset do Kaggle (download anônimo, sem token) e gera `data/bookcrossing/ratings.csv` e `books.csv`.

### Usar outra base de dados

Basta editar `config.yaml`:

```yaml
dataset:
  interactions_path: data/minha_base.csv
  sep: ";"
  columns: {user: cliente, item: produto, rating: nota, timestamp: data}   # rating/timestamp podem ser null
  items_path: data/produtos.csv        # opcional (títulos); null se não houver
  items_columns: {item: id, title: nome, extra: categoria, author: null, year: null}
  rating_scale: [1, 5]
```

Se a base não tiver nota (`rating: null`), toda interação vale 1,0 (**feedback implícito**) e o modelo passa a trabalhar sem centralização. Sem coluna de data (`timestamp: null`), use `evaluation.split: random`.

## 4. Método

- **Divisão treino/teste:** hold-out **por usuário**. 20 % das avaliações de cada usuário (com ≥ 5 avaliações) vão para teste, sorteadas com semente fixa (`seed: 42`). O Book-Crossing não tem data, então a divisão temporal não é possível; o código a oferece (`split: temporal`) para bases com timestamp. Resultado: 68.133 interações de treino e 14.815 de teste.
- **Item relevante:** nota ≥ 8 no teste.
- **Recomendação:** o modelo pontua todos os livros, **os já conhecidos pelo usuário são mascarados** e sai o top-N.
- **Hiperparâmetros** (`config.yaml`): 30 vizinhos, shrinkage 10, mínimo de 10 votos para a popularidade.
- **Detalhe de implementação:** os 30 vizinhos são os 30 itens (ou usuários) mais similares em toda a base, e não apenas entre os que o usuário avaliou. É uma simplificação comum que mantém o cálculo vetorizado e rápido.

### Métricas

| Métrica | O que mede | Sentido |
|---|---|---|
| **Precision@10** | fração dos 10 recomendados que o usuário de fato gostou | ↑ |
| **Recall@10** | fração dos itens que ele gostou que apareceram no top-10 | ↑ |
| **MAP@10** | precisão média nas posições dos acertos (premia acertos no topo) | ↑ |
| **NDCG@10** | qualidade do ranking, com desconto logarítmico pela posição | ↑ |
| **RMSE / MAE** | erro da nota prevista nos pares de teste | ↓ |
| **Cobertura** | fração do catálogo que aparece em alguma recomendação | ↑ |
| **Diversidade** | 1 − similaridade média entre os itens da lista | ↑ |

## 5. Resultados

Gerados por `python scripts/evaluate.py` (arquivo `results/metrics.csv`). As métricas de ranking são calculadas nos usuários que têm ao menos um item relevante no teste.

| Modelo | Precision@10 | Recall@10 | MAP@10 | NDCG@10 | Cobertura | Diversidade | RMSE | MAE |
|---|---|---|---|---|---|---|---|---|
| Aleatório | 0,0006 | 0,0026 | 0,0006 | 0,0013 | 0,9998 | 0,988 | — | — |
| Popularidade | 0,0023 | 0,0109 | 0,0045 | 0,0071 | 0,004 | 0,952 | 1,713 | 1,352 |
| User-based CF | 0,0060 | 0,0279 | 0,0118 | 0,0186 | 0,866 | 0,967 | 1,687 | 1,262 |
| **Item-based CF** | **0,0131** | **0,0563** | **0,0323** | **0,0451** | **0,942** | 0,941 | **1,667** | **1,214** |

**Leitura:**

- O item-based CF tem NDCG@10 **6,4 vezes maior** que a popularidade (0,045 contra 0,007) e 35 vezes o do aleatório. O user-based fica no meio (2,6× a popularidade). Recomendar pelo gosto de cada leitor compensa.
- A popularidade recomenda quase sempre os mesmos livros (cobertura de 0,4 % do catálogo). O CF chega a 87–94 %, então sugere livros variados.
- O aleatório tem cobertura e diversidade altas, mas acerta quase nada. Essas duas métricas só fazem sentido lidas junto com as de acurácia.
- Nas notas previstas, o item-based tem o menor RMSE (1,667) e o menor MAE (1,214), à frente do user-based e da popularidade.
- Os valores absolutos de precisão são baixos, o que é esperado: a matriz tem 99,65 % de células vazias e o teste só conta como acerto o que o usuário **de fato avaliou** depois. Um livro bom que ele nunca avaliou conta como erro.

### Exemplos para cinco usuários

Em `results/exemplos_5_usuarios.md` há, para cinco usuários de teste, os livros mais bem avaliados, o top-10 recomendado, a justificativa ("por semelhança com…"), os acertos no teste e o caso do usuário sem histórico. Trecho:

> **Usuário 261603** — 22 livros avaliados (série *The Wheel of Time*, nota 10)
> 1. *Lord of Chaos (The Wheel of Time, Book 6)* — por semelhança com *The Fires of Heaven* e *The Dragon Reborn*
> 2. *The Path of Daggers (The Wheel of Time, Book 8)* — ✅ acerto no teste
> 3. *Winter's Heart (The Wheel of Time, Book 9)* — ✅ acerto no teste

## 6. Limitações

- **Cold-start:** usuário sem histórico recebe apenas os populares, sem personalização. A personalização começa na primeira avaliação feita na interface.
- **Esparsidade e viés de popularidade:** com poucos dados por item, as similaridades são ruidosas. O shrinkage ameniza, mas não elimina.
- **Avaliação offline:** o teste só conhece o que o usuário avaliou. Recomendações boas mas nunca avaliadas contam como erro, então as métricas subestimam a qualidade real.
- **Base muito esparsa:** depois do filtro, 99,65 % da matriz é vazia, e 63 % das interações originais (nota 0) foram descartadas por não serem notas reais. Sobram 4.368 dos 271 mil livros e 5.431 usuários. A similaridade usa matrizes densas item × item, o que só cabe em memória para catálogos de alguns milhares de itens.
- **Edições duplicadas:** o mesmo livro pode ter vários ISBNs (edições diferentes) e aparecer mais de uma vez nas listas, porque o modelo os trata como itens distintos.
- **Sem data:** sem timestamp, o teste é um sorteio e não simula "prever o futuro".
- **Sem conteúdo:** o modelo ignora categoria, autor e sinopse. Um híbrido (CF + conteúdo) ajudaria em itens novos.
- **Sem validação de hiperparâmetros:** vizinhos e shrinkage seguem valores usuais, não foram otimizados.
- **Uso acadêmico:** não há autenticação, concorrência nem persistência robusta. As avaliações da interface vão para um CSV local.

## 7. Conclusão

A filtragem colaborativa superou os baselines em todas as métricas de ranking. O item-based teve o melhor desempenho geral e ainda oferece explicações às recomendações. A abordagem é simples, interpretável e funciona sobre qualquer base de interações. Como próximos passos, ficam a otimização dos hiperparâmetros, a fatoração de matrizes (ex.: SVD/ALS) e um modelo híbrido para reduzir o problema do cold-start.

## 8. Como executar

Requer Python 3.11+ (testado com 3.13).

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python scripts/prepare_data.py    # baixa o Book-Crossing (Kaggle) e prepara data/bookcrossing/
python scripts/evaluate.py        # (opcional) regenera métricas e exemplos em results/
streamlit run app.py              # abre a interface em http://localhost:8501
pytest                            # testes das métricas e das regras de recomendação
```

### Interface

| Aba | Função |
|---|---|
| 🏠 Visão geral | números da base, etapas do método e funil do tratamento dos dados |
| 📖 Histórico | livros avaliados pelo usuário (com capa, autor e categoria), nota média e distribuições |
| ✨ Recomendações | top-N sem livros já lidos, com justificativa e afinidade; comparação lado a lado com a popularidade |
| ⭐ Avaliar livros | busca por título ou autor e nota; grava em `data/user_ratings.csv` e atualiza as recomendações |
| 📊 Resultados | métricas, gráficos, leitura dos números e os exemplos para 5 usuários |

As capas vêm da Open Library (pelo ISBN) e exigem internet; sem elas, o cartão mostra um gradiente com a inicial do título.

### Roteiro sugerido para a demonstração

1. Apresentar o problema e a base na aba **Visão geral** (números, etapas do método, funil do tratamento).
2. Escolher um usuário, mostrar o **Histórico** e depois as **Recomendações** (destacar que nada do histórico se repete e a justificativa).
3. Ativar **Comparar com o baseline de popularidade** nas recomendações e trocar o modelo (item-based → user-based).
4. Selecionar **Novo usuário (sem histórico)**: aparecem os populares.
5. Na aba **Avaliar livros**, avaliar 3 ou 4 livros e ver as recomendações passarem a ser personalizadas.
6. Fechar na aba **Resultados** com as métricas.

## Estrutura

```
app.py                 interface Streamlit
config.yaml            base de dados, mapeamento de colunas e hiperparâmetros
scripts/               prepare_data.py, evaluate.py
src/recsys/            data.py, models.py, metrics.py, evaluation.py
tests/                 testes das métricas e dos recomendadores
results/               metrics.csv e exemplos_5_usuarios.md (gerados)
```
