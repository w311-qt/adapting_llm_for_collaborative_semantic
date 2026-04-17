---
marp: true
theme: default
paginate: true
math: mathjax
style: |
  section {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 20px;
    padding: 40px 50px;
  }
  h1 {
    color: #1a1a2e;
    border-bottom: 3px solid #e94560;
    padding-bottom: 8px;
    font-size: 32px;
  }
  h2 { color: #16213e; font-size: 26px; }
  table { width: 100%; border-collapse: collapse; font-size: 17px; }
  th { background: #16213e; color: white; padding: 6px 10px; text-align: left; }
  td { padding: 5px 10px; border-bottom: 1px solid #ddd; }
  code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-size: 15px; }
  pre { background: #f4f4f4; padding: 12px; border-radius: 6px; font-size: 14px; }
  blockquote {
    border-left: 4px solid #e94560;
    padding-left: 12px;
    color: #444;
    margin: 10px 0;
  }
---

# Adapting LLMs by Integrating Collaborative Semantics for Recommendation

*Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation*

LC-Rec / CoLLM

---

## Проблема: два несовместимых мира

Коллаборативная фильтрация (CF):
- предметы — числовые ID
- эмбеддинги учатся на матрице взаимодействий
- знает, кто что покупал вместе, но не что это такое

Большие языковые модели (LLM):
- работают с токенами из текстового словаря
- знают, что предметы означают, но не знают ничего о поведении пользователей

> Semantic gap: `ITEM_42` — бессмысленная строка для GPT/LLaMA.  
> Поведенческое сходство ≠ лингвистическое сходство.

---

## Почему предыдущие методы не справлялись

| Метод | Идея | Почему не работает |
|---|---|---|
| P5 / TALLRec | только текст → LLM | весь CF-сигнал теряется |
| IDRec | ID как токены, файн-тюн LLM целиком | catastrophic forgetting, 7B параметров |
| Soft Prompting | обучаемые непрерывные промпты | нет внешнего сигнала, случайная инициализация |
| Two-Tower | CF + LLM раздельно | LLM генерирует без CF-сигнала |

Ни один подход не использует готовые CF-эмбеддинги как входной сигнал для LLM.

---

## Идея

Перевести CF-эмбеддинги в пространство LLM через обучаемый проекционный слой:

```
CF embedding  ──►  MLP Projection  ──►  soft token  ──►  LLM
  (d_cf=64)          ~1M параметров      (d_lm=128)    (заморожен/LoRA)
```

CF-эмбеддинг — готовый поведенческий prior из LightGCN.  
MLP переводит его в пространство, понятное трансформеру, не разрушая языковые представления.

Обучается только проекционный слой. CF-эмбеддинги и большая часть LLM заморожены.

---

## Архитектура: общая схема

```
Данные взаимодействий          Текстовые метаданные
        │                               │
        ▼                               ▼
   LightGCN / BPR               LLM Tokenizer
        │                               │
   E_cf ∈ ℝ^{|I|×64}                   │
        │                               │
        ▼                               │
   MLP Projection  ─────────────────────┤
    64 → 256 → 128                      │
        │                               ▼
        └──────────────►  Входная последовательность LLM
                          [soft_i1][текст_i1][soft_i2][текст_i2]...
                                         │
                                   Transformer
                                  (causal mask)
                                         │
                                  Prediction Head
                                  → логиты по каталогу
```

---

## Архитектура: Projection MLP и тензорный поток

$$\mathbf{p}_i = \mathrm{LayerNorm}\!\left(\mathbf{W}_2 \cdot \mathrm{GELU}(\mathbf{W}_1 \mathbf{e}_i^{\mathrm{cf}} + \mathbf{b}_1) + \mathbf{b}_2\right)$$

| Шаг | Форма | Что происходит |
|---|---|---|
| `item_ids` | `(B, T)` | история пользователя |
| CF lookup | `(B, T, 64)` | поведенческий prior, заморожен |
| MLP проекция | `(B, T, 128)` | soft-токены для трансформера |
| + positional | `(B, T, 128)` | позиционные эмбеддинги |
| Transformer | `(B, T, 128)` | causal self-attention |
| последний токен | `(B, 128)` | представление всей истории |
| head | `(B, 500)` | логиты по каталогу |

---

## Архитектура: инжекция soft-токенов

Входная последовательность — смесь soft и текстовых токенов:

```
[SYS] <p_i1> "Inception" , <p_i2> "Interstellar" , <p_i3> ? [/INST]
        ↑                    ↑
    soft-токен           soft-токен
  (вектор из MLP)      (вектор из MLP)
```

Causal attention mask — позиция $t$ видит только $0 \ldots t$:

$$A_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}$$

Soft-токены участвуют в self-attention наравне с текстовыми.  
Трансформер сам учится совмещать поведенческий и языковой сигнал.

---

## Функции потерь

Рекомендательная потеря — кросс-энтропия:

$$\mathcal{L}_{\mathrm{rec}} = -\frac{1}{N}\sum_{(\mathcal{S}_u,\, i^+) \in \mathcal{D}} \log \hat{p}(i^+ \mid \mathcal{S}_u)$$

Потеря выравнивания — MLP не должен ломать геометрию CF-пространства:

$$\mathcal{L}_{\mathrm{align}} = \frac{1}{N^2}\sum_{i,j} \Bigl(\cos(\mathbf{p}_i, \mathbf{p}_j) - \cos(\mathbf{e}_i^{\mathrm{cf}}, \mathbf{e}_j^{\mathrm{cf}})\Bigr)^2$$

$$\mathcal{L} = \mathcal{L}_{\mathrm{rec}} + \alpha \cdot \mathcal{L}_{\mathrm{align}} + \beta \cdot \mathcal{L}_{\mathrm{lm}}, \quad \alpha \approx 0.5,\ \beta \approx 0.05$$

---

## Датасеты, метрики, схема валидации

Датасеты: MovieLens-1M, Amazon Beauty/Sports, Yelp. Leave-one-out split.

Метрики:

$$\mathrm{HR@K} = \frac{1}{|\mathcal{U}|}\sum_u \mathbf{1}[i_u^* \in \mathrm{Top\text{-}K}]$$

$$\mathrm{NDCG@K} = \frac{1}{|\mathcal{U}|}\sum_u \frac{\mathbf{1}[i_u^* \in \mathrm{Top\text{-}K}]}{\log_2(\mathrm{rank}(i_u^*) + 1)}$$

HR@K — попал ли правильный предмет в топ (бинарно).  
NDCG@K — штраф за то, что он оказался ниже в списке.  
Нужны обе: высокий HR при низком NDCG означает, что модель угадывает, но ставит ответ в конец.

---

## Результаты

| Модель | ML-1M HR@10 | ML-1M NDCG@10 | Amazon HR@10 | Amazon NDCG@10 |
|---|---|---|---|---|
| SASRec | 0.812 | 0.598 | 0.421 | 0.287 |
| BERT4Rec | 0.798 | 0.581 | 0.408 | 0.271 |
| P5 | 0.731 | 0.541 | 0.463 | 0.312 |
| TALLRec | 0.769 | 0.562 | 0.471 | 0.321 |
| **LC-Rec** | **0.847** | **0.631** | **0.503** | **0.348** |

На плотных данных (ML-1M) прирост ~4% к лучшему бейзлайну.  
На разреженных (Amazon) +6–8% — именно там, где CF страдает от cold-start.

---

## Ограничения

| Проблема | Цифры |
|---|---|
| Latency | 200–500 мс / запрос, SLA на проде < 50 мс |
| Размер output head | 350M SKU × 768 ≈ 1 ТБ |
| Стоимость обучения | $500–2000 на GPU, SASRec — часы на CPU |
| Cold-start новых предметов | нет CF-эмбеддинга → нет fallback |

Про валидность экспериментов:
- бейзлайны не оптимизированы, SASRec с тюнингом часто догоняет LLM-методы
- sampled evaluation (100 негативов) завышает метрики
- CF обучен на полном датасете — риск data leakage в тестовую часть

---

## Применимость в индустрии

Где работает:
- каталог до 100K предметов, нет жёсткого SLA
- batch-рекомендации (email, push) — latency не критична
- conversational recommendation, кросс-доменные задачи
- частый cold-start + богатые текстовые описания

Где не работает:
- homepage feed (Netflix, YouTube) — нужно < 50 мс P99
- каталоги в миллионы позиций
- mobile/edge-деплой — 7B не влезают

Путь к production: дистилляция + ANN retrieval (FAISS) + quantization.

---

## Заключение

Статья предлагает архитектурно аккуратное решение: MLP-проекция как мост между CF и LLM. Это работает и даёт прирост там, где поведенческий сигнал разреженный.

Использовать в реальном highload сегодня не получится — latency и масштаб каталога это исключают. Но метод задаёт правильный принцип, на котором будут строиться более эффективные подходы.

Наиболее перспективные направления: дистилляция знаний из LLM+CF в компактную CF-модель, замена полного softmax на ANN-поиск по проецированным эмбеддингам.
