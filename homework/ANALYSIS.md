# Аналіз: Ollama vs OpenAI на задачі extraction

## Налаштування

- **Self-hosted:** Ollama 0.x, модель `llama2:latest` (7B, Q4_0), localhost:11434, `format=json`
- **Cloud:** OpenAI `gpt-4o-mini`, `response_format=json_object`
- **Датасети:** 3 транскрипти зустрічей українською (`simple`, `chaotic`, `technical`)
- **Метрики:** JSON-валідність, tasks_found / expected, hallucinations (owner ∉ списку реальних учасників), input/output/total tokens, cost (USD), latency (sec)

## Результати

| Dataset   | Provider | JSON | Tasks | Halluc | Tokens | Cost      | Latency |
|-----------|----------|------|-------|--------|--------|-----------|---------|
| simple    | Ollama   | ✅   | 4/4   | +4*    | 814    | $0        | 11.8s   |
| simple    | OpenAI   | ✅   | 4/4   | 0      | 745    | $0.000215 | 6.7s    |
| chaotic   | Ollama   | ✅   | 3/6   | +3     | 1159   | $0        | 11.7s   |
| chaotic   | OpenAI   | ✅   | 6/6   | 0      | 1144   | $0.000309 | 4.7s    |
| technical | Ollama   | ✅   | 4/5   | +1     | 1459   | $0        | 15.0s   |
| technical | OpenAI   | ✅   | 5/5   | 0      | 1404   | $0.000374 | 7.9s    |

\* `simple/ollama`: llama2 переклала імена та summary на англійську (Anna, Ivan, Martin, Olena замість кирилиці), порушивши вимогу мови. Завдання знайдено правильно, але heuristic-перевірка по списку справжніх учасників (кирилиця) флагнула 4 owner-и як «галюцинації». Це інша категорія дефекту — failure to follow language constraint, але теж критична для прод-юзкейсу.

**Сумарно:**
- OpenAI: **15/15** завдань, 0 галюцинацій, 100% JSON-валідність на всіх датасетах
- Ollama: **11/15** завдань, лінгвістична нестабільність, провал на хаотичному тексті (3/6)
- Latency: OpenAI ~5–8s, Ollama ~12–15s (на M1/M2 без GPU)
- Cost OpenAI за 3 датасети: $0.0009 — тривіально

## 1. Коли використовувати Ollama (self-hosted)?

**Підходить, коли:**
- **Privacy / compliance** — дані не можуть залишити периметр (медицина, банкінг, держсектор, юридичні документи)
- **Високий обсяг** — мільйони викликів/день, де навіть $0.0003/запит складаються в тисячі $
- **Простий well-defined формат** — короткі тексти, англійська, мало edge cases
- **Toleranable якість 80–90%** — є людська перевірка downstream або це fallback
- **Стабільний інтернет не гарантований** (offline-сценарії)

**Критерії вибору:**
- Чи є GPU (мінімум 8GB VRAM для 7B-моделей у Q4)? Якщо CPU-only — latency 10–15s/запит, не для real-time.
- Чи задача витримує 70–85% точність? Для нашого extraction — ні, втрачаємо завдання на хаотичних текстах.
- Чи є людина-валідатор у ланцюгу? Без неї помилки доїдуть до прода.

**Для нашої задачі (extraction з українських транскриптів):** llama2 7B не підходить — губить мову, пропускає завдання на розмовному тексті. Можна спробувати `mistral`, `llama3.1:8b`, `qwen2.5:7b` — вони сучасніші і зазвичай тримають мультиязичність краще.

## 2. Коли використовувати OpenAI (cloud)?

**Підходить, коли:**
- Якість критична, помилка коштує дорого (юзер бачить результат напряму)
- Низький-середній обсяг — costs не домінують над dev-часом
- Treба structured output (`response_format=json_object` працює надійно)
- Багатомовність / морфологічно складні мови
- Швидкий time-to-market — не треба інфраструктуру піднімати

**Уроки з прогону:**
- `response_format={"type":"json_object"}` дав 100% JSON-валідність, нуль ручного парсингу
- На хаотичному тексті (`chaotic` — 6 завдань vs 3 у Ollama) перевага особливо помітна
- Рахунок реалістичний: $0.0009 за 3 повних запити — дешевше за 1 хвилину розробника
- `usage.prompt_tokens` / `completion_tokens` — точний облік, не треба heuristic'и

**Коли не варто:** масові batch-обробки логів, embedding-pipeline на мільйони документів — там self-hosted виграє.

## 3. Гібридний підхід

Так, ensemble має сенс. Декілька шаблонів:

**a) Cascade (cheap-first):**
```
result = call_ollama(text)
if not is_valid(result) or confidence_low(result):
    result = call_openai(text)
```
Окупається, якщо ≥60% запитів проходять Ollama чисто. У нашому прогоні — лише `simple/ollama` пройшов би (та й то з мовним дефектом), тож тут cascade не виграє. Виграє на простіших задачах (класифікація, sentiment).

**b) Parallel + voting:** запустити обидві моделі, взяти OpenAI як ground-truth, Ollama як «sanity check» для виявлення промпт-ін'єкцій або data-drift. Дорого, але корисно для security-sensitive flows.

**c) Router by complexity:** короткий чистий текст → Ollama, довгий/хаотичний/мультимовний → OpenAI. Класифікатор-роутер можна зробити тим же Ollama (він з length+language detection справляється).

**d) OpenAI для extraction → Ollama для трансформацій:** дорогий мозок витягує сутності, дешева модель робить форматування / переклад / summary.

Для нашого ДЗ найрозумніше: **Ollama як fallback на випадок outage OpenAI**, з логуванням для подальшого аналізу. Реверсний cascade (OpenAI primary, Ollama backup) виправдан, коли uptime критичний.

## 4. Розширення

**Мультимовність (uk → en → uk):**
- Простий шлях: окремий translation-pass перед extraction (`uk → en`), потім extraction на англійській (моделі сильніші), потім переклад відповідей назад. Ризик: втрата нюансів при подвійному перекладі, плюс латенсі ×3.
- Кращий шлях: одна модель з добрим мультилінгвістичним покриттям (gpt-4o-mini OK, для local — `qwen2.5`, `aya-23`). У промпті явно: «Output language must match input language. Do not translate names.»
- Для Ollama llama2 — обовʼязково `system`-prompt + few-shot з кириличними прикладами, інакше повертатиметься на англійську (як видно з `simple/ollama`).

**Confidence score per task:**
- **Self-reported:** просити модель додати поле `confidence: 0..1` у кожне завдання. Швидко, але калібрування поганe — моделі схильні до overconfidence.
- **Logprobs (OpenAI):** `logprobs=True` + усереднення по токенах поля `task` / `deadline`. Ближче до реального confidence, але дорожче.
- **Ensemble agreement:** запустити N разів з temperature=0.3, рахувати % збігів. Дорого, але надійно.
- **External validator:** окрема модель-критик, яка читає `(text, extracted_task)` → score. Найкраща якість, +1 round-trip.

Найпрактичніше для прода: self-reported confidence + список heuristic-фільтрів (owner ∈ списку учасників? deadline парситься як дата? task non-empty?), які знижують confidence при провалі будь-якого з них.
