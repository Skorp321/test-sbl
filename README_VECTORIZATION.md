# Векторизация документов .docx для ChromaDB

Набор скриптов для подготовки, векторизации и поиска документов в формате .docx с использованием ChromaDB и LangChain.

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
# Активируйте виртуальную среду
source venv/bin/activate

# Установите зависимости
pip install -r requirements_vector.txt
```

### 2. Обработка документа

```bash
# Векторизация client.docx
python docx_to_chromadb.py
```

### 3. Поиск в базе данных

```bash
# Интерактивный поиск
python search_demo.py
```

## 📁 Структура проекта

```
project/
├── docx_to_chromadb.py          # Основной скрипт векторизации
├── search_demo.py               # Демо поиска
├── improved_text_splitter.py    # Улучшенный text splitter
├── requirements_vector.txt      # Зависимости
├── client.docx                  # Исходный документ
├── chromadb_documents/          # База данных ChromaDB
│   ├── contract_documents_structured/
│   └── contract_documents_recursive/
└── README_VECTORIZATION.md      # Этот файл
```

## 🔧 Основные компоненты

### 1. DocxToChromaDB
Главный класс для обработки .docx документов:
- Читает .docx файлы
- Создает два типа фрагментов (структурные и рекурсивные)
- Векторизует фрагменты с помощью sentence-transformers
- Сохраняет в ChromaDB

### 2. ImprovedTextSplitter
Улучшенный text splitter для юридических документов:
- Распознает структуру договоров
- Выделяет разделы, пункты, подпункты
- Сохраняет контекстную информацию

### 3. DocumentSearch
Класс для поиска в векторной базе данных:
- Поиск по семантическому сходству
- Поддержка двух типов коллекций
- Ранжирование результатов

## 📊 Типы фрагментов

### Структурные фрагменты
- Основаны на структуре документа
- Сохраняют иерархию разделов
- Включают контекстные заголовки
- Коллекция: `contract_documents_structured`

**Пример метаданных:**
```json
{
  "source": "client.docx",
  "type": "subsection",
  "header": "Пункт: 8. Утрата или повреждение | Подпункт: 8.1. Лизингополучатель...",
  "chunk_size": 1247
}
```

### Рекурсивные фрагменты
- Равномерная разбивка текста
- Фиксированный размер (800 символов)
- Перекрытие (100 символов)
- Коллекция: `contract_documents_recursive`

**Пример метаданных:**
```json
{
  "source": "client.docx",
  "type": "recursive_split",
  "splitter_type": "recursive",
  "chunk_size": 785
}
```

## 🔍 Возможности поиска

### Семантический поиск
- Поиск по смыслу, а не по точному совпадению
- Ранжирование по релевантности
- Поддержка сложных запросов

### Примеры запросов
```
"лизинговые платежи"           # Финансовые условия
"страхование объекта"          # Страховые обязательства
"права лизингодателя"          # Права сторон
"форс-мажор"                   # Форс-мажорные обстоятельства
```

## 🛠️ Настройка

### Модели эмбеддингов
По умолчанию используется `sentence-transformers/all-MiniLM-L6-v2`.

Для изменения модели:
```python
processor = DocxToChromaDB(
    embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
```

### Настройка фрагментации
```python
result = processor.process_docx_file(
    "document.docx",
    use_both_splitters=True,
    recursive_chunk_size=1000,  # Размер фрагмента
    recursive_chunk_overlap=200  # Перекрытие
)
```

## 📈 Результаты обработки client.docx

```
Извлечено текста: 46,164 символов
Структурных фрагментов: 159
Рекурсивных фрагментов: 58
Найдено разделов: 23
Найдено подпунктов: 100
```

## 🔍 Примеры использования

### Базовый поиск
```python
from docx_to_chromadb import DocxToChromaDB

processor = DocxToChromaDB()
results = processor.search_similar("лизинговые платежи", k=5)

for result in results:
    print(f"Релевантность: {result['similarity_score']:.4f}")
    print(f"Содержимое: {result['content'][:200]}...")
```

### Программный поиск
```python
from search_demo import DocumentSearch

searcher = DocumentSearch()
results = searcher.search_both("страхование объекта", k=3)

# Структурные результаты
for result in results["structured"]:
    print(result["metadata"]["header"])

# Рекурсивные результаты
for result in results["recursive"]:
    print(result["content"])
```

## 🚫 Устранение неполадок

### Проблема: Ошибка импорта sentence-transformers
```bash
pip install sentence-transformers torch
```

### Проблема: Ошибка ChromaDB
```bash
pip install chromadb --upgrade
```

### Проблема: Нет модели эмбеддингов
```bash
# Модель загрузится автоматически при первом запуске
# Для кэширования:
export TRANSFORMERS_CACHE=/path/to/cache
```

## 📋 Требования

- Python 3.8+
- 8GB RAM (для sentence-transformers)
- 2GB свободного места (для моделей)

## 🔧 Опциональные настройки

### Использование OpenAI embeddings
```bash
export OPENAI_API_KEY=your_api_key
```

### Кастомная модель
```python
processor = DocxToChromaDB(
    embedding_model="cointegrated/rubert-tiny2"  # Русскоязычная модель
)
```

## 📊 Сравнение подходов

| Метод | Преимущества | Недостатки |
|-------|-------------|------------|
| Структурный | Сохраняет контекст<br>Точная навигация | Зависит от структуры |
| Рекурсивный | Равномерное покрытие<br>Универсальность | Может разрывать смысл |

## 🎯 Рекомендации

1. **Для навигации по документу** - используйте структурные фрагменты
2. **Для общего поиска** - используйте рекурсивные фрагменты
3. **Для лучших результатов** - комбинируйте оба подхода

## 🔄 Обновление данных

Для обновления векторной базы данных:
```bash
# Удалите старую базу данных
rm -rf chromadb_documents/

# Запустите обработку заново
python docx_to_chromadb.py
```

---

*Система готова к обработке других .docx документов. Замените `client.docx` на нужный файл.* 