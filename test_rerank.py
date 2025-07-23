from langchain.schema import Document
from langchain_community.document_transformers import LongContextReorder
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import numpy as np
from typing import List, Tuple
import time
import requests

class MonoT5Reranker:
    def __init__(self, model_name: str = "castorini/monot5-large-msmarco", max_retries: int = 3):
        """
        Инициализация MonoT5 реранкера
        """
        print(f"Загрузка модели {model_name}...")
        
        # Retry логика для загрузки токенизатора
        for attempt in range(max_retries):
            try:
                self.tokenizer = T5Tokenizer.from_pretrained(
                    model_name,
                    local_files_only=False
                )
                break
            except (requests.exceptions.ConnectionError, 
                    requests.exceptions.Timeout,
                    ConnectionError,
                    Exception) as e:
                print(f"Попытка {attempt + 1}/{max_retries} загрузки токенизатора не удалась: {e}")
                if attempt == max_retries - 1:
                    raise Exception(f"Не удалось загрузить токенизатор после {max_retries} попыток")
                time.sleep(2 ** attempt)  # Экспоненциальная задержка
        
        # Retry логика для загрузки модели
        for attempt in range(max_retries):
            try:
                self.model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
                break
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    ConnectionError,
                    Exception) as e:
                print(f"Попытка {attempt + 1}/{max_retries} загрузки модели не удалась: {e}")
                if attempt == max_retries - 1:
                    raise Exception(f"Не удалось загрузить модель после {max_retries} попыток")
                time.sleep(2 ** attempt)  # Экспоненциальная задержка
            
        self.model.eval()
        # Определяем устройство модели
        self.device = next(self.model.parameters()).device
        print(f"Модель загружена на устройство: {self.device}")
        
        # Получаем ID токенов для 'true' и 'false'
        self.true_token_id = self.tokenizer.encode('true', add_special_tokens=False)[0]
        self.false_token_id = self.tokenizer.encode('false', add_special_tokens=False)[0]
        print(f"True token ID: {self.true_token_id}, False token ID: {self.false_token_id}")
    
    def rerank_documents(self, query: str, documents: List[Document], top_k: int = None) -> List[Tuple[Document, float]]:
        """
        Реранкинг документов для заданного запроса
        
        Args:
            query: Поисковый запрос
            documents: Список документов LangChain
            top_k: Количество топ документов для возврата
            
        Returns:
            Список кортежей (документ, скор) отсортированных по релевантности
        """
        if not documents:
            return []
        
        scores = []
        
        # Обрабатываем документы по одному для стабильности
        for doc in documents:
            # MonoT5 формат: "Query: {query} Document: {document} Relevant:"
            prompt = f"Query: {query} Document: {doc.page_content} Relevant:"
        
        # Токенизация
        inputs = self.tokenizer(
                prompt,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
            # Перенос на устройство модели
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
        with torch.no_grad():
                # Для T5 нужно использовать generation
                # Начинаем генерацию с pad токена
                decoder_input_ids = torch.tensor([[self.tokenizer.pad_token_id]], device=self.device)
                
                # Получаем логиты для первого сгенерированного токена
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    decoder_input_ids=decoder_input_ids
                )
                
                # Получаем логиты для следующего токена после pad токена
                logits = outputs.logits[0, 0, :]  # [vocab_size]
                
                # Получаем логиты для 'true' и 'false'
                true_logit = logits[self.true_token_id].item()
                false_logit = logits[self.false_token_id].item()
                
                # Вычисляем вероятность через softmax
                true_prob = torch.softmax(torch.tensor([false_logit, true_logit]), dim=0)[1].item()
                scores.append(true_prob)
        
        # Создание результатов с скорами
        scored_docs = list(zip(documents, scores))
        
        # Сортировка по убыванию скора
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Возврат топ-k результатов
        if top_k:
            scored_docs = scored_docs[:top_k]
        
        return scored_docs

# Пример использования с LangChain
def example_usage():
    # Создание тестовых документов
    documents = [
        Document(
            page_content="Python - это высокоуровневый язык программирования общего назначения.",
            metadata={"source": "doc1.txt"}
        ),
        Document(
            page_content="Машинное обучение использует алгоритмы для анализа данных.",
            metadata={"source": "doc2.txt"}
        ),
        Document(
            page_content="LangChain - это фреймворк для разработки приложений с использованием языковых моделей.",
            metadata={"source": "doc3.txt"}
        ),
        Document(
            page_content="Реранкинг улучшает качество поиска путем переупорядочивания результатов.",
            metadata={"source": "doc4.txt"}
        ),
        Document(
            page_content="Искусственный интеллект революционизирует многие отрасли промышленности.",
            metadata={"source": "doc5.txt"}
        )
    ]
    
    # Инициализация реранкера
    reranker = MonoT5Reranker()
    
    # Запрос пользователя
    query = "Что такое LangChain и как он используется?"
    
    print(f"Запрос: {query}\n")
    print("Исходный порядок документов:")
    for i, doc in enumerate(documents, 1):
        print(f"{i}. {doc.page_content[:60]}...")
    
    # Реранкинг
    reranked_docs = reranker.rerank_documents(query, documents, top_k=5)
    
    print(f"\nРезультаты реранкинга (топ-3):")
    for i, (doc, score) in enumerate(reranked_docs, 1):
        print(f"{i}. [Скор: {score:.4f}] {doc.page_content}")
        print(f"   Источник: {doc.metadata['source']}\n")

# Интеграция с LangChain RAG pipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM

class RAGWithReranking:
    def __init__(self, documents: List[Document], max_retries: int = 3):
        # Инициализация компонентов с retry логикой
        print("Загрузка embedding модели...")
        
        for attempt in range(max_retries):
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                break
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    ConnectionError,
                    Exception) as e:
                print(f"Попытка {attempt + 1}/{max_retries} загрузки embedding модели не удалась: {e}")
                if attempt == max_retries - 1:
                    raise Exception(f"Не удалось загрузить embedding модель после {max_retries} попыток")
                time.sleep(2 ** attempt)
        
        print("Создание векторного хранилища...")
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        print("Загрузка reranker модели...")
        self.reranker = MonoT5Reranker()
    
    def retrieve_and_rerank(self, query: str, k: int = 10, rerank_top_k: int = 5) -> List[Document]:
        """
        Поиск документов с последующим реранкингом
        """
        # Первичный поиск по эмбеддингам
        initial_docs = self.vectorstore.similarity_search(query, k=k)
        
        print(f"Найдено {len(initial_docs)} документов при первичном поиске")
        
        # Реранкинг найденных документов
        reranked_results = self.reranker.rerank_documents(query, initial_docs, top_k=rerank_top_k)
        
        # Возвращаем только документы без скоров
        return reranked_results
        #return [doc for doc, score in reranked_results]

# Пример полного RAG pipeline с реранкингом
def full_rag_example():
    # Создание большого набора документов
    large_doc_set = [
        Document(page_content="Python - объектно-ориентированный язык программирования.", metadata={"id": 1}),
        Document(page_content="LangChain помогает создавать приложения с LLM.", metadata={"id": 2}),
        Document(page_content="Векторные базы данных хранят эмбеддинги документов.", metadata={"id": 3}),
        Document(page_content="Трансформеры - это архитектура нейронных сетей.", metadata={"id": 4}),
        Document(page_content="Реранкинг повышает точность поиска информации.", metadata={"id": 5}),
        Document(page_content="FAISS - библиотека для эффективного поиска по векторам.", metadata={"id": 6}),
        Document(page_content="Hugging Face предоставляет модели машинного обучения.", metadata={"id": 7}),
        Document(page_content="RAG объединяет поиск и генерацию текста.", metadata={"id": 8}),
    ]
    
    # Создание RAG системы с реранкингом
    rag_system = RAGWithReranking(large_doc_set)
    
    # Тестовый запрос
    query = "Как использовать LangChain для RAG?"
    
    print(f"Запрос: {query}")
    print("="*50)
    
    # Получение результатов с реранкингом
    final_docs = rag_system.retrieve_and_rerank(query, k=6, rerank_top_k=5)
    
    print(f"\nФинальные результаты после реранкинга:")
    for i, doc in enumerate(final_docs, 1):
        print(f"{i}. {doc[0].page_content}")
        print(f"   ID: {doc[0].metadata['id']}")
        print(f"   Скор: {doc[1]}\n")

if __name__ == "__main__":
    print("Пример 1: Базовый реранкинг")
    print("="*50)
    example_usage()
    
    print("\n" + "="*80 + "\n")
    
    print("Пример 2: RAG с реранкингом")
    print("="*50)
    full_rag_example()