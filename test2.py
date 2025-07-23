import os
import argparse
from pathlib import Path
from typing import List, Optional, Tuple
import time
import requests

from improved_text_splitter import ImprovedTextSplitter
from text_splitter import TextSplitter

# LangChain импорты
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Для работы с Excel
import pandas as pd

# Для реранкинга
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

class MonoT5Reranker:
    def __init__(self, model_name: str = "castorini/monot5-large-msmarco", max_retries: int = 3):
        """
        Инициализация MonoT5 реранкера
        
        Args:
            model_name: Название модели реранкера
            max_retries: Количество попыток загрузки модели
        """
        self.model_name = model_name
        print(f"Загрузка модели реранкера {model_name}...")
        
        # Retry логика для загрузки токенизатора
        for attempt in range(max_retries):
            try:
                self.tokenizer = T5Tokenizer.from_pretrained(
                    model_name,
                    legacy=False
                )
                break
            except (requests.exceptions.ConnectionError, 
                    requests.exceptions.Timeout,
                    ConnectionError,
                    Exception) as e:
                print(f"Попытка {attempt + 1}/{max_retries} загрузки токенизатора не удалась: {e}")
                if attempt == max_retries - 1:
                    raise Exception(f"Не удалось загрузить токенизатор после {max_retries} попыток")
                time.sleep(2 ** attempt)
        
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
                time.sleep(2 ** attempt)
                
        self.model.eval()
        self.device = next(self.model.parameters()).device
        print(f"MonoT5 реранкер загружен на устройство: {self.device}")
        
        # Получаем ID токенов для 'true' и 'false'
        self.true_token_id = self.tokenizer.encode('true', add_special_tokens=False)[0]
        self.false_token_id = self.tokenizer.encode('false', add_special_tokens=False)[0]
        print(f"True token ID: {self.true_token_id}, False token ID: {self.false_token_id}")
        
    def rerank(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> List[Tuple[Document, float]]:
        """
        Реранкинг документов по релевантности к запросу
        
        Args:
            query: Поисковый запрос
            documents: Список документов для реранкинга
            top_k: Количество топ документов для возврата (None - все)
            
        Returns:
            Отсортированный по релевантности список кортежей (документ, скор)
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
        
        # Сортируем документы по скорам
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Возвращаем топ-k результатов
        if top_k is not None:
            scored_docs = scored_docs[:top_k]
        
        return scored_docs

class DocumentProcessor:
    def __init__(self, embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 reranker_model: str = "castorini/monot5-large-msmarco", 
                 use_reranker: bool = True):
        """
        Инициализация процессора документов
        
        Args:
            embeddings_model: Название модели для создания эмбеддингов
            reranker_model: Название модели реранкера
            use_reranker: Использовать ли реранкер
        """
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embeddings_model
        )
        """self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )"""
        self.text_splitter = ImprovedTextSplitter()
        self.faiss: Optional[FAISS] = None
        self.vectorstore: Optional[EnsembleRetriever] = None
        
        # Инициализация реранкера
        self.use_reranker = use_reranker
        if self.use_reranker:
            try:
                print("Загрузка модели реранкера...")
                self.reranker = MonoT5Reranker(reranker_model)
                print("✓ Реранкер загружен")
            except Exception as e:
                print(f"⚠️  Ошибка при загрузке реранкера: {e}")
                print("Продолжаем работу без реранкера")
                self.use_reranker = False
                self.reranker = None
        else:
            self.reranker = None
    
    def load_docx_files(self, file_paths: List[str]) -> List[Document]:
        """Загружает DOCX файлы"""
        documents = []
        for file_path in file_paths:
            if file_path.lower().endswith('.docx'):
                try:
                    loader = Docx2txtLoader(file_path)
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata['source'] = file_path
                    documents.extend(docs)
                    print(f"✓ Загружен файл: {file_path}")
                except Exception as e:
                    print(f"✗ Ошибка при загрузке {file_path}: {e}")
        return documents
    
    def load_xlsx_files(self, file_paths: List[str]) -> List[Document]:
        """Загружает XLSX файлы"""
        documents = []
        for file_path in file_paths:
            if file_path.lower().endswith('.xlsx'):
                try:
                    # Читаем все листы Excel файла
                    excel_file = pd.ExcelFile(file_path)
                    for sheet_name in excel_file.sheet_names:
                        df = pd.read_excel(file_path, sheet_name=sheet_name)
                        
                        # Преобразуем DataFrame в текст
                        content = f"Лист: {sheet_name}\n\n"
                        content += df.to_string(index=False)
                        
                        doc = Document(
                            page_content=content,
                            metadata={
                                'source': file_path,
                                'sheet': sheet_name
                            }
                        )
                        documents.append(doc)
                    
                    print(f"✓ Загружен файл: {file_path} ({len(excel_file.sheet_names)} листов)")
                except Exception as e:
                    print(f"✗ Ошибка при загрузке {file_path}: {e}")
        return documents
    
    def load_documents(self, directory: str) -> List[Document]:
        """
        Загружает все поддерживаемые документы из директории
        
        Args:
            directory: Путь к директории с документами
            
        Returns:
            Список загруженных документов
        """
        supported_extensions = ['.docx', '.xlsx']
        file_paths = []
        
        # Находим все поддерживаемые файлы
        for ext in supported_extensions:
            file_paths.extend(Path(directory).glob(f"**/*{ext}"))
            file_paths.extend(Path(directory).glob(f"**/*{ext.upper()}"))
        
        file_paths = [str(f) for f in file_paths]
        
        if not file_paths:
            print(f"Не найдено файлов в директории {directory}")
            return []
        
        print(f"Найдено файлов: {len(file_paths)}")
        
        # Загружаем документы
        documents = []
        documents.extend(self.load_docx_files(file_paths))
        documents.extend(self.load_xlsx_files(file_paths))
        
        return documents
    
    def create_vectorstore(self, documents: List[Document]) -> None:
        """
        Создает векторное хранилище из документов
        
        Args:
            documents: Список документов для векторизации
        """
        if not documents:
            print("Нет документов для векторизации")
            return
        
        print(f"Разбиение {len(documents)} документов на чанки...")
        
        # Разбиваем документы на чанки
        #chunks = self.text_splitter.split_documents(documents)
        chunks = []
        for document in documents:
            one_chunks = self.text_splitter.split_text_with_context(document.page_content)
            for chunk in one_chunks:
                # Только содержательные части превращаем в Document
                if chunk['type'] == 'content':
                    # Можно добавить метаданные, например, источник и заголовок
                    metadata = dict(document.metadata) if hasattr(document, "metadata") else {}
                    if 'header' in chunk:
                        metadata['header'] = chunk['header']
                    chunks.append(Document(page_content=chunk['content'], metadata=metadata))
        print(f"Создано {len(chunks)} чанков")
        
        # Создаем векторное хранилище
        print("Создание эмбеддингов и векторного хранилища...")
        
        # Если используется реранкер, настраиваем ретриверы на получение большего количества документов
        retriever_k = 20 if self.use_reranker else 4
        
        bm25 = BM25Retriever.from_documents(chunks)
        bm25.k = retriever_k
        self.faiss = FAISS.from_documents(chunks, self.embeddings)
        print("✓ Векторное хранилище создано")
        retriever = self.faiss.as_retriever(
                search_type="similarity",
                k=retriever_k,
                score_threshold=None,
            )

        self.vectorstore = EnsembleRetriever(
            retrievers=[bm25, retriever],
            weights=[0.5, 0.5],
        )
        
        if self.use_reranker:
            print(f"Ретриверы настроены на получение {retriever_k} документов для реранкинга")
            
    
    def save_vectorstore(self, path: str) -> None:
        """Сохраняет векторное хранилище"""
        if self.faiss is None:
            print("Нет векторного хранилища для сохранения")
            return
        
        self.faiss.save_local(path)
        print(f"✓ Векторное хранилище сохранено: {path}")
    
    def load_vectorstore(self, path: str) -> bool:
        """
        Загружает векторное хранилище
        
        Args:
            path: Путь к сохраненному векторному хранилищу
            
        Returns:
            True если загрузка успешна, False иначе
        """
        try:
            self.faiss = FAISS.load_local(
                path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            
            # Настраиваем ensemble retriever
            retriever_k = 20 if self.use_reranker else 4
            
            # Создаем BM25 retriever (нужно будет пересоздать из документов)
            # Для упрощения пока используем только FAISS
            retriever = self.faiss.as_retriever(
                search_type="similarity",
                k=retriever_k,
                score_threshold=None,
            )
            
            # Временно используем только FAISS retriever
            self.vectorstore = retriever
            
            print(f"✓ Векторное хранилище загружено: {path}")
            if self.use_reranker:
                print(f"Ретривер настроен на получение {retriever_k} документов для реранкинга")
            return True
        except Exception as e:
            print(f"✗ Ошибка при загрузке векторного хранилища: {e}")
            return False
    
    def search(self, query: str, k: int = 5, initial_k: Optional[int] = None) -> List[Tuple[Document, float]]:
        """
        Поиск по векторному хранилищу с реранкингом
        
        Args:
            query: Поисковый запрос
            k: Количество итоговых результатов
            initial_k: Количество документов для первичного поиска (больше k для реранкинга)
            
        Returns:
            Список кортежей (документ, скор)
        """
        if self.vectorstore is None:
            print("Векторное хранилище не загружено")
            return []
        
        # Если используется реранкер, получаем больше документов для последующего реранкинга
        if self.use_reranker and self.reranker is not None:
            search_k = initial_k or min(k * 3, 20)  # Берем в 3 раза больше документов или максимум 20
            results = self.vectorstore.invoke(query)
            
            print(f"Реранкинг {len(results)} документов...")
            # Применяем реранкинг и возвращаем топ-k результатов
            reranked_results = self.reranker.rerank(query, results, top_k=k)
            print(f"✓ Возвращено {len(reranked_results)} реранжированных результатов")
            return reranked_results
        else:
            # Без реранкинга - возвращаем результаты поиска без скоров (устанавливаем скор 0.0)
            results = self.vectorstore.invoke(query)
            return [(doc, 0.0) for doc in results[:k]]

def interactive_search(processor: DocumentProcessor):
    """Интерактивный поиск в командной строке"""
    print("\n" + "="*60)
    print("ИНТЕРАКТИВНЫЙ ПОИСК ПО ДОКУМЕНТАМ")
    print("="*60)
    if processor.use_reranker:
        print("🔄 Реранкер активен: результаты будут дополнительно отсортированы по релевантности")
    else:
        print("⚠️  Реранкер отключен")
    print("Введите поисковый запрос (или 'quit' для выхода)")
    print("-"*60)
    
    while True:
        try:
            query = input("\n🔍 Запрос: ").strip()
            
            if query.lower() in ['quit', 'exit', 'выход', 'q']:
                print("До свидания!")
                break
            
            if not query:
                print("Пустой запрос. Попробуйте снова.")
                continue
            
            print(f"\nПоиск по запросу: '{query}'...")
            results = processor.search(query, k=5)
            
            if not results:
                print("❌ Результатов не найдено")
                continue
            
            print(f"\n📋 Найдено результатов: {len(results)}")
            print("-" * 60)
            
            for i, (doc, score) in enumerate(results, 1):
                print(f"\n#{i} [Скор: {score:.4f}]")
                print(f"Источник: {doc.metadata.get('source', 'Неизвестно')}")
                if 'sheet' in doc.metadata:
                    print(f"Лист: {doc.metadata['sheet']}")
                if 'header' in doc.metadata:
                    print(f"Заголовок: {doc.metadata['header']}")
                print(f"Содержание: {doc.page_content}")
                print("*"*60)
        except KeyboardInterrupt:
            print("\n\nПрерывание пользователем. До свидания!")
            break
        except Exception as e:
            print(f"Ошибка: {e}")

def main():
    parser = argparse.ArgumentParser(description="Обработка документов и поиск с FAISS")
    parser.add_argument("--docs", "-d", type=str, help="Путь к директории с документами", default="data")
    parser.add_argument("--vectorstore", "-v", type=str, default="./vectorstore", 
                       help="Путь к векторному хранилищу")
    parser.add_argument("--load-only", "-l", action="store_true", 
                       help="Только загрузить существующее векторное хранилище")
    parser.add_argument("--no-reranker", action="store_true", 
                       help="Отключить реранкер")
    parser.add_argument("--reranker-model", type=str, default="castorini/monot5-large-msmarco",
                       help="Модель реранкера")
    
    args = parser.parse_args()
    
    # Инициализация процессора
    print("Инициализация процессора документов...")
    processor = DocumentProcessor(
        use_reranker=not args.no_reranker,
        reranker_model=args.reranker_model
    )
    
    if args.load_only:
        # Только загружаем существующее хранилище
        if not processor.load_vectorstore(args.vectorstore):
            print("Не удалось загрузить векторное хранилище")
            return
    else:
        if not args.docs:
            print("Необходимо указать путь к документам (--docs)")
            return
        
        if not os.path.exists(args.docs):
            print(f"Директория не найдена: {args.docs}")
            return
        
        # Загружаем и обрабатываем документы
        documents = processor.load_documents(args.docs)
        
        if not documents:
            print("Не найдено документов для обработки")
            return
        
        # Создаем векторное хранилище
        processor.create_vectorstore(documents)
        
        # Сохраняем векторное хранилище
        processor.save_vectorstore(args.vectorstore)
    
    # Запускаем интерактивный поиск
    interactive_search(processor)

if __name__ == "__main__":
    main()