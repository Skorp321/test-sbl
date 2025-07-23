#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для загрузки, векторизации и поиска по документам
Поддерживает форматы: .docx, .xlsx
"""

import os
import argparse
from pathlib import Path
from typing import List, Optional
import pickle

# LangChain импорты
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Для работы с Excel
import pandas as pd

class DocumentProcessor:
    def __init__(self, embeddings_model: str = "intfloat/multilingual-e5-large"):
        """
        Инициализация процессора документов
        
        Args:
            embeddings_model: Название модели для создания эмбеддингов
        """
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embeddings_model,
            model_kwargs={'device': 'cpu'}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.vectorstore: Optional[FAISS] = None
    
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
        chunks = self.text_splitter.split_documents(documents)
        print(f"Создано {len(chunks)} чанков")
        
        # Создаем векторное хранилище
        print("Создание эмбеддингов и векторного хранилища...")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        print("✓ Векторное хранилище создано")
    
    def save_vectorstore(self, path: str) -> None:
        """Сохраняет векторное хранилище"""
        if self.vectorstore is None:
            print("Нет векторного хранилища для сохранения")
            return
        
        self.vectorstore.save_local(path)
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
            self.vectorstore = FAISS.load_local(path, self.embeddings)
            print(f"✓ Векторное хранилище загружено: {path}")
            return True
        except Exception as e:
            print(f"✗ Ошибка при загрузке векторного хранилища: {e}")
            return False
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        Поиск по векторному хранилищу
        
        Args:
            query: Поисковый запрос
            k: Количество результатов
            
        Returns:
            Список найденных документов
        """
        if self.vectorstore is None:
            print("Векторное хранилище не загружено")
            return []
        
        results = self.vectorstore.similarity_search(query, k=k)
        return results

def interactive_search(processor: DocumentProcessor):
    """Интерактивный поиск в командной строке"""
    print("\n" + "="*60)
    print("ИНТЕРАКТИВНЫЙ ПОИСК ПО ДОКУМЕНТАМ")
    print("="*60)
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
            
            for i, doc in enumerate(results, 1):
                print(f"\n#{i}")
                print(f"Источник: {doc.metadata.get('source', 'Неизвестно')}")
                if 'sheet' in doc.metadata:
                    print(f"Лист: {doc.metadata['sheet']}")
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
    
    args = parser.parse_args()
    
    # Инициализация процессора
    print("Инициализация процессора документов...")
    processor = DocumentProcessor()
    
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