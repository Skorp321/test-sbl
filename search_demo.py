#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Демонстрационный скрипт для поиска в векторной базе данных ChromaDB
"""

import os
from typing import List, Dict, Any
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

class DocumentSearch:
    """Класс для поиска документов в ChromaDB"""
    
    def __init__(self, 
                 chromadb_path: str = "./chromadb_documents",
                 embedding_model: str = "intfloat/multilingual-e5-large"):
        """
        Инициализация поиска
        
        Args:
            chromadb_path: Путь к базе данных ChromaDB
            embedding_model: Модель для создания эмбеддингов
        """
        self.chromadb_path = chromadb_path
        self.embedding_model = embedding_model
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
    def search_structured(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Поиск в структурированных фрагментах"""
        try:
            vectorstore = Chroma(
                collection_name="contract_documents_structured",
                embedding_function=self.embeddings,
                persist_directory=self.chromadb_path
            )
            
            # Запрашиваем больше результатов для дедупликации
            results = vectorstore.similarity_search_with_score(query, k=k*3)
            
            formatted_results = []
            seen_contents = set()  # Для отслеживания уникальности
            
            for doc, score in results:
                # Создаем ключ для дедупликации на основе содержимого
                content_key = doc.page_content[:100]  # Первые 100 символов
                
                if content_key not in seen_contents:
                    seen_contents.add(content_key)
                    formatted_results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "similarity_score": score
                    })
                    
                    # Останавливаемся когда получили нужное количество уникальных результатов
                    if len(formatted_results) >= k:
                        break
            
            return formatted_results
            
        except Exception as e:
            print(f"Ошибка при поиске в структурированных фрагментах: {e}")
            return []
    
    def search_recursive(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Поиск в рекурсивных фрагментах"""
        try:
            vectorstore = Chroma(
                collection_name="contract_documents_recursive",
                embedding_function=self.embeddings,
                persist_directory=self.chromadb_path
            )
            
            # Запрашиваем больше результатов для дедупликации
            results = vectorstore.similarity_search_with_score(query, k=k*3)
            
            formatted_results = []
            seen_contents = set()  # Для отслеживания уникальности
            
            for doc, score in results:
                # Создаем ключ для дедупликации на основе содержимого
                content_key = doc.page_content[:100]  # Первые 100 символов
                
                if content_key not in seen_contents:
                    seen_contents.add(content_key)
                    formatted_results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "similarity_score": score
                    })
                    
                    # Останавливаемся когда получили нужное количество уникальных результатов
                    if len(formatted_results) >= k:
                        break
            
            return formatted_results
            
        except Exception as e:
            print(f"Ошибка при поиске в рекурсивных фрагментах: {e}")
            return []
    
    def search_both(self, query: str, k: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """Поиск в обеих коллекциях"""
        return {
            "structured": self.search_structured(query, k),
            "recursive": self.search_recursive(query, k)
        }
    
    def search_with_deduplication(self, query: str, k: int = 5, 
                                similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Поиск с улучшенной дедупликацией и фильтрацией по схожести"""
        try:
            # Объединяем результаты из обеих коллекций
            structured_results = self.search_structured(query, k*2)
            recursive_results = self.search_recursive(query, k*2)
            
            # Объединяем и сортируем по релевантности
            all_results = structured_results + recursive_results
            all_results.sort(key=lambda x: x['similarity_score'])
            
            # Дедупликация с учетом метаданных
            unique_results = []
            seen_keys = set()
            
            for result in all_results:
                # Создаем уникальный ключ на основе содержимого и метаданных
                content_preview = result['content'][:150]
                metadata_key = f"{result['metadata'].get('chunk_id', '')}_{result['metadata'].get('type', '')}"
                unique_key = f"{content_preview}_{metadata_key}"
                
                # Проверяем уникальность и минимальную схожесть
                if (unique_key not in seen_keys and 
                    result['similarity_score'] <= similarity_threshold):
                    seen_keys.add(unique_key)
                    unique_results.append(result)
                    
                    if len(unique_results) >= k:
                        break
            
            return unique_results
            
        except Exception as e:
            print(f"Ошибка при поиске с дедупликацией: {e}")
            return []
    
    def print_search_results(self, results: List[Dict[str, Any]], title: str = "Результаты поиска"):
        """Красивый вывод результатов поиска"""
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")
        
        if not results:
            print("Результаты не найдены.")
            return
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Похожесть: {result['similarity_score']:.4f}")
            print(f"Тип: {result['metadata'].get('type', 'N/A')}")
            
            if 'header' in result['metadata']:
                print(f"Заголовок: {result['metadata']['header']}")
            
            print(f"Источник: {result['metadata'].get('source', 'N/A')}")
            print(f"Размер: {result['metadata'].get('chunk_size', 'N/A')} символов")
            
            # Показываем первые 300 символов содержимого
            content = result['content']
            if len(content) > 300:
                content = content[:300] + "..."
            
            print(f"Содержимое:\n{content}")
            print(f"{'-'*40}")

def interactive_search():
    """Интерактивный поиск"""
    print("=== ИНТЕРАКТИВНЫЙ ПОИСК В ВЕКТОРНОЙ БАЗЕ ДАННЫХ ===")
    
    # Проверяем наличие базы данных
    if not os.path.exists("./chromadb_documents"):
        print("Векторная база данных не найдена!")
        print("Сначала запустите docx_to_chromadb.py для создания базы данных.")
        return
    
    searcher = DocumentSearch()
    
    print("\nДоступные команды:")
    print("- Введите запрос для поиска (с дедупликацией)")
    print("- 'demo' - демонстрационные запросы")
    print("- 'both' - поиск в обеих коллекциях отдельно")
    print("- 'quit' - выход")
    
    while True:
        query = input("\nВведите запрос: ").strip()
        
        if query.lower() == 'quit':
            break
        
        if query.lower() == 'both':
            # Поиск в обеих коллекциях отдельно
            search_query = input("Введите запрос для поиска в обеих коллекциях: ").strip()
            if search_query:
                results = searcher.search_both(search_query, k=3)
                
                searcher.print_search_results(
                    results["structured"], 
                    f"Структурированные фрагменты по запросу: '{search_query}'"
                )
                
                searcher.print_search_results(
                    results["recursive"], 
                    f"Рекурсивные фрагменты по запросу: '{search_query}'"
                )
            continue
            
        if query.lower() == 'demo':
            demo_queries = [
                "лизинговые платежи",
                "страхование объекта лизинга",
                "права лизингодателя",
                "договор поставки",
                "возврат объекта",
                "форс-мажор"
            ]
            
            for demo_query in demo_queries:
                print(f"\n{'='*60}")
                print(f"ДЕМО-ЗАПРОС: '{demo_query}'")
                print(f"{'='*60}")
                
                # Используем улучшенный поиск с дедупликацией
                results = searcher.search_with_deduplication(demo_query, k=3, similarity_threshold=0.9)
                
                searcher.print_search_results(
                    results, 
                    f"Лучшие результаты (с дедупликацией)"
                )
            
            continue
        
        if not query:
            continue
        
        # Выполняем поиск с дедупликацией
        results = searcher.search_with_deduplication(query, k=3, similarity_threshold=0.9)
        
        # Выводим результаты
        searcher.print_search_results(
            results, 
            f"Лучшие результаты по запросу: '{query}' (с дедупликацией)"
        )

def main():
    """Основная функция"""
    try:
        interactive_search()
    except KeyboardInterrupt:
        print("\n\nВыход из программы.")
    except Exception as e:
        print(f"\nОшибка: {e}")

if __name__ == "__main__":
    main() 