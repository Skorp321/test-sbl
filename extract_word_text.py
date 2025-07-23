#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для извлечения текста из файла client.doc
"""

import os
import re
from text_splitter import TextSplitter

def extract_text_from_doc(file_path):
    """Извлекает текст из .doc файла разными способами"""
    
    # Способ 1: python-docx
    try:
        print("Пытаемся использовать python-docx...")
        from docx import Document
        doc = Document(file_path)
        text_parts = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text.strip())
        if text_parts:
            return '\n'.join(text_parts)
    except Exception as e:
        print(f"python-docx не сработал: {e}")
    
    # Способ 2: antiword (если установлен)
    try:
        print("Пытаемся использовать antiword...")
        import subprocess
        result = subprocess.run(['antiword', file_path], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception as e:
        print(f"antiword не сработал: {e}")
    
    # Способ 3: извлечение текста из бинарного файла
    try:
        print("Пытаемся извлечь текст из бинарного файла...")
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Декодируем в разных кодировках
        for encoding in ['utf-8', 'cp1251', 'latin1', 'utf-16']:
            try:
                decoded = content.decode(encoding, errors='ignore')
                # Ищем русский текст
                russian_text = re.findall(r'[А-Яа-я\s\.\,\!\?\:\;\(\)\-\d]+', decoded)
                if russian_text:
                    text = ' '.join(russian_text)
                    # Очищаем от лишних символов
                    text = re.sub(r'\s+', ' ', text)
                    # Разбиваем на предложения
                    sentences = re.split(r'[\.!?]+', text)
                    meaningful_sentences = []
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if len(sentence) > 10 and re.search(r'[А-Яа-я]{3,}', sentence):
                            meaningful_sentences.append(sentence)
                    
                    if meaningful_sentences:
                        print(f"Найдено {len(meaningful_sentences)} предложений в кодировке {encoding}")
                        return '\n'.join(meaningful_sentences)
            except:
                continue
    except Exception as e:
        print(f"Извлечение из бинарного файла не удалось: {e}")
    
    # Способ 4: поиск специфических паттернов
    try:
        print("Ищем специфические паттерны в файле...")
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Ищем конкретные фразы
        patterns = [
            r'ДОГОВОР\s+ЛИЗИНГА',
            r'[А-Я][А-Я\s]{10,}',
            r'\d+\.\s*[А-Я][а-я\s]{10,}',
            r'[А-Яа-я]{3,}[а-я\s\,\.\!\?\:\;\(\)\-\d]{20,}'
        ]
        
        found_text = []
        for encoding in ['utf-8', 'cp1251', 'latin1']:
            try:
                decoded = content.decode(encoding, errors='ignore')
                for pattern in patterns:
                    matches = re.findall(pattern, decoded)
                    for match in matches:
                        if len(match) > 10:
                            found_text.append(match.strip())
            except:
                continue
        
        if found_text:
            return '\n'.join(found_text)
    except Exception as e:
        print(f"Поиск паттернов не удался: {e}")
    
    return None

def clean_text(text):
    """Очищает текст от мусора"""
    if not text:
        return ""
    
    # Удаляем лишние пробелы
    text = re.sub(r'\s+', ' ', text)
    
    # Удаляем служебные символы
    text = re.sub(r'[^\w\s\.\,\!\?\:\;\(\)\-\№\"\']', ' ', text)
    
    # Удаляем строки короче 5 символов
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if len(line) > 5:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def main():
    """Основная функция"""
    print("=== ИЗВЛЕЧЕНИЕ ТЕКСТА ИЗ client.doc ===\n")
    
    file_path = "client.docx"
    
    if not os.path.exists(file_path):
        print(f"Файл {file_path} не найден!")
        return
    
    # Извлекаем текст
    extracted_text = extract_text_from_doc(file_path)
    
    if not extracted_text:
        print("Не удалось извлечь текст из файла!")
        return
    
    print(f"Извлечено {len(extracted_text)} символов")
    
    # Очищаем текст
    cleaned_text = clean_text(extracted_text)
    
    print(f"После очистки: {len(cleaned_text)} символов")
    
    # Выводим первые 1000 символов
    print("\n=== ИЗВЛЕЧЕННЫЙ ТЕКСТ ===")
    print(cleaned_text[:1000])
    
    if len(cleaned_text) > 1000:
        print("...")
    
    # Сохраняем в файл
    with open("client_doc_extracted.txt", "w", encoding="utf-8") as f:
        f.write("=== ИЗВЛЕЧЕННЫЙ ТЕКСТ ИЗ client.doc ===\n\n")
        f.write(cleaned_text)
    
    print(f"\nТекст сохранен в файл: client_doc_extracted.txt")
    
    # Обрабатываем с помощью TextSplitter
    if cleaned_text:
        print("\n=== ОБРАБОТКА С ПОМОЩЬЮ TextSplitter ===")
        splitter = TextSplitter()
        parts = splitter.split_text(cleaned_text, add_headers=True)
        
        print(f"Текст разбит на {len(parts)} частей")
        
        # Выводим результат
        formatted_output = splitter.format_output(parts)
        print("\n=== РЕЗУЛЬТАТ ОБРАБОТКИ ===")
        print(formatted_output[:1000])
        
        # Сохраняем результат
        with open("client_doc_final.txt", "w", encoding="utf-8") as f:
            f.write("=== ФИНАЛЬНЫЙ РЕЗУЛЬТАТ ОБРАБОТКИ client.doc ===\n\n")
            f.write(formatted_output)
        
        print(f"\nФинальный результат сохранен в файл: client_doc_final.txt")

if __name__ == "__main__":
    main() 