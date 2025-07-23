#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Улучшенная версия TextSplitter для работы с договорами и документами
"""

import re
from typing import List, Dict, Tuple

class ImprovedTextSplitter:
    def __init__(self):
        # Улучшенные образцы для разбивки текста
        self.patterns = {
            # Заголовки разделов (прописными буквами)
            'section': r'^([А-Я][А-Я\s\-]+)$',
            
            # Основные разделы (только цифра с точкой)
            'main_section': r'^(\d+\.\s*.+)$',
            
            # Подразделы (цифра.цифра. текст)
            'subsection': r'^(\d+\.\d+\.\s*.+)$',
            
            # Подпункты (цифра.цифра.цифра. текст)
            'subpoint': r'^(\d+\.\d+\.\d+\.\s*.+)$',
            
            # Пункты с буквами (а), б), в) и т.д.
            'letter_point': r'^([а-я]\)\s*.+)$',
            
            # Обычные абзацы
            'paragraph': r'^(.+)$'
        }
        
        # Структура для хранения иерархии
        self.structure = {
            'current_section': '',
            'current_main_section': '',
            'current_subsection': '',
            'current_subpoint': '',
            'current_letter_point': ''
        }
    
    def identify_text_type(self, line: str) -> str:
        """Определяет тип строки по образцам"""
        line = line.strip()
        if not line:
            return 'empty'
        
        # Проверяем в порядке приоритета (от более специфичного к общему)
        if re.match(self.patterns['subpoint'], line):
            return 'subpoint'
        elif re.match(self.patterns['subsection'], line):
            return 'subsection'
        elif re.match(self.patterns['letter_point'], line):
            return 'letter_point'
        elif re.match(self.patterns['main_section'], line):
            return 'main_section'
        elif re.match(self.patterns['section'], line):
            return 'section'
        else:
            return 'paragraph'
    
    def update_structure(self, line: str, line_type: str):
        """Обновляет текущую структуру документа"""
        if line_type == 'section':
            self.structure['current_section'] = line.strip()
            self.structure['current_main_section'] = ''
            self.structure['current_subsection'] = ''
            self.structure['current_subpoint'] = ''
            self.structure['current_letter_point'] = ''
        elif line_type == 'main_section':
            self.structure['current_main_section'] = line.strip()
            self.structure['current_subsection'] = ''
            self.structure['current_subpoint'] = ''
            self.structure['current_letter_point'] = ''
        elif line_type == 'subsection':
            self.structure['current_subsection'] = line.strip()
            self.structure['current_subpoint'] = ''
            self.structure['current_letter_point'] = ''
        elif line_type == 'subpoint':
            self.structure['current_subpoint'] = line.strip()
            self.structure['current_letter_point'] = ''
        elif line_type == 'letter_point':
            self.structure['current_letter_point'] = line.strip()
    
    def get_current_header(self) -> str:
        """Формирует заголовок для текущего контекста"""
        header_parts = []
        
        if self.structure['current_section']:
            header_parts.append(f"Раздел: {self.structure['current_section']}")
        
        if self.structure['current_main_section']:
            header_parts.append(f"Пункт: {self.structure['current_main_section']}")
        
        if self.structure['current_subsection']:
            header_parts.append(f"Подпункт: {self.structure['current_subsection']}")
        
        if self.structure['current_subpoint']:
            header_parts.append(f"Подраздел: {self.structure['current_subpoint']}")
        
        if self.structure['current_letter_point']:
            header_parts.append(f"Пункт: {self.structure['current_letter_point']}")
        
        return " | ".join(header_parts) if header_parts else "Документ"
    
    def split_text(self, text: str, add_headers: bool = True) -> List[Dict]:
        """Разбивает текст на части с добавлением заголовков"""
        lines = text.split('\n')
        result = []
        current_paragraph = []
        
        for line in lines:
            line_type = self.identify_text_type(line)
            
            if line_type == 'empty':
                continue
            
            # Если встретили структурный элемент, сохраняем накопленный абзац
            if line_type in ['section', 'main_section', 'subsection', 'subpoint', 'letter_point'] and current_paragraph:
                if add_headers:
                    header = self.get_current_header()
                    result.append({
                        'type': 'content',
                        'header': header,
                        'content': '\n'.join(current_paragraph).strip()
                    })
                else:
                    result.append({
                        'type': 'content',
                        'content': '\n'.join(current_paragraph).strip()
                    })
                current_paragraph = []
            
            # Обновляем структуру
            self.update_structure(line, line_type)
            
            # Добавляем структурные элементы как отдельные записи
            if line_type in ['section', 'main_section', 'subsection', 'subpoint', 'letter_point']:
                result.append({
                    'type': line_type,
                    'content': line.strip()
                })
            else:
                # Накапливаем обычные абзацы
                current_paragraph.append(line)
        
        # Добавляем последний абзац, если он есть
        if current_paragraph:
            if add_headers:
                header = self.get_current_header()
                result.append({
                    'type': 'content',
                    'header': header,
                    'content': '\n'.join(current_paragraph).strip()
                })
            else:
                result.append({
                    'type': 'content',
                    'content': '\n'.join(current_paragraph).strip()
                })
        
        return result
    
    def format_output(self, parts: List[Dict], show_structure: bool = False) -> str:
        """Форматирует результат для вывода"""
        output = []
        
        for part in parts:
            if show_structure and part['type'] in ['section', 'main_section', 'subsection', 'subpoint', 'letter_point']:
                output.append(f"\n{'='*50}")
                output.append(f"[{part['type'].upper()}] {part['content']}")
                output.append(f"{'='*50}")
            elif part['type'] == 'content':
                if 'header' in part:
                    output.append(f"\n--- {part['header']} ---")
                output.append(part['content'])
                output.append("")
        
        return '\n'.join(output)
    
    def get_statistics(self, parts: List[Dict]) -> Dict:
        """Возвращает статистику по структуре документа"""
        stats = {
            'sections': 0,
            'main_sections': 0,
            'subsections': 0,
            'subpoints': 0,
            'letter_points': 0,
            'content_parts': 0
        }
        
        for part in parts:
            if part['type'] == 'section':
                stats['sections'] += 1
            elif part['type'] == 'main_section':
                stats['main_sections'] += 1
            elif part['type'] == 'subsection':
                stats['subsections'] += 1
            elif part['type'] == 'subpoint':
                stats['subpoints'] += 1
            elif part['type'] == 'letter_point':
                stats['letter_points'] += 1
            elif part['type'] == 'content':
                stats['content_parts'] += 1
        
        return stats

# Функция для обработки client.docx с улучшенным сплиттером
def process_client_docx_improved():
    """Обрабатывает client.docx с улучшенным сплиттером"""
    try:
        from docx import Document
        
        print("=== УЛУЧШЕННАЯ ОБРАБОТКА client.docx ===\n")
        
        # Читаем файл
        doc = Document("client.docx")
        text_parts = []
        
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text:
                text_parts.append(text)
        
        full_text = '\n'.join(text_parts)
        
        print(f"Извлечено {len(full_text)} символов")
        
        # Обрабатываем улучшенным сплиттером
        splitter = ImprovedTextSplitter()
        parts = splitter.split_text(full_text, add_headers=True)
        
        # Получаем статистику
        stats = splitter.get_statistics(parts)
        
        # Выводим результат
        formatted_output = splitter.format_output(parts)
        
        print("\n=== СТАТИСТИКА ===")
        print(f"Разделов: {stats['sections']}")
        print(f"Основных пунктов: {stats['main_sections']}")
        print(f"Подпунктов: {stats['subsections']}")
        print(f"Подразделов: {stats['subpoints']}")
        print(f"Буквенных пунктов: {stats['letter_points']}")
        print(f"Содержательных частей: {stats['content_parts']}")
        
        # Сохраняем результат
        with open("client_docx_improved.txt", "w", encoding="utf-8") as f:
            f.write("=== УЛУЧШЕННАЯ ОБРАБОТКА client.docx ===\n\n")
            f.write(formatted_output)
        
        # Сохраняем структурный анализ
        structured_output = splitter.format_output(parts, show_structure=True)
        with open("client_docx_improved_structure.txt", "w", encoding="utf-8") as f:
            f.write("=== УЛУЧШЕННЫЙ СТРУКТУРНЫЙ АНАЛИЗ client.docx ===\n\n")
            f.write(structured_output)
        
        print("\n=== НАЙДЕННЫЕ ОСНОВНЫЕ ПУНКТЫ ===")
        main_sections = [p for p in parts if p['type'] == 'main_section']
        for i, section in enumerate(main_sections[:10], 1):  # Первые 10
            print(f"{i}. {section['content']}")
        
        print("\n=== НАЙДЕННЫЕ ПОДПУНКТЫ ===")
        subsections = [p for p in parts if p['type'] == 'subsection']
        for i, subsection in enumerate(subsections[:10], 1):  # Первые 10
            print(f"{i}. {subsection['content']}")
        
        print("\n=== СОХРАНЕННЫЕ ФАЙЛЫ ===")
        print("1. client_docx_improved.txt - улучшенная обработка")
        print("2. client_docx_improved_structure.txt - структурный анализ")
        
        return True
        
    except Exception as e:
        print(f"Ошибка: {e}")
        return False

if __name__ == "__main__":
    process_client_docx_improved() 