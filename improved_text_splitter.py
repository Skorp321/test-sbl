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
        
        # Новые атрибуты для полного контекста
        self.hierarchy_content = {
            'section': '',
            'main_section': '',
            'subsection': '',
            'subpoint': '',
            'letter_point': ''
        }
        self.current_hierarchy = []
    
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
    
    def update_hierarchy_content(self, line: str, line_type: str):
        """Обновляет полное содержимое каждого уровня иерархии"""
        if line_type == 'section':
            self.hierarchy_content['section'] = line.strip()
            self.hierarchy_content['main_section'] = ''
            self.hierarchy_content['subsection'] = ''
            self.hierarchy_content['subpoint'] = ''
            self.hierarchy_content['letter_point'] = ''
        elif line_type == 'main_section':
            self.hierarchy_content['main_section'] = line.strip()
            self.hierarchy_content['subsection'] = ''
            self.hierarchy_content['subpoint'] = ''
            self.hierarchy_content['letter_point'] = ''
        elif line_type == 'subsection':
            self.hierarchy_content['subsection'] = line.strip()
            self.hierarchy_content['subpoint'] = ''
            self.hierarchy_content['letter_point'] = ''
        elif line_type == 'subpoint':
            self.hierarchy_content['subpoint'] = line.strip()
            self.hierarchy_content['letter_point'] = ''
        elif line_type == 'letter_point':
            self.hierarchy_content['letter_point'] = line.strip()
    
    def get_full_context(self) -> str:
        """Формирует полный контекст от корневого раздела до текущего подпункта"""
        context_parts = []
        
        if self.hierarchy_content['section']:
            context_parts.append(self.hierarchy_content['section'])
        
        if self.hierarchy_content['main_section']:
            context_parts.append(self.hierarchy_content['main_section'])
        
        if self.hierarchy_content['subsection']:
            context_parts.append(self.hierarchy_content['subsection'])
        
        if self.hierarchy_content['subpoint']:
            context_parts.append(self.hierarchy_content['subpoint'])
        
        if self.hierarchy_content['letter_point']:
            context_parts.append(self.hierarchy_content['letter_point'])
        
        return ''.join(context_parts)
    
    def split_text_with_context(self, text: str) -> List[Dict]:
        """Разбивает текст на части с полным контекстом иерархии"""
        lines = text.split('\n')
        result = []
        current_content = []
        current_main_header = ""  # Основной заголовок раздела (например "16. Право истребования...")
        current_subheader = ""    # Текущий подзаголовок (например "16.1. В случае...")
        
        def save_current_fragment():
            """Сохраняет текущий фрагмент если есть заголовки"""
            if current_main_header and current_subheader:
                if current_content:
                    content_text = '\n'.join(current_content).strip()
                    fragment_content = current_main_header + '\n' + current_subheader + '\n' + content_text
                else:
                    fragment_content = current_main_header + '\n' + current_subheader
                
                result.append({
                    'type': 'content',
                    'content': fragment_content,
                    'header': current_main_header.strip()
                })
        
        for line in lines:
            line_type = self.identify_text_type(line)
            
            if line_type == 'empty':
                if current_content and current_subheader:  # Добавляем пустые строки только если есть активный подраздел
                    current_content.append('')
                continue
            
            # Если встретили новый основной раздел (например "16. Право истребования...")
            if line_type == 'main_section':
                # Сохраняем предыдущий фрагмент
                save_current_fragment()
                
                # Сохраняем новый основной заголовок
                current_main_header = line.strip()
                current_subheader = ""
                current_content = []
                
                # Обновляем структуру
                self.update_hierarchy_content(line, line_type)
                self.update_structure(line, line_type)
                
            # Если встретили подраздел (например "16.1. В случае...")
            elif line_type == 'subsection':
                # Сохраняем предыдущий фрагмент
                save_current_fragment()
                
                # Начинаем новый подраздел
                current_subheader = line.strip()
                current_content = []
                
                # Обновляем структуру
                self.update_hierarchy_content(line, line_type)
                self.update_structure(line, line_type)
                
            # Если встретили подпункт или пункт с буквой
            elif line_type in ['subpoint', 'letter_point']:
                # Добавляем к содержимому текущего подраздела
                if current_subheader:  # Только если есть активный подраздел
                    current_content.append(line.strip())
                
                # Обновляем структуру
                self.update_hierarchy_content(line, line_type)
                self.update_structure(line, line_type)
                
            # Если встретили раздел (заголовок прописными буквами)
            elif line_type == 'section':
                # Сохраняем предыдущий фрагмент
                save_current_fragment()
                
                # Начинаем новый раздел
                current_main_header = line.strip()
                current_subheader = ""
                current_content = []
                
                # Обновляем структуру
                self.update_hierarchy_content(line, line_type)
                self.update_structure(line, line_type)
                
            else:
                # Накапливаем содержимое обычных абзацев
                if current_subheader:  # Только если есть активный подраздел
                    current_content.append(line.strip())
        
        # Добавляем последний фрагмент
        save_current_fragment()
        
        return result
    
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
        doc = Document("data/client.docx")
        text_parts = []
        
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text:
                text_parts.append(text)
        
        full_text = '\n'.join(text_parts)
        
        print(f"Извлечено {len(full_text)} символов")
        
        # Обрабатываем улучшенным сплиттером с контекстом
        splitter = ImprovedTextSplitter()
        parts = splitter.split_text_with_context(full_text)
        
        # Получаем статистику для контекстных фрагментов
        print("\n=== СТАТИСТИКА ===")
        print(f"Создано контекстных фрагментов: {len(parts)}")
        
        # Выводим примеры фрагментов с контекстом
        print("\n=== ПРИМЕРЫ ФРАГМЕНТОВ С КОНТЕКСТОМ ===")
        for i, part in enumerate(parts[:5], 1):  # Первые 5 фрагментов
            context = part.get('context', '')
            content = part['content']
            print(f"\nФрагмент {i}:")
            print(f"Контекст: {context[:100]}...")
            print(f"Полное содержание: {content[:200]}...")
            print("-" * 50)
        
        # Сохраняем результат
        with open("client_docx_improved.txt", "w", encoding="utf-8") as f:
            f.write("=== УЛУЧШЕННАЯ ОБРАБОТКА С КОНТЕКСТОМ client.docx ===\n\n")
            for i, part in enumerate(parts, 1):
                f.write(f"=== ФРАГМЕНТ {i} ===\n")
                f.write(f"Контекст: {part.get('context', '')}\n")
                f.write(f"Содержание:\n{part['content']}\n\n")
        
        # Сохраняем структурный анализ
        with open("client_docx_improved_structure.txt", "w", encoding="utf-8") as f:
            f.write("=== СТРУКТУРНЫЙ АНАЛИЗ С КОНТЕКСТОМ client.docx ===\n\n")
            for i, part in enumerate(parts, 1):
                f.write(f"Фрагмент {i}:\n")
                f.write(f"- Контекст: {part.get('context', 'Нет контекста')}\n")
                f.write(f"- Длина содержания: {len(part['content'])} символов\n")
                f.write(f"- Первые 100 символов: {part['content'][:100]}...\n")
                f.write("-" * 60 + "\n\n")
        
        print("\n=== СОХРАНЕННЫЕ ФАЙЛЫ ===")
        print("1. client_docx_improved.txt - улучшенная обработка с контекстом")
        print("2. client_docx_improved_structure.txt - структурный анализ с контекстом")
        
        return True
        
    except Exception as e:
        print(f"Ошибка: {e}")
        return False

if __name__ == "__main__":
    process_client_docx_improved() 