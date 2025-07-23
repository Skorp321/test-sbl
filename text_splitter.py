import re
from typing import List, Dict, Tuple

class TextSplitter:
    def __init__(self):
        # Образцы для разбивки текста
        self.patterns = {
            'section': r'^([А-Я][А-Я\s\-]+)$',  # Заголовки разделов (прописными буквами)
            'subsection': r'^(\d+\.\s*.+)$',     # Подразделы (начинаются с цифры и точки)
            'subpoint': r'^(\d+\.\d+\.\s*.+)$', # Подпункты (цифра.цифра. текст)
            'paragraph': r'^(.+)$'               # Обычные абзацы
        }
        
        # Структура для хранения иерархии
        self.structure = {
            'current_section': '',
            'current_subsection': '',
            'current_subpoint': ''
        }
    
    def identify_text_type(self, line: str) -> str:
        """Определяет тип строки по образцам"""
        line = line.strip()
        if not line:
            return 'empty'
        
        # Проверяем в порядке приоритета
        if re.match(self.patterns['subpoint'], line):
            return 'subpoint'
        elif re.match(self.patterns['subsection'], line):
            return 'subsection'
        elif re.match(self.patterns['section'], line):
            return 'section'
        else:
            return 'paragraph'
    
    def update_structure(self, line: str, line_type: str):
        """Обновляет текущую структуру документа"""
        if line_type == 'section':
            self.structure['current_section'] = line.strip()
            self.structure['current_subsection'] = ''
            self.structure['current_subpoint'] = ''
        elif line_type == 'subsection':
            self.structure['current_subsection'] = line.strip()
            self.structure['current_subpoint'] = ''
        elif line_type == 'subpoint':
            self.structure['current_subpoint'] = line.strip()
    
    def get_current_header(self) -> str:
        """Формирует заголовок для текущего контекста"""
        header_parts = []
        
        if self.structure['current_section']:
            header_parts.append(f"Раздел: {self.structure['current_section']}")
        
        if self.structure['current_subsection']:
            header_parts.append(f"Подраздел: {self.structure['current_subsection']}")
        
        if self.structure['current_subpoint']:
            header_parts.append(f"Подпункт: {self.structure['current_subpoint']}")
        
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
            if line_type in ['section', 'subsection', 'subpoint'] and current_paragraph:
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
            if line_type in ['section', 'subsection', 'subpoint']:
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
            if show_structure and part['type'] in ['section', 'subsection', 'subpoint']:
                output.append(f"\n{'='*50}")
                output.append(f"[{part['type'].upper()}] {part['content']}")
                output.append(f"{'='*50}")
            elif part['type'] == 'content':
                if 'header' in part:
                    output.append(f"\n--- {part['header']} ---")
                output.append(part['content'])
                output.append("")
        
        return '\n'.join(output)
    
    
    # Пример использования
sample_text_default = """ВВЕДЕНИЕ

Данный документ содержит основные положения.

1. Основные понятия

В настоящем разделе определяются ключевые термины.

1.1. Определение терминов

Термин - это слово или словосочетание.

Понятие - это форма мышления.

1.2. Область применения

Настоящие определения применяются во всех случаях.

2. Требования к документации

Документация должна соответствовать стандартам.

2.1. Структура документа

Документ должен иметь четкую структуру.

Каждый раздел должен быть пронумерован.

ЗАКЛЮЧЕНИЕ

В заключение следует отметить важность соблюдения требований."""

def main(sample_text=None):
    if sample_text is None:
        sample_text = sample_text_default

    print("=== ДЕМОНСТРАЦИЯ РАБОТЫ СКРИПТА ===\n")
    
    # Создаем экземпляр класса
    splitter = TextSplitter()
    
    # Разбиваем текст
    parts = splitter.split_text(sample_text)
    
    # Выводим результат
    formatted_output = splitter.format_output(parts)
    print(formatted_output)
    
    print("\n" + "="*60)
    print("СТРУКТУРНЫЙ АНАЛИЗ:")
    print("="*60)
    
    # Показываем структурный анализ
    structured_output = splitter.format_output(parts, show_structure=True)
    print(structured_output)


def interactive_mode():
    """Интерактивный режим для ввода пользователем"""
    print("\n=== ИНТЕРАКТИВНЫЙ РЕЖИМ ===")
    print("Введите текст для разбивки (для завершения ввода введите 'END' на отдельной строке):")
    
    lines = []
    while True:
        try:
            line = input()
            if line.strip() == 'END':
                break
            lines.append(line)
        except KeyboardInterrupt:
            print("\nОтмена ввода.")
            return
    
    if not lines:
        print("Текст не введен.")
        return
    
    user_text = '\n'.join(lines)
    
    # Обрабатываем текст
    splitter = TextSplitter()
    parts = splitter.split_text(user_text)
    
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТ ОБРАБОТКИ:")
    print("="*50)
    
    formatted_output = splitter.format_output(parts)
    print(formatted_output)


if __name__ == "__main__":
    
    # Демонстрация с примером
    main()
    
    # Интерактивный режим
    while True:
        choice = input("\nХотите обработать свой текст? (y/n): ").strip().lower()
        if choice in ['y', 'yes', 'да', 'д']:
            interactive_mode()
        elif choice in ['n', 'no', 'нет', 'н']:
            break
        else:
            print("Пожалуйста, введите 'y' или 'n'")
    
    print("Работа завершена!")