#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для проверки зависимостей перед запуском векторизации
"""

import sys
import subprocess
import importlib.util
from typing import Dict, List, Tuple

def check_module(module_name: str, install_name: str = None) -> bool:
    """Проверяет наличие модуля"""
    try:
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    except ImportError:
        return False

def get_module_version(module_name: str) -> str:
    """Получает версию модуля"""
    try:
        module = importlib.import_module(module_name)
        return getattr(module, '__version__', 'unknown')
    except:
        return 'not found'

def check_dependencies() -> Dict[str, Tuple[bool, str]]:
    """Проверяет все зависимости"""
    dependencies = {
        'docx': ('python-docx', 'python-docx'),
        'langchain': ('langchain', 'langchain'),
        'langchain_community': ('langchain-community', 'langchain-community'),
        'langchain_huggingface': ('langchain-huggingface', 'langchain-huggingface'),
        'chromadb': ('chromadb', 'chromadb'),
        'sentence_transformers': ('sentence-transformers', 'sentence-transformers'),
        'torch': ('torch', 'torch'),
        'numpy': ('numpy', 'numpy'),
        'transformers': ('transformers', 'transformers'),
        'openai': ('openai', 'openai'),
        'uuid': ('uuid', 'built-in'),
        'os': ('os', 'built-in'),
        'typing': ('typing', 'built-in'),
        'dataclasses': ('dataclasses', 'built-in'),
        'pathlib': ('pathlib', 'built-in'),
        're': ('re', 'built-in')
    }
    
    results = {}
    
    for module_name, (display_name, install_name) in dependencies.items():
        is_available = check_module(module_name)
        version = get_module_version(module_name) if is_available else 'not found'
        results[display_name] = (is_available, version)
    
    return results

def check_python_version() -> bool:
    """Проверяет версию Python"""
    return sys.version_info >= (3, 8)

def check_disk_space() -> bool:
    """Проверяет доступное место на диске"""
    try:
        import shutil
        free_bytes = shutil.disk_usage('.').free
        free_gb = free_bytes / (1024**3)
        return free_gb >= 3.0  # Минимум 3GB
    except:
        return True  # Если не можем проверить, считаем что места достаточно

def install_missing_packages(missing_packages: List[str]):
    """Устанавливает недостающие пакеты"""
    if not missing_packages:
        return
    
    print(f"\n🔧 Устанавливаю недостающие пакеты: {', '.join(missing_packages)}")
    
    for package in missing_packages:
        try:
            print(f"Устанавливаю {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} успешно установлен")
        except subprocess.CalledProcessError as e:
            print(f"❌ Ошибка установки {package}: {e}")

def main():
    """Основная функция проверки"""
    print("=== ПРОВЕРКА ЗАВИСИМОСТЕЙ ДЛЯ ВЕКТОРИЗАЦИИ ===\n")
    
    # Проверка версии Python
    print("🐍 Проверка версии Python...")
    if check_python_version():
        print(f"✅ Python {sys.version.split()[0]} - OK")
    else:
        print(f"❌ Python {sys.version.split()[0]} - Требуется Python 3.8+")
        return False
    
    # Проверка места на диске
    print("\n💾 Проверка места на диске...")
    if check_disk_space():
        print("✅ Достаточно места на диске")
    else:
        print("⚠️  Мало места на диске (рекомендуется 3GB+)")
    
    # Проверка зависимостей
    print("\n📦 Проверка зависимостей...")
    results = check_dependencies()
    
    available = []
    missing = []
    
    for package, (is_available, version) in results.items():
        if is_available:
            if version == 'built-in':
                print(f"✅ {package} (встроенный)")
            else:
                print(f"✅ {package} v{version}")
            available.append(package)
        else:
            print(f"❌ {package} - не найден")
            if package not in ['built-in']:
                missing.append(package)
    
    print(f"\n📊 Статистика:")
    print(f"   Доступно: {len(available)}")
    print(f"   Отсутствует: {len(missing)}")
    
    # Критические зависимости
    critical_deps = ['python-docx', 'langchain', 'langchain-community', 'langchain-huggingface', 'chromadb', 'sentence-transformers']
    missing_critical = [dep for dep in missing if dep in critical_deps]
    
    if missing_critical:
        print(f"\n🚨 Критические зависимости отсутствуют:")
        for dep in missing_critical:
            print(f"   - {dep}")
        
        print(f"\n💡 Для установки выполните:")
        print(f"   pip install {' '.join(missing_critical)}")
        
        # Предложение автоматической установки
        response = input("\n❓ Установить недостающие пакеты автоматически? (y/n): ")
        if response.lower() in ['y', 'yes', 'да', 'д']:
            install_missing_packages(missing_critical)
        
        return False
    
    # Проверка наличия файлов
    print("\n📁 Проверка файлов...")
    required_files = ['client.docx', 'improved_text_splitter.py']
    
    for file_name in required_files:
        try:
            with open(file_name, 'r') as f:
                print(f"✅ {file_name} найден")
        except FileNotFoundError:
            print(f"❌ {file_name} не найден")
            if file_name == 'client.docx':
                print("   Поместите ваш .docx файл в текущую директорию")
            elif file_name == 'improved_text_splitter.py':
                print("   Убедитесь, что файл improved_text_splitter.py доступен")
    
    print("\n✅ Проверка завершена!")
    print("\n🚀 Готов к запуску:")
    print("   python docx_to_chromadb.py")
    print("   python search_demo.py")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nПрерывание пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\nОшибка: {e}")
        sys.exit(1) 