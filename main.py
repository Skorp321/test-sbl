#!/usr/bin/env python3
"""
AI-агент для анализа целесообразности участия в конкурсах
Система включает трех специализированных агентов:
1. Юрист - анализирует правовые аспекты
2. Экономист - оценивает финансовую целесообразность  
3. Руководитель - принимает итоговое решение
"""

import os
import sys
from dotenv import load_dotenv
from concurse.contest_analysis_graph import ContestAnalysisGraph, create_sample_contest
from concurse.models import ContestInfo
import json


def setup_environment():
    """Настройка окружения"""
    load_dotenv()
    
    # Отключаем LangSmith если нет API ключа
    if not os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
    
    if not os.getenv("MISTRAL_API_KEY"):
        print("❌ Ошибка: Не найден MISTRAL_API_KEY")
        print("Создайте файл .env и добавьте:")
        print("MISTRAL_API_KEY=your_mistral_api_key_here")
        sys.exit(1)


def print_detailed_analysis(result: dict):
    """Вывод детального анализа"""
    
    print("\n" + "="*80)
    print("📋 ДЕТАЛЬНЫЙ АНАЛИЗ КОНКУРСА")
    print("="*80)
    
    # Информация о конкурсе
    contest = result.get("contest_info", {})
    print(f"\n📄 ИНФОРМАЦИЯ О КОНКУРСЕ:")
    print(f"Название: {contest.get('title', 'Не указано')}")
    print(f"Бюджет: {contest.get('budget', 'Не указан'):,} руб." if contest.get('budget') else "Бюджет: Не указан")
    print(f"Срок подачи: {contest.get('deadline', 'Не указан')}")
    
    # Юридическое заключение
    legal = result.get("legal_analysis", {})
    if legal:
        print(f"\n🏛️ ЮРИДИЧЕСКОЕ ЗАКЛЮЧЕНИЕ:")
        print(f"Уровень риска: {legal.get('risk_level', 'Не определен').upper()}")
        print(f"Уверенность анализа: {legal.get('confidence', 0):.0%}")
        
        risks = legal.get('legal_risks', [])
        if risks:
            print("Основные юридические риски:")
            for i, risk in enumerate(risks[:3], 1):
                print(f"  {i}. {risk}")
        
        docs = legal.get('required_documents', [])
        if docs:
            print("Необходимые документы:")
            for i, doc in enumerate(docs[:3], 1):
                print(f"  {i}. {doc}")
        
        print(f"Рекомендация юриста: {legal.get('recommendation', 'Не предоставлена')[:200]}...")
    
    # Экономическое заключение
    economic = result.get("economic_analysis", {})
    if economic:
        print(f"\n💰 ЭКОНОМИЧЕСКОЕ ЗАКЛЮЧЕНИЕ:")
        print(f"Уровень риска: {economic.get('risk_level', 'Не определен').upper()}")
        print(f"Уверенность анализа: {economic.get('confidence', 0):.0%}")
        
        costs = economic.get('estimated_costs', {})
        if costs:
            total_cost = sum(costs.values()) if isinstance(costs, dict) else costs
            print(f"Ожидаемые затраты: {total_cost:,.0f} руб.")
            if isinstance(costs, dict):
                for category, cost in costs.items():
                    print(f"  • {category}: {cost:,.0f} руб.")
        
        profit = economic.get('potential_profit')
        if profit:
            print(f"Потенциальная прибыль: {profit:,.0f} руб.")
        
        roi = economic.get('roi_estimate')
        if roi:
            print(f"Ожидаемый ROI: {roi:.1%}")
        
        opportunities = economic.get('market_opportunities', [])
        if opportunities:
            print("Рыночные возможности:")
            for i, opp in enumerate(opportunities[:3], 1):
                print(f"  {i}. {opp}")
        
        print(f"Рекомендация экономиста: {economic.get('recommendation', 'Не предоставлена')[:200]}...")
    
    # Решение руководителя
    decision = result.get("managerial_decision", {})
    if decision:
        print(f"\n👔 РЕШЕНИЕ РУКОВОДИТЕЛЯ:")
        decision_text = decision.get('decision', 'Не принято')
        print(f"ИТОГОВОЕ РЕШЕНИЕ: {decision_text.upper()}")
        print(f"Уверенность в решении: {decision.get('confidence', 0):.0%}")
        
        factors = decision.get('key_factors', [])
        if factors:
            print("Ключевые факторы решения:")
            for i, factor in enumerate(factors, 1):
                print(f"  {i}. {factor}")
        
        conditions = decision.get('conditions', [])
        if conditions:
            print("Условия участия:")
            for i, condition in enumerate(conditions, 1):
                print(f"  {i}. {condition}")
        
        steps = decision.get('next_steps', [])
        if steps:
            print("Следующие шаги:")
            for i, step in enumerate(steps, 1):
                print(f"  {i}. {step}")
        
        print(f"Обоснование: {decision.get('reasoning', 'Не предоставлено')[:300]}...")


def create_custom_contest():
    """Создание пользовательского конкурса"""
    print("\n📝 СОЗДАНИЕ НОВОГО КОНКУРСА ДЛЯ АНАЛИЗА")
    print("-" * 50)
    
    title = input("Введите название конкурса: ").strip()
    if not title:
        title = "Пользовательский конкурс"
    
    description = input("Введите описание конкурса: ").strip()
    if not description:
        description = "Описание не предоставлено"
    
    budget_input = input("Введите бюджет конкурса (руб., или Enter для пропуска): ").strip()
    budget = None
    if budget_input:
        try:
            budget = float(budget_input.replace(",", "").replace(" ", ""))
        except ValueError:
            print("Некорректный формат бюджета, пропускаю...")
    
    deadline = input("Введите срок подачи заявки (или Enter для пропуска): ").strip()
    
    print("Введите требования к участникам (по одному на строку, пустая строка для завершения):")
    requirements = []
    while True:
        req = input("  Требование: ").strip()
        if not req:
            break
        requirements.append(req)
    
    return ContestInfo(
        title=title,
        description=description,
        budget=budget,
        deadline=deadline if deadline else None,
        requirements=requirements,
        documentation=[],
        evaluation_criteria=[],
        additional_info={}
    )


def main():
    """Основная функция"""
    setup_environment()
    
    print("🤖 AI-АГЕНТ АНАЛИЗА КОНКУРСОВ")
    print("=" * 50)
    print("Система анализа целесообразности участия в конкурсах")
    print("Включает заключения юриста, экономиста и решение руководителя")
    print()
    
    # Создаем граф анализа
    try:
        analysis_graph = ContestAnalysisGraph()
        print("✅ Система инициализирована успешно")
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        return
    
    while True:
        print("\n" + "="*50)
        print("ВЫБЕРИТЕ ДЕЙСТВИЕ:")
        print("1. Анализ примера конкурса")
        print("2. Создать и проанализировать свой конкурс")
        print("3. Выход")
        
        choice = input("\nВаш выбор (1-3): ").strip()
        
        if choice == "1":
            # Анализ примера
            print("\n🔍 Анализирую пример конкурса...")
            contest = create_sample_contest()
            
            try:
                result = analysis_graph.analyze_contest(contest)
                
                # Краткое резюме
                summary = analysis_graph.get_analysis_summary(result)
                print(summary)
                
                # Предложение детального анализа
                detail = input("\nПоказать детальный анализ? (y/n): ").strip().lower()
                if detail in ['y', 'yes', 'да', 'д']:
                    print_detailed_analysis(result)
                
            except Exception as e:
                print(f"❌ Ошибка при анализе: {e}")
        
        elif choice == "2":
            # Пользовательский конкурс
            try:
                contest = create_custom_contest()
                print(f"\n🔍 Анализирую конкурс: {contest.title}")
                
                result = analysis_graph.analyze_contest(contest)
                
                # Краткое резюме
                summary = analysis_graph.get_analysis_summary(result)
                print(summary)
                
                # Предложение детального анализа
                detail = input("\nПоказать детальный анализ? (y/n): ").strip().lower()
                if detail in ['y', 'yes', 'да', 'д']:
                    print_detailed_analysis(result)
                
            except Exception as e:
                print(f"❌ Ошибка при анализе: {e}")
        
        elif choice == "3":
            print("\n👋 До свидания!")
            break
        
        else:
            print("❌ Некорректный выбор. Попробуйте снова.")


if __name__ == "__main__":
    main() 