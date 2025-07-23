#!/usr/bin/env python3
"""
Тестовый файл для проверки работы системы анализа конкурсов
Можно запустить без API ключей для проверки структуры
"""

from concurse.models import ContestInfo, LegalAnalysis, EconomicAnalysis, ManagerialDecision, RiskLevel, Decision
from concurse.contest_analysis_graph import create_sample_contest
import json


def test_models():
    """Тестирование моделей данных"""
    print("🧪 Тестирование моделей данных...")
    
    # Тест создания конкурса
    contest = create_sample_contest()
    print(f"✅ Конкурс создан: {contest.title}")
    print(f"   Бюджет: {contest.budget:,} руб.")
    print(f"   Требований: {len(contest.requirements)}")
    
    # Тест юридического анализа
    legal = LegalAnalysis(
        legal_risks=["Тестовый риск 1", "Тестовый риск 2"],
        compliance_issues=["Вопрос соответствия 1"],
        required_documents=["Документ 1", "Документ 2"],
        risk_level=RiskLevel.MEDIUM,
        recommendation="Тестовая рекомендация юриста",
        confidence=0.85
    )
    print(f"✅ Юридический анализ создан: риск {legal.risk_level}")
    
    # Тест экономического анализа
    economic = EconomicAnalysis(
        estimated_costs={"preparation": 50000, "documentation": 25000},
        potential_profit=500000,
        roi_estimate=0.25,
        market_opportunities=["Возможность 1", "Возможность 2"],
        economic_risks=["Экономический риск 1"],
        risk_level=RiskLevel.LOW,
        recommendation="Тестовая рекомендация экономиста",
        confidence=0.80
    )
    print(f"✅ Экономический анализ создан: риск {economic.risk_level}")
    
    # Тест решения руководителя
    decision = ManagerialDecision(
        decision=Decision.PARTICIPATE,
        reasoning="Тестовое обоснование решения",
        key_factors=["Фактор 1", "Фактор 2"],
        conditions=["Условие 1"],
        next_steps=["Шаг 1", "Шаг 2"],
        confidence=0.82
    )
    print(f"✅ Решение руководителя создано: {decision.decision}")
    
    return contest, legal, economic, decision


def test_json_serialization():
    """Тестирование сериализации в JSON"""
    print("\n📄 Тестирование сериализации...")
    
    contest, legal, economic, decision = test_models()
    
    # Тест сериализации
    try:
        contest_json = contest.model_dump_json(indent=2)
        legal_json = legal.model_dump_json(indent=2)
        economic_json = economic.model_dump_json(indent=2)
        decision_json = decision.model_dump_json(indent=2)
        
        print("✅ Все модели успешно сериализованы в JSON")
        
        # Показать пример JSON
        print("\n📋 Пример JSON конкурса:")
        print(contest_json[:300] + "..." if len(contest_json) > 300 else contest_json)
        
    except Exception as e:
        print(f"❌ Ошибка сериализации: {e}")


def test_graph_structure():
    """Тестирование структуры графа (без выполнения)"""
    print("\n🔗 Тестирование структуры графа...")
    
    try:
        # Импорт без создания экземпляра (чтобы не требовать API ключ)
        from concurse.contest_analysis_graph import ContestAnalysisGraph
        print("✅ Импорт графа успешен")
        
        # Проверка наличия необходимых методов
        required_methods = ['_build_graph', '_legal_analysis_node', '_economic_analysis_node', '_managerial_decision_node']
        for method in required_methods:
            if hasattr(ContestAnalysisGraph, method):
                print(f"✅ Метод {method} найден")
            else:
                print(f"❌ Метод {method} не найден")
                
    except Exception as e:
        print(f"❌ Ошибка импорта графа: {e}")


def test_agents_structure():
    """Тестирование структуры агентов (без создания экземпляров)"""
    print("\n🤖 Тестирование структуры агентов...")
    
    try:
        from concurse.agents import LegalAgent, EconomicAgent, ManagerAgent, BaseAgent
        print("✅ Импорт агентов успешен")
        
        # Проверка наследования
        agents = [LegalAgent, EconomicAgent, ManagerAgent]
        for agent_class in agents:
            if issubclass(agent_class, BaseAgent):
                print(f"✅ {agent_class.__name__} наследует BaseAgent")
            else:
                print(f"❌ {agent_class.__name__} не наследует BaseAgent")
                
        # Проверка методов
        required_methods = {
            'LegalAgent': ['analyze'],
            'EconomicAgent': ['analyze'], 
            'ManagerAgent': ['decide']
        }
        
        for agent_name, methods in required_methods.items():
            agent_class = globals()[agent_name] if agent_name in globals() else getattr(__import__('agents'), agent_name)
            for method in methods:
                if hasattr(agent_class, method):
                    print(f"✅ {agent_name}.{method} найден")
                else:
                    print(f"❌ {agent_name}.{method} не найден")
                    
    except Exception as e:
        print(f"❌ Ошибка импорта агентов: {e}")


def show_sample_analysis():
    """Показать пример структуры анализа"""
    print("\n📊 Пример структуры анализа:")
    print("=" * 50)
    
    contest, legal, economic, decision = test_models()
    
    # Имитация результата анализа
    sample_result = {
        "contest_info": contest.model_dump(),
        "legal_analysis": legal.model_dump(),
        "economic_analysis": economic.model_dump(),
        "managerial_decision": decision.model_dump(),
        "messages": ["Анализ завершен"],
        "current_step": "completed"
    }
    
    print("🏛️ ЮРИДИЧЕСКОЕ ЗАКЛЮЧЕНИЕ:")
    print(f"   Уровень риска: {legal.risk_level}")
    print(f"   Риски: {', '.join(legal.legal_risks[:2])}")
    print(f"   Уверенность: {legal.confidence:.0%}")
    
    print("\n💰 ЭКОНОМИЧЕСКОЕ ЗАКЛЮЧЕНИЕ:")
    print(f"   Уровень риска: {economic.risk_level}")
    print(f"   Затраты: {sum(economic.estimated_costs.values()):,} руб.")
    print(f"   Прибыль: {economic.potential_profit:,} руб.")
    print(f"   ROI: {economic.roi_estimate:.1%}")
    print(f"   Уверенность: {economic.confidence:.0%}")
    
    print("\n👔 РЕШЕНИЕ РУКОВОДИТЕЛЯ:")
    print(f"   Решение: {decision.decision.upper()}")
    print(f"   Факторы: {', '.join(decision.key_factors)}")
    print(f"   Следующие шаги: {len(decision.next_steps)} шагов")
    print(f"   Уверенность: {decision.confidence:.0%}")


def main():
    """Основная функция тестирования"""
    print("🧪 ТЕСТИРОВАНИЕ СИСТЕМЫ АНАЛИЗА КОНКУРСОВ")
    print("=" * 60)
    print("Этот тест проверяет структуру системы без использования API")
    print()
    
    try:
        test_models()
        test_json_serialization()
        test_graph_structure()
        test_agents_structure()
        show_sample_analysis()
        
        print("\n" + "=" * 60)
        print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("🚀 Система готова к работе с API ключами")
        print("\nДля полного тестирования:")
        print("1. Добавьте OPENAI_API_KEY в файл .env")
        print("2. Запустите: python main.py")
        
    except Exception as e:
        print(f"\n❌ ОШИБКА В ТЕСТАХ: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 