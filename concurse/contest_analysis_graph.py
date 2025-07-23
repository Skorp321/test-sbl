from typing import Dict, Any, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from concurse.models import ContestAnalysisState, ContestInfo
from concurse.agents import LegalAgent, EconomicAgent, ManagerAgent
import json


class ContestAnalysisGraph:
    """Граф для анализа конкурсов с тремя агентами"""
    
    def __init__(self):
        self.legal_agent = LegalAgent()
        self.economic_agent = EconomicAgent()
        self.manager_agent = ManagerAgent()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Построение графа анализа"""
        
        # Определяем состояние графа
        class GraphState(TypedDict):
            contest_info: Dict[str, Any]
            legal_analysis: Dict[str, Any]
            economic_analysis: Dict[str, Any]
            managerial_decision: Dict[str, Any]
            messages: list
            current_step: str
        
        # Сохраняем класс состояния для использования в методах
        self.GraphState = GraphState
        
        # Создаем граф
        workflow = StateGraph(GraphState)
        
        # Добавляем узлы
        workflow.add_node("legal_agent", self._legal_analysis_node)
        workflow.add_node("economic_agent", self._economic_analysis_node)
        workflow.add_node("manager_agent", self._managerial_decision_node)
        
        # Определяем маршруты
        workflow.set_entry_point("legal_agent")
        workflow.add_edge("legal_agent", "economic_agent")
        workflow.add_edge("economic_agent", "manager_agent")
        workflow.add_edge("manager_agent", END)
        
        # Компилируем граф с checkpointer для сохранения состояния
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    def _legal_analysis_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Узел юридического анализа"""
        print("🏛️ Начинаю юридический анализ...")
        
        # Создаем объект состояния для агента
        contest_state = ContestAnalysisState(
            contest_info=ContestInfo(**state["contest_info"]),
            current_step="legal_analysis"
        )
        
        # Проводим анализ
        legal_analysis = self.legal_agent.analyze(contest_state)
        
        # Обновляем состояние
        state["legal_analysis"] = legal_analysis.model_dump()
        state["current_step"] = "legal_analysis_completed"
        state["messages"].append("Юридический анализ завершен")
        
        print(f"✅ Юридический анализ завершен. Уровень риска: {legal_analysis.risk_level}")
        return state
    
    def _economic_analysis_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Узел экономического анализа"""
        print("💰 Начинаю экономический анализ...")
        
        # Создаем объект состояния для агента
        contest_state = ContestAnalysisState(
            contest_info=ContestInfo(**state["contest_info"]),
            legal_analysis=state.get("legal_analysis"),
            current_step="economic_analysis"
        )
        
        # Проводим анализ
        economic_analysis = self.economic_agent.analyze(contest_state)
        
        # Обновляем состояние
        state["economic_analysis"] = economic_analysis.model_dump()
        state["current_step"] = "economic_analysis_completed"
        state["messages"].append("Экономический анализ завершен")
        
        print(f"✅ Экономический анализ завершен. Уровень риска: {economic_analysis.risk_level}")
        return state
    
    def _managerial_decision_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Узел принятия управленческого решения"""
        print("👔 Принимаю управленческое решение...")
        
        # Создаем объект состояния для агента
        from concurse.models import LegalAnalysis, EconomicAnalysis
        
        contest_state = ContestAnalysisState(
            contest_info=ContestInfo(**state["contest_info"]),
            legal_analysis=LegalAnalysis(**state["legal_analysis"]) if state.get("legal_analysis") else None,
            economic_analysis=EconomicAnalysis(**state["economic_analysis"]) if state.get("economic_analysis") else None,
            current_step="managerial_decision"
        )
        
        # Принимаем решение
        managerial_decision = self.manager_agent.decide(contest_state)
        
        # Обновляем состояние
        state["managerial_decision"] = managerial_decision.model_dump()
        state["current_step"] = "completed"
        state["messages"].append("Управленческое решение принято")
        
        print(f"✅ Решение принято: {managerial_decision.decision}")
        return state
    
    def analyze_contest(self, contest_info: ContestInfo, thread_id: str = "default") -> Dict[str, Any]:
        """Анализ конкурса"""
        
        # Начальное состояние
        initial_state = {
            "contest_info": contest_info.model_dump(),
            "legal_analysis": {},
            "economic_analysis": {},
            "managerial_decision": {},
            "messages": ["Начинаю анализ конкурса"],
            "current_step": "initial"
        }
        
        # Конфигурация для выполнения
        config = {"configurable": {"thread_id": thread_id}}
        
        print(f"🚀 Начинаю анализ конкурса: {contest_info.title}")
        
        # Выполняем граф
        result = self.graph.invoke(initial_state, config)
        
        return result
    
    def get_analysis_summary(self, result: Dict[str, Any]) -> str:
        """Получение краткого резюме анализа"""
        
        if not result.get("managerial_decision"):
            return "Анализ не завершен"
        
        legal = result.get("legal_analysis", {})
        economic = result.get("economic_analysis", {})
        decision = result.get("managerial_decision", {})
        
        summary = f"""
📊 РЕЗЮМЕ АНАЛИЗА КОНКУРСА

🏛️ ЮРИДИЧЕСКОЕ ЗАКЛЮЧЕНИЕ:
• Уровень риска: {legal.get('risk_level', 'Не определен').replace('RiskLevel.', '').replace('LOW', 'низкий').replace('MEDIUM', 'средний').replace('HIGH', 'высокий')}
• Основные риски: {', '.join(legal.get('legal_risks', [])[:2])}
• Уверенность: {legal.get('confidence', 0):.0%}

💰 ЭКОНОМИЧЕСКОЕ ЗАКЛЮЧЕНИЕ:
• Уровень риска: {economic.get('risk_level', 'Не определен').replace('RiskLevel.', '').replace('LOW', 'низкий').replace('MEDIUM', 'средний').replace('HIGH', 'высокий')}
• Ожидаемые затраты: {sum(economic.get('estimated_costs', {}).values()) if isinstance(economic.get('estimated_costs', {}), dict) else economic.get('estimated_costs', 0):,.0f} руб.
• Потенциальная прибыль: {f"{economic.get('potential_profit', 0):,.0f} руб." if economic.get('potential_profit') else 'Не определена'}
• Уверенность: {economic.get('confidence', 0):.0%}

👔 РЕШЕНИЕ РУКОВОДИТЕЛЯ:
• Решение: {decision.get('decision', 'Не принято').replace('Decision.', '').replace('PARTICIPATE', 'УЧАСТВОВАТЬ').replace('NOT_PARTICIPATE', 'НЕ УЧАСТВОВАТЬ').replace('NEED_MORE_INFO', 'ТРЕБУЕТСЯ ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ')}
• Уверенность: {decision.get('confidence', 0):.0%}
• Ключевые факторы: {', '.join(decision.get('key_factors', [])[:2])}

📋 СЛЕДУЮЩИЕ ШАГИ:
{chr(10).join([f"• {step}" for step in decision.get('next_steps', [])])}
        """
        
        return summary.strip()


def create_sample_contest() -> ContestInfo:
    """Создание примера конкурса для тестирования"""
    return ContestInfo(
        title="Разработка информационной системы для государственного учреждения",
        description="Конкурс на разработку и внедрение информационной системы управления документооборотом для министерства. Система должна обеспечивать электронный документооборот, интеграцию с существующими системами и соответствие требованиям информационной безопасности.",
        budget=5000000.0,
        deadline="2024-03-15",
        requirements=[
            "Опыт разработки государственных информационных систем не менее 3 лет",
            "Наличие сертификатов соответствия ФСТЭК",
            "Штат разработчиков не менее 10 человек",
            "Наличие службы технической поддержки"
        ],
        documentation=[
            "Техническое задание",
            "Коммерческое предложение",
            "Справки о выполненных работах",
            "Сертификаты соответствия",
            "Финансовые гарантии"
        ],
        evaluation_criteria=[
            "Соответствие техническому заданию (40%)",
            "Стоимость работ (30%)",
            "Опыт и квалификация (20%)",
            "Сроки выполнения (10%)"
        ],
        additional_info={
            "contract_duration": "12 месяцев",
            "warranty_period": "24 месяца",
            "payment_terms": "поэтапная оплата",
            "penalties": "0.1% за каждый день просрочки"
        }
    ) 