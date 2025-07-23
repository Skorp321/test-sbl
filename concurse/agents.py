import os
from typing import Dict, Any
from langchain_mistralai import ChatMistralAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from concurse.models import (
    ContestAnalysisState, 
    LegalAnalysis, 
    EconomicAnalysis, 
    ManagerialDecision,
    RiskLevel,
    Decision
)
import json


class BaseAgent:
    """Базовый класс для всех агентов"""
    
    def __init__(self, model_name: str = "mistral-large-latest"):
        self.llm = ChatMistralAI(
            mistral_api_key=os.getenv("MISTRAL_API_KEY"),
            model=model_name,
            temperature=0.1
        )

    def _parse_json_response(self, response: str, expected_keys: list) -> Dict[str, Any]:
        """Парсинг JSON ответа от LLM"""
        try:
            # Попытка найти JSON в ответе
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                parsed = json.loads(json_str)
                return parsed
        except json.JSONDecodeError:
            pass
        
        # Если не удалось распарсить JSON, создаем базовую структуру
        result = {}
        for key in expected_keys:
            if key in response.lower():
                result[key] = "Информация извлечена из текста"
        return result


class LegalAgent(BaseAgent):
    """Агент-юрист для анализа юридических аспектов конкурса"""
    
    def __init__(self):
        super().__init__()
        self.system_prompt = """
        Вы - опытный юрист, специализирующийся на анализе государственных и коммерческих конкурсов.
        Ваша задача - провести тщательный юридический анализ конкурса и оценить риски участия.
        
        При анализе обращайте внимание на:
        1. Соответствие требованиям законодательства
        2. Юридические риски и их уровень
        3. Необходимые документы и их доступность
        4. Возможные правовые последствия
        5. Соблюдение процедур конкурса
        
        Ответ должен быть структурированным и содержать конкретные рекомендации.
        """
    
    def analyze(self, state: ContestAnalysisState) -> LegalAnalysis:
        """Проведение юридического анализа"""
        
        contest_info = state.contest_info
        
        prompt = f"""
        Проведите юридический анализ следующего конкурса:
        
        Название: {contest_info.title}
        Описание: {contest_info.description}
        Бюджет: {contest_info.budget if contest_info.budget else 'Не указан'}
        Срок подачи: {contest_info.deadline if contest_info.deadline else 'Не указан'}
        Требования: {', '.join(contest_info.requirements) if contest_info.requirements else 'Не указаны'}
        Документы: {', '.join(contest_info.documentation) if contest_info.documentation else 'Не указаны'}
        Критерии оценки: {', '.join(contest_info.evaluation_criteria) if contest_info.evaluation_criteria else 'Не указаны'}
        
        Предоставьте анализ в следующем JSON формате:
        {{
            "legal_risks": ["риск1", "риск2", ...],
            "compliance_issues": ["вопрос1", "вопрос2", ...],
            "required_documents": ["документ1", "документ2", ...],
            "risk_level": "низкий/средний/высокий",
            "recommendation": "подробная рекомендация",
            "confidence": 0.85
        }}
        """
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        try:
            # Попытка распарсить JSON ответ
            response_text = response.content
            parsed_data = self._parse_json_response(
                response_text, 
                ["legal_risks", "compliance_issues", "required_documents", "risk_level", "recommendation", "confidence"]
            )
            
            # Создание объекта LegalAnalysis с обработкой ошибок
            legal_analysis = LegalAnalysis(
                legal_risks=parsed_data.get("legal_risks", ["Требуется дополнительный анализ"]),
                compliance_issues=parsed_data.get("compliance_issues", ["Требуется проверка соответствия"]),
                required_documents=parsed_data.get("required_documents", ["Стандартный пакет документов"]),
                risk_level=RiskLevel.MEDIUM if parsed_data.get("risk_level", "средний").lower() not in ["низкий", "высокий"] else RiskLevel(parsed_data.get("risk_level", "средний").lower()),
                recommendation=parsed_data.get("recommendation", response_text),
                confidence=float(parsed_data.get("confidence", 0.7))
            )
            
        except Exception as e:
            # Fallback анализ на основе текста ответа
            legal_analysis = LegalAnalysis(
                legal_risks=["Требуется детальный анализ документации конкурса"],
                compliance_issues=["Необходима проверка соответствия всем требованиям"],
                required_documents=contest_info.documentation if contest_info.documentation else ["Стандартный пакет документов"],
                risk_level=RiskLevel.MEDIUM,
                recommendation=response.content,
                confidence=0.7
            )
        
        return legal_analysis


class EconomicAgent(BaseAgent):
    """Агент-экономист для анализа экономической целесообразности"""
    
    def __init__(self):
        super().__init__()
        self.system_prompt = """
        Вы - опытный экономист и финансовый аналитик, специализирующийся на оценке инвестиционных проектов и конкурсов.
        Ваша задача - провести экономический анализ участия в конкурсе.
        
        При анализе учитывайте:
        1. Затраты на участие (подготовка документов, время, ресурсы)
        2. Потенциальную прибыль и ROI
        3. Рыночные возможности
        4. Экономические риски
        5. Альтернативные варианты использования ресурсов
        
        Предоставьте количественные оценки где это возможно.
        """
    
    def analyze(self, state: ContestAnalysisState) -> EconomicAnalysis:
        """Проведение экономического анализа"""
        
        contest_info = state.contest_info
        legal_analysis = state.legal_analysis
        
        legal_context = ""
        if legal_analysis:
            legal_context = f"""
            Юридический анализ показал:
            - Уровень риска: {legal_analysis.risk_level}
            - Основные риски: {', '.join(legal_analysis.legal_risks[:3])}
            - Рекомендация юриста: {legal_analysis.recommendation[:200]}...
            """
        
        prompt = f"""
        Проведите экономический анализ участия в конкурсе:
        
        Название: {contest_info.title}
        Описание: {contest_info.description}
        Бюджет: {contest_info.budget if contest_info.budget else 'Не указан'}
        Срок подачи: {contest_info.deadline if contest_info.deadline else 'Не указан'}
        
        {legal_context}
        
        Предоставьте анализ в следующем JSON формате:
        {{
            "estimated_costs": {{
                "preparation": 50000,
                "documentation": 25000,
                "participation": 15000
            }},
            "potential_profit": 500000,
            "roi_estimate": 0.25,
            "market_opportunities": ["возможность1", "возможность2", ...],
            "economic_risks": ["риск1", "риск2", ...],
            "risk_level": "низкий/средний/высокий",
            "recommendation": "подробная рекомендация",
            "confidence": 0.80
        }}
        """
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        try:
            response_text = response.content
            parsed_data = self._parse_json_response(
                response_text,
                ["estimated_costs", "potential_profit", "roi_estimate", "market_opportunities", "economic_risks", "risk_level", "recommendation", "confidence"]
            )
            
            economic_analysis = EconomicAnalysis(
                estimated_costs=parsed_data.get("estimated_costs", {"preparation": 50000, "total": 100000}),
                potential_profit=parsed_data.get("potential_profit"),
                roi_estimate=parsed_data.get("roi_estimate"),
                market_opportunities=parsed_data.get("market_opportunities", ["Требуется дополнительный анализ рынка"]),
                economic_risks=parsed_data.get("economic_risks", ["Требуется оценка экономических рисков"]),
                risk_level=RiskLevel.MEDIUM if parsed_data.get("risk_level", "средний").lower() not in ["низкий", "высокий"] else RiskLevel(parsed_data.get("risk_level", "средний").lower()),
                recommendation=parsed_data.get("recommendation", response_text),
                confidence=float(parsed_data.get("confidence", 0.7))
            )
            
        except Exception as e:
            economic_analysis = EconomicAnalysis(
                estimated_costs={"preparation": 50000, "documentation": 25000, "total": 75000},
                potential_profit=contest_info.budget * 0.15 if contest_info.budget else None,
                roi_estimate=None,
                market_opportunities=["Требуется детальный анализ рыночных возможностей"],
                economic_risks=["Требуется оценка экономических рисков"],
                risk_level=RiskLevel.MEDIUM,
                recommendation=response.content,
                confidence=0.7
            )
        
        return economic_analysis


class ManagerAgent(BaseAgent):
    """Агент-руководитель для принятия итогового решения"""
    
    def __init__(self):
        super().__init__()
        self.system_prompt = """
        Вы - опытный руководитель компании, принимающий стратегические решения на основе анализа экспертов.
        Ваша задача - принять взвешенное решение об участии в конкурсе, основываясь на заключениях юриста и экономиста.
        
        При принятии решения учитывайте:
        1. Соотношение рисков и возможностей
        2. Стратегические цели компании
        3. Доступные ресурсы
        4. Альтернативные возможности
        5. Долгосрочные перспективы
        
        Решение должно быть обоснованным и содержать конкретные следующие шаги.
        """
    
    def decide(self, state: ContestAnalysisState) -> ManagerialDecision:
        """Принятие управленческого решения"""
        
        contest_info = state.contest_info
        legal_analysis = state.legal_analysis
        economic_analysis = state.economic_analysis
        
        if not legal_analysis or not economic_analysis:
            raise ValueError("Для принятия решения необходимы заключения юриста и экономиста")
        
        prompt = f"""
        Примите решение об участии в конкурсе на основе следующих данных:
        
        ИНФОРМАЦИЯ О КОНКУРСЕ:
        Название: {contest_info.title}
        Описание: {contest_info.description}
        Бюджет: {contest_info.budget if contest_info.budget else 'Не указан'}
        
        ЮРИДИЧЕСКОЕ ЗАКЛЮЧЕНИЕ:
        Уровень риска: {legal_analysis.risk_level}
        Основные риски: {', '.join(legal_analysis.legal_risks)}
        Рекомендация: {legal_analysis.recommendation}
        Уверенность: {legal_analysis.confidence}
        
        ЭКОНОМИЧЕСКОЕ ЗАКЛЮЧЕНИЕ:
        Уровень риска: {economic_analysis.risk_level}
        Ожидаемые затраты: {economic_analysis.estimated_costs}
        Потенциальная прибыль: {economic_analysis.potential_profit}
        ROI: {economic_analysis.roi_estimate}
        Рекомендация: {economic_analysis.recommendation}
        Уверенность: {economic_analysis.confidence}
        
        Предоставьте решение в следующем JSON формате:
        {{
            "decision": "участвовать/не участвовать/требуется дополнительная информация",
            "reasoning": "подробное обоснование решения",
            "key_factors": ["фактор1", "фактор2", ...],
            "conditions": ["условие1", "условие2", ...],
            "next_steps": ["шаг1", "шаг2", ...],
            "confidence": 0.85
        }}
        """
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        try:
            response_text = response.content
            parsed_data = self._parse_json_response(
                response_text,
                ["decision", "reasoning", "key_factors", "conditions", "next_steps", "confidence"]
            )
            
            # Определение решения
            decision_text = parsed_data.get("decision", "требуется дополнительная информация").lower()
            if "участвовать" in decision_text and "не участвовать" not in decision_text:
                decision = Decision.PARTICIPATE
            elif "не участвовать" in decision_text:
                decision = Decision.NOT_PARTICIPATE
            else:
                decision = Decision.NEED_MORE_INFO
            
            managerial_decision = ManagerialDecision(
                decision=decision,
                reasoning=parsed_data.get("reasoning", response_text),
                key_factors=parsed_data.get("key_factors", ["Анализ экспертных заключений"]),
                conditions=parsed_data.get("conditions", []),
                next_steps=parsed_data.get("next_steps", ["Требуется определение следующих шагов"]),
                confidence=float(parsed_data.get("confidence", 0.7))
            )
            
        except Exception as e:
            # Fallback решение
            avg_confidence = (legal_analysis.confidence + economic_analysis.confidence) / 2
            
            if legal_analysis.risk_level == RiskLevel.LOW and economic_analysis.risk_level == RiskLevel.LOW:
                decision = Decision.PARTICIPATE
            elif legal_analysis.risk_level == RiskLevel.HIGH or economic_analysis.risk_level == RiskLevel.HIGH:
                decision = Decision.NOT_PARTICIPATE
            else:
                decision = Decision.NEED_MORE_INFO
            
            managerial_decision = ManagerialDecision(
                decision=decision,
                reasoning=response.content,
                key_factors=[f"Юридический риск: {legal_analysis.risk_level}", f"Экономический риск: {economic_analysis.risk_level}"],
                conditions=[],
                next_steps=["Требуется детальный анализ"],
                confidence=avg_confidence
            )
        
        return managerial_decision 