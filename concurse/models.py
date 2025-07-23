from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class RiskLevel(str, Enum):
    LOW = "низкий"
    MEDIUM = "средний"
    HIGH = "высокий"


class Decision(str, Enum):
    PARTICIPATE = "участвовать"
    NOT_PARTICIPATE = "не участвовать"
    NEED_MORE_INFO = "требуется дополнительная информация"


class ContestInfo(BaseModel):
    """Информация о конкурсе для анализа"""
    title: str = Field(description="Название конкурса")
    description: str = Field(description="Описание конкурса")
    budget: Optional[float] = Field(description="Бюджет конкурса", default=None)
    deadline: Optional[str] = Field(description="Срок подачи заявки", default=None)
    requirements: List[str] = Field(description="Требования к участникам", default_factory=list)
    documentation: List[str] = Field(description="Необходимые документы", default_factory=list)
    evaluation_criteria: List[str] = Field(description="Критерии оценки", default_factory=list)
    additional_info: Dict[str, Any] = Field(description="Дополнительная информация", default_factory=dict)


class LegalAnalysis(BaseModel):
    """Юридическое заключение"""
    legal_risks: List[str] = Field(description="Выявленные юридические риски")
    compliance_issues: List[str] = Field(description="Вопросы соответствия требованиям")
    required_documents: List[str] = Field(description="Необходимые документы")
    risk_level: RiskLevel = Field(description="Уровень юридического риска")
    recommendation: str = Field(description="Рекомендация юриста")
    confidence: float = Field(description="Уверенность в анализе (0-1)", ge=0, le=1)


class EconomicAnalysis(BaseModel):
    """Экономическое заключение"""
    estimated_costs: Dict[str, float] = Field(description="Оценочные затраты")
    potential_profit: Optional[float] = Field(description="Потенциальная прибыль", default=None)
    roi_estimate: Optional[float] = Field(description="Оценка ROI", default=None)
    market_opportunities: List[str] = Field(description="Рыночные возможности")
    economic_risks: List[str] = Field(description="Экономические риски")
    risk_level: RiskLevel = Field(description="Уровень экономического риска")
    recommendation: str = Field(description="Рекомендация экономиста")
    confidence: float = Field(description="Уверенность в анализе (0-1)", ge=0, le=1)


class ManagerialDecision(BaseModel):
    """Решение руководителя"""
    decision: Decision = Field(description="Итоговое решение")
    reasoning: str = Field(description="Обоснование решения")
    key_factors: List[str] = Field(description="Ключевые факторы решения")
    conditions: List[str] = Field(description="Условия участия (если применимо)", default_factory=list)
    next_steps: List[str] = Field(description="Следующие шаги")
    confidence: float = Field(description="Уверенность в решении (0-1)", ge=0, le=1)


class ContestAnalysisState(BaseModel):
    """Состояние анализа конкурса"""
    contest_info: ContestInfo
    legal_analysis: Optional[LegalAnalysis] = None
    economic_analysis: Optional[EconomicAnalysis] = None
    managerial_decision: Optional[ManagerialDecision] = None
    messages: List[str] = Field(default_factory=list)
    current_step: str = "initial" 