import streamlit as st
import os
import sys
from datetime import datetime, date
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
import json

# Добавляем текущую директорию в путь для импорта модулей
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from concurse.contest_analysis_graph import ContestAnalysisGraph, create_sample_contest
from concurse.models import ContestInfo, RiskLevel, Decision

# Настройка страницы
st.set_page_config(
    page_title="AI-Агент Анализа Конкурсов",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Загрузка переменных окружения
load_dotenv()

# Кастомные стили
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .agent-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: white;
        display: inline-block;
        margin: 0.2rem;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #FF9800, #F57C00);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: white;
        display: inline-block;
        margin: 0.2rem;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #F44336, #D32F2F);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: white;
        display: inline-block;
        margin: 0.2rem;
    }
    
    .decision-participate {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        padding: 1rem 2rem;
        border-radius: 25px;
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    
    .decision-not-participate {
        background: linear-gradient(135deg, #F44336, #D32F2F);
        padding: 1rem 2rem;
        border-radius: 25px;
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    
    .decision-need-info {
        background: linear-gradient(135deg, #FF9800, #F57C00);
        padding: 1rem 2rem;
        border-radius: 25px;
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def setup_environment():
    """Настройка окружения"""
    if not os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
    
    if not os.getenv("MISTRAL_API_KEY"):
        st.error("❌ Не найден MISTRAL_API_KEY. Добавьте его в переменные окружения или файл .env")
        st.info("Создайте файл .env и добавьте: MISTRAL_API_KEY=your_mistral_api_key_here")
        st.stop()

def get_risk_level_style(risk_level):
    """Получение стиля для уровня риска"""
    if isinstance(risk_level, str):
        risk_level = risk_level.lower()
    else:
        risk_level = str(risk_level).lower()
    
    if "низкий" in risk_level or "low" in risk_level:
        return "risk-low"
    elif "высокий" in risk_level or "high" in risk_level:
        return "risk-high"
    else:
        return "risk-medium"

def get_decision_style(decision):
    """Получение стиля для решения"""
    if isinstance(decision, str):
        decision_text = decision.lower()
    else:
        decision_text = str(decision).lower()
    
    if "участвовать" in decision_text and "не участвовать" not in decision_text:
        return "decision-participate"
    elif "не участвовать" in decision_text:
        return "decision-not-participate"
    else:
        return "decision-need-info"

def format_currency(amount):
    """Форматирование валюты"""
    if amount is None:
        return "Не указано"
    return f"{amount:,.0f} ₽".replace(",", " ")

def create_risk_chart(legal_risk, economic_risk):
    """Создание диаграммы рисков"""
    risk_mapping = {"низкий": 1, "средний": 2, "высокий": 3}
    
    legal_val = risk_mapping.get(str(legal_risk).lower(), 2)
    economic_val = risk_mapping.get(str(economic_risk).lower(), 2)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=[legal_val, economic_val],
        theta=['Юридический риск', 'Экономический риск'],
        fill='toself',
        name='Уровень рисков',
        line_color='rgb(102, 126, 234)',
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 3],
                tickvals=[1, 2, 3],
                ticktext=['Низкий', 'Средний', 'Высокий']
            )
        ),
        showlegend=False,
        title="Анализ рисков",
        height=400
    )
    
    return fig

def create_financial_chart(costs, profit):
    """Создание финансовой диаграммы"""
    if not costs or not profit:
        return None
    
    total_costs = sum(costs.values()) if isinstance(costs, dict) else costs
    
    fig = go.Figure(data=[
        go.Bar(name='Затраты', x=['Финансы'], y=[total_costs], marker_color='#FF6B6B'),
        go.Bar(name='Прибыль', x=['Финансы'], y=[profit], marker_color='#4ECDC4')
    ])
    
    fig.update_layout(
        title='Финансовый анализ',
        yaxis_title='Сумма (₽)',
        barmode='group',
        height=400
    )
    
    return fig

def display_contest_form():
    """Форма для ввода данных о конкурсе"""
    st.markdown('<div class="main-header">🤖 AI-Агент Анализа Конкурсов</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown("### 📝 Информация о конкурсе")
        
        col1, col2 = st.columns(2)
        
        with col1:
            title = st.text_input(
                "Название конкурса *",
                placeholder="Введите название конкурса",
                help="Краткое и понятное название конкурса"
            )
            
            budget = st.number_input(
                "Бюджет конкурса (₽)",
                min_value=0.0,
                step=10000.0,
                format="%.0f",
                help="Общий бюджет конкурса в рублях"
            )
            
            deadline = st.date_input(
                "Срок подачи заявки",
                min_value=date.today(),
                help="Последний день подачи заявки"
            )
        
        with col2:
            description = st.text_area(
                "Описание конкурса *",
                height=100,
                placeholder="Подробное описание конкурса, его целей и задач",
                help="Детальное описание конкурса"
            )
            
            requirements = st.text_area(
                "Требования к участникам",
                height=100,
                placeholder="Каждое требование с новой строки",
                help="Основные требования к участникам конкурса"
            )
        
        documentation = st.text_area(
            "Необходимые документы",
            placeholder="Каждый документ с новой строки",
            help="Список документов, которые нужно подготовить"
        )
        
        evaluation_criteria = st.text_area(
            "Критерии оценки",
            placeholder="Каждый критерий с новой строки",
            help="Критерии, по которым будут оценивать заявки"
        )
    
    return {
        'title': title,
        'description': description,
        'budget': budget if budget > 0 else None,
        'deadline': deadline.strftime("%Y-%m-%d") if deadline else None,
        'requirements': [req.strip() for req in requirements.split('\n') if req.strip()] if requirements else [],
        'documentation': [doc.strip() for doc in documentation.split('\n') if doc.strip()] if documentation else [],
        'evaluation_criteria': [crit.strip() for crit in evaluation_criteria.split('\n') if crit.strip()] if evaluation_criteria else []
    }

def display_analysis_results(result):
    """Отображение результатов анализа"""
    st.markdown("## 📊 Результаты анализа")
    
    # Общая информация о конкурсе
    contest = result.get("contest_info", {})
    st.markdown("### 📋 Информация о конкурсе")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Название", contest.get('title', 'Не указано'))
    with col2:
        budget = contest.get('budget')
        st.metric("Бюджет", format_currency(budget))
    with col3:
        st.metric("Срок подачи", contest.get('deadline', 'Не указан'))
    
    # Анализ агентов
    legal = result.get("legal_analysis", {})
    economic = result.get("economic_analysis", {})
    decision = result.get("managerial_decision", {})
    
    # Создание вкладок для детального анализа
    tab1, tab2, tab3, tab4 = st.tabs(["🏛️ Юридический анализ", "💰 Экономический анализ", "👔 Решение руководителя", "📈 Визуализация"])
    
    with tab1:
        if legal:
            st.markdown("### 🏛️ Заключение юриста")
            
            col1, col2 = st.columns(2)
            with col1:
                risk_level = str(legal.get('risk_level', 'средний'))
                risk_class = get_risk_level_style(risk_level)
                st.markdown(f'<div class="{risk_class}">Уровень риска: {risk_level.upper()}</div>', unsafe_allow_html=True)
                
                confidence = legal.get('confidence', 0)
                st.metric("Уверенность анализа", f"{confidence:.0%}")
            
            with col2:
                st.markdown("**Основные юридические риски:**")
                risks = legal.get('legal_risks', [])
                for i, risk in enumerate(risks[:5], 1):
                    st.write(f"{i}. {risk}")
            
            st.markdown("**Необходимые документы:**")
            docs = legal.get('required_documents', [])
            for i, doc in enumerate(docs[:5], 1):
                st.write(f"📄 {doc}")
            
            st.markdown("**Рекомендация юриста:**")
            st.info(legal.get('recommendation', 'Рекомендация не предоставлена'))
    
    with tab2:
        if economic:
            st.markdown("### 💰 Заключение экономиста")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                risk_level = str(economic.get('risk_level', 'средний'))
                risk_class = get_risk_level_style(risk_level)
                st.markdown(f'<div class="{risk_class}">Уровень риска: {risk_level.upper()}</div>', unsafe_allow_html=True)
                
                confidence = economic.get('confidence', 0)
                st.metric("Уверенность анализа", f"{confidence:.0%}")
            
            with col2:
                costs = economic.get('estimated_costs', {})
                total_cost = sum(costs.values()) if isinstance(costs, dict) else costs
                st.metric("Ожидаемые затраты", format_currency(total_cost))
                
                profit = economic.get('potential_profit')
                st.metric("Потенциальная прибыль", format_currency(profit))
            
            with col3:
                roi = economic.get('roi_estimate')
                if roi:
                    st.metric("Ожидаемый ROI", f"{roi:.1%}")
                
                if isinstance(costs, dict) and profit:
                    net_profit = profit - sum(costs.values())
                    st.metric("Чистая прибыль", format_currency(net_profit))
            
            if isinstance(costs, dict):
                st.markdown("**Детализация затрат:**")
                for category, cost in costs.items():
                    st.write(f"• {category}: {format_currency(cost)}")
            
            opportunities = economic.get('market_opportunities', [])
            if opportunities:
                st.markdown("**Рыночные возможности:**")
                for i, opp in enumerate(opportunities[:5], 1):
                    st.write(f"{i}. {opp}")
            
            st.markdown("**Рекомендация экономиста:**")
            st.info(economic.get('recommendation', 'Рекомендация не предоставлена'))
    
    with tab3:
        if decision:
            st.markdown("### 👔 Решение руководителя")
            
            # Итоговое решение
            decision_text = str(decision.get('decision', 'Не принято'))
            decision_class = get_decision_style(decision_text)
            
            if "участвовать" in decision_text.lower() and "не участвовать" not in decision_text.lower():
                decision_display = "✅ УЧАСТВОВАТЬ"
            elif "не участвовать" in decision_text.lower():
                decision_display = "❌ НЕ УЧАСТВОВАТЬ"
            else:
                decision_display = "⚠️ ТРЕБУЕТСЯ ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ"
            
            st.markdown(f'<div class="{decision_class}">{decision_display}</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                confidence = decision.get('confidence', 0)
                st.metric("Уверенность в решении", f"{confidence:.0%}")
                
                factors = decision.get('key_factors', [])
                if factors:
                    st.markdown("**Ключевые факторы решения:**")
                    for i, factor in enumerate(factors, 1):
                        st.write(f"{i}. {factor}")
            
            with col2:
                conditions = decision.get('conditions', [])
                if conditions:
                    st.markdown("**Условия участия:**")
                    for i, condition in enumerate(conditions, 1):
                        st.write(f"{i}. {condition}")
                
                steps = decision.get('next_steps', [])
                if steps:
                    st.markdown("**Следующие шаги:**")
                    for i, step in enumerate(steps, 1):
                        st.write(f"🔸 {step}")
            
            st.markdown("**Обоснование решения:**")
            st.success(decision.get('reasoning', 'Обоснование не предоставлено'))
    
    with tab4:
        st.markdown("### 📈 Визуализация результатов")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if legal and economic:
                legal_risk = str(legal.get('risk_level', 'средний'))
                economic_risk = str(economic.get('risk_level', 'средний'))
                risk_chart = create_risk_chart(legal_risk, economic_risk)
                st.plotly_chart(risk_chart, use_container_width=True)
        
        with col2:
            if economic:
                costs = economic.get('estimated_costs', {})
                profit = economic.get('potential_profit')
                financial_chart = create_financial_chart(costs, profit)
                if financial_chart:
                    st.plotly_chart(financial_chart, use_container_width=True)
        
        # Сводная таблица
        st.markdown("### 📋 Сводная таблица")
        summary_data = []
        
        if legal:
            summary_data.append({
                "Эксперт": "🏛️ Юрист",
                "Уровень риска": str(legal.get('risk_level', 'средний')).upper(),
                "Уверенность": f"{legal.get('confidence', 0):.0%}",
                "Ключевая рекомендация": legal.get('recommendation', '')[:100] + "..."
            })
        
        if economic:
            summary_data.append({
                "Эксперт": "💰 Экономист", 
                "Уровень риска": str(economic.get('risk_level', 'средний')).upper(),
                "Уверенность": f"{economic.get('confidence', 0):.0%}",
                "Ключевая рекомендация": economic.get('recommendation', '')[:100] + "..."
            })
        
        if decision:
            summary_data.append({
                "Эксперт": "👔 Руководитель",
                "Уровень риска": "-",
                "Уверенность": f"{decision.get('confidence', 0):.0%}",
                "Ключевая рекомендация": decision.get('reasoning', '')[:100] + "..."
            })
        
        if summary_data:
            st.table(summary_data)

def main():
    """Основная функция приложения"""
    setup_environment()
    
    # Боковая панель
    with st.sidebar:
        st.markdown("## ⚙️ Настройки")
        
        analysis_mode = st.selectbox(
            "Режим анализа",
            ["Новый конкурс", "Пример конкурса"],
            help="Выберите режим работы с системой"
        )
        
        st.markdown("---")
        st.markdown("### 🤖 О системе")
        st.markdown("""
        **AI-Агент Анализа Конкурсов** включает:
        
        🏛️ **Юрист** - анализ правовых рисков
        
        💰 **Экономист** - оценка финансовой целесообразности
        
        👔 **Руководитель** - принятие итогового решения
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Статистика")
        if 'analysis_count' not in st.session_state:
            st.session_state.analysis_count = 0
        st.metric("Проведено анализов", st.session_state.analysis_count)
    
    # Основной контент
    if analysis_mode == "Пример конкурса":
        st.markdown('<div class="main-header">🤖 AI-Агент Анализа Конкурсов</div>', unsafe_allow_html=True)
        st.markdown("### 📋 Анализ примера конкурса")
        
        if st.button("🚀 Запустить анализ примера", type="primary", use_container_width=True):
            with st.spinner("🔍 Анализирую пример конкурса..."):
                try:
                    contest = create_sample_contest()
                    analysis_graph = ContestAnalysisGraph()
                    result = analysis_graph.analyze_contest(contest)
                    
                    st.session_state.analysis_count += 1
                    st.session_state.last_result = result
                    
                    st.success("✅ Анализ завершен успешно!")
                    display_analysis_results(result)
                    
                except Exception as e:
                    st.error(f"❌ Ошибка при анализе: {str(e)}")
    
    else:  # Новый конкурс
        contest_data = display_contest_form()
        
        if st.button("🚀 Запустить анализ", type="primary", use_container_width=True):
            if not contest_data['title'] or not contest_data['description']:
                st.error("❌ Пожалуйста, заполните обязательные поля: название и описание конкурса")
            else:
                with st.spinner("🔍 Анализирую конкурс... Это может занять несколько минут."):
                    try:
                        contest = ContestInfo(
                            title=contest_data['title'],
                            description=contest_data['description'],
                            budget=contest_data['budget'],
                            deadline=contest_data['deadline'],
                            requirements=contest_data['requirements'],
                            documentation=contest_data['documentation'],
                            evaluation_criteria=contest_data['evaluation_criteria'],
                            additional_info={}
                        )
                        
                        analysis_graph = ContestAnalysisGraph()
                        result = analysis_graph.analyze_contest(contest)
                        
                        st.session_state.analysis_count += 1
                        st.session_state.last_result = result
                        
                        st.success("✅ Анализ завершен успешно!")
                        display_analysis_results(result)
                        
                    except Exception as e:
                        st.error(f"❌ Ошибка при анализе: {str(e)}")
                        st.info("Проверьте правильность API ключа Mistral AI и подключение к интернету")
    
    # Показать последний результат если есть
    if hasattr(st.session_state, 'last_result') and analysis_mode == "Новый конкурс":
        if st.button("📊 Показать последний анализ"):
            display_analysis_results(st.session_state.last_result)

if __name__ == "__main__":
    main() 