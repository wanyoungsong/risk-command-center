import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import plotly.graph_objects as go
from pyvis.network import Network
import time
from datetime import datetime, timedelta
import pandas as pd
from scipy.stats import norm
import google.generativeai as genai
from neo4j import GraphDatabase
import json
import re

# ==========================================
# 0. API 및 DB 접속 설정
# ==========================================
GOOGLE_API_KEY = None
NEO4J_URI = None
NEO4J_USER = None
NEO4J_PASSWORD = None

try:
    from google.colab import userdata
    GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")
    NEO4J_URI = userdata.get("NEO4J_URI")
    NEO4J_USER = userdata.get("NEO4J_USER")
    NEO4J_PASSWORD = userdata.get("NEO4J_PASSWORD")
except (ImportError, Exception):
    pass

if not GOOGLE_API_KEY:
    try:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
        NEO4J_URI = st.secrets["NEO4J_URI"]
        NEO4J_USER = st.secrets["NEO4J_USER"]
        NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]
    except (KeyError, FileNotFoundError, Exception):
        pass

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.error("⚠️ GOOGLE_API_KEY가 설정되지 않았습니다.")

# ==========================================
# 1. 공통 엔진 모듈 및 데이터 셋업
# ==========================================
@st.cache_data
def generate_daily_risk_factors(days=30):
    np.random.seed(42)
    dates = [datetime.now().date() - timedelta(days=x) for x in range(days)]
    dates.reverse()
    data = {'Date': dates}

    data['KTB_6M'] = np.linspace(3.20, 3.45, days) + np.random.normal(0, 0.03, days)
    data['KTB_1Y'] = np.linspace(3.25, 3.55, days) + np.random.normal(0, 0.04, days)
    data['KTB_3Y'] = np.linspace(3.30, 3.70, days) + np.random.normal(0, 0.05, days)
    data['KTB_5Y'] = np.linspace(3.40, 3.90, days) + np.random.normal(0, 0.06, days)
    data['Corp_6M'] = np.linspace(3.80, 4.10, days) + np.random.normal(0, 0.05, days)
    data['Corp_1Y'] = np.linspace(3.90, 4.30, days) + np.random.normal(0, 0.06, days)

    data['KOSPI200_Close'] = np.linspace(360, 330, days) + np.random.normal(0, 3, days)
    data['Samsung_Close'] = np.linspace(78000, 71000, days) + np.random.normal(0, 800, days)
    data['SKHynix_Close'] = np.linspace(160000, 145000, days) + np.random.normal(0, 1500, days)
    data['Naver_Close'] = np.linspace(200000, 180000, days) + np.random.normal(0, 2000, days)

    data['KOSPI200_Intensity'] = np.random.uniform(85, 115, days)
    data['Samsung_Intensity'] = np.random.uniform(90, 120, days)
    data['SKHynix_Intensity'] = np.random.uniform(80, 110, days)
    data['Naver_Intensity'] = np.random.uniform(70, 105, days)

    data['Vol_KOSPI200'] = np.linspace(0.15, 0.28, days) + np.random.normal(0, 0.01, days)
    data['Vol_Samsung'] = np.linspace(0.18, 0.32, days) + np.random.normal(0, 0.01, days)
    data['Vol_SKHynix'] = np.linspace(0.25, 0.40, days) + np.random.normal(0, 0.02, days)
    data['Vol_Naver'] = np.linspace(0.30, 0.45, days) + np.random.normal(0, 0.02, days)

    return pd.DataFrame(data)

df_market_data = generate_daily_risk_factors(30)

bond_portfolio = [
    {"bond_id": "B01", "name": "국고채_6개월_A", "tenor": 0.5, "curve": "KTB_6M", "face_value": 10000, "qty": 1000},
    {"bond_id": "B02", "name": "국고채_1년_A", "tenor": 1.0, "curve": "KTB_1Y", "face_value": 10000, "qty": 1500},
    {"bond_id": "B03", "name": "국고채_3년_A", "tenor": 3.0, "curve": "KTB_3Y", "face_value": 10000, "qty": 2000},
    {"bond_id": "B04", "name": "국고채_5년_A", "tenor": 5.0, "curve": "KTB_5Y", "face_value": 10000, "qty": 800},
    {"bond_id": "B05", "name": "회사채_6개월_A", "tenor": 0.5, "curve": "Corp_6M", "face_value": 10000, "qty": 1200},
    {"bond_id": "B06", "name": "회사채_1년_A", "tenor": 1.0, "curve": "Corp_1Y", "face_value": 10000, "qty": 2500},
    {"bond_id": "B07", "name": "국고채_1년_B", "tenor": 1.0, "curve": "KTB_1Y", "face_value": 10000, "qty": 3000},
    {"bond_id": "B08", "name": "국고채_3년_B", "tenor": 3.0, "curve": "KTB_3Y", "face_value": 10000, "qty": 1000},
    {"bond_id": "B09", "name": "회사채_6개월_B", "tenor": 0.5, "curve": "Corp_6M", "face_value": 10000, "qty": 5000},
    {"bond_id": "B10", "name": "회사채_1년_B", "tenor": 1.0, "curve": "Corp_1Y", "face_value": 10000, "qty": 2000},
]
df_bonds = pd.DataFrame(bond_portfolio)

els_portfolio = [
    {"els_id": "E01", "name": "ELS_KOSPI_SAMSUNG_KI50", "asset1": "KOSPI200_Close", "asset2": "Samsung_Close", "ki_barrier": 50, "qty": 1000},
    {"els_id": "E02", "name": "ELS_KOSPI_HYNIX_KI55", "asset1": "KOSPI200_Close", "asset2": "SKHynix_Close", "ki_barrier": 55, "qty": 1200},
    {"els_id": "E03", "name": "ELS_SAMSUNG_NAVER_KI55", "asset1": "Samsung_Close", "asset2": "Naver_Close", "ki_barrier": 55, "qty": 800},
    {"els_id": "E04", "name": "ELS_KOSPI_NAVER_KI60", "asset1": "KOSPI200_Close", "asset2": "Naver_Close", "ki_barrier": 60, "qty": 1500},
    {"els_id": "E05", "name": "ELS_HYNIX_NAVER_KI60", "asset1": "SKHynix_Close", "asset2": "Naver_Close", "ki_barrier": 60, "qty": 2000},
    {"els_id": "E06", "name": "ELS_KOSPI_SAMSUNG_KI65", "asset1": "KOSPI200_Close", "asset2": "Samsung_Close", "ki_barrier": 65, "qty": 1000},
    {"els_id": "E07", "name": "ELS_KOSPI_HYNIX_KI65", "asset1": "KOSPI200_Close", "asset2": "SKHynix_Close", "ki_barrier": 65, "qty": 900},
    {"els_id": "E08", "name": "ELS_SAMSUNG_NAVER_KI70", "asset1": "Samsung_Close", "asset2": "Naver_Close", "ki_barrier": 70, "qty": 1100},
    {"els_id": "E09", "name": "ELS_HYNIX_NAVER_KI70", "asset1": "SKHynix_Close", "asset2": "Naver_Close", "ki_barrier": 70, "qty": 3000},
    {"els_id": "E10", "name": "ELS_KOSPI_NAVER_KI70", "asset1": "KOSPI200_Close", "asset2": "Naver_Close", "ki_barrier": 70, "qty": 2500},
]
df_els = pd.DataFrame(els_portfolio)

@st.cache_data
def revalue_bonds_multi(df, current_mkt, base_mkt):
    results = df.copy()
    results['base_rate'] = results['curve'].map(lambda x: base_mkt[x]) / 100.0
    results['current_rate'] = results['curve'].map(lambda x: current_mkt[x]) / 100.0
    results['old_price'] = results['face_value'] / ((1 + results['base_rate']) ** results['tenor'])
    results['new_price'] = results['face_value'] / ((1 + results['current_rate']) ** results['tenor'])
    results['price_change'] = results['new_price'] - results['old_price']
    results['pnl'] = results['price_change'] * results['qty']
    return results

@st.cache_data
def revalue_els_multi(df, current_mkt, base_mkt):
    results = df.copy()
    r1 = (results['asset1'].map(lambda x: current_mkt[x]) / results['asset1'].map(lambda x: base_mkt[x])) * 100
    r2 = (results['asset2'].map(lambda x: current_mkt[x]) / results['asset2'].map(lambda x: base_mkt[x])) * 100
    worst_perf = np.minimum(r1, r2)
    distance_to_ki = worst_perf - results['ki_barrier']

    gamma_bleed = np.where(distance_to_ki > 0, 2000 * np.exp(-0.15 * distance_to_ki), 3000 + (results['ki_barrier'] - worst_perf) * 100)
    
    vol_asset1 = results['asset1'].apply(lambda x: x.replace('_Close', ''))
    vol_asset2 = results['asset2'].apply(lambda x: x.replace('_Close', ''))
    avg_vol = (vol_asset1.map(lambda x: current_mkt[f'Vol_{x}']) + vol_asset2.map(lambda x: current_mkt[f'Vol_{x}'])) / 2.0
    vega_loss = np.maximum(0, avg_vol - 0.20) * 50000

    avg_intensity = (vol_asset1.map(lambda x: current_mkt[f'{x}_Intensity']) + vol_asset2.map(lambda x: current_mkt[f'{x}_Intensity'])) / 2.0
    liquidity_loss = np.where(avg_intensity < 100, (100 - avg_intensity) * 20, 0)

    book_pnl_per_unit = - gamma_bleed - vega_loss - liquidity_loss

    initial_gamma = np.where(100 - results['ki_barrier'] > 0, 2000 * np.exp(-0.15 * (100 - results['ki_barrier'])), 0)
    initial_vol = (vol_asset1.map(lambda x: base_mkt[f'Vol_{x}']) + vol_asset2.map(lambda x: base_mkt[f'Vol_{x}'])) / 2.0
    initial_vega = np.maximum(0, initial_vol - 0.20) * 50000
    initial_pnl_per_unit = - initial_gamma - initial_vega

    results['old_price'] = initial_pnl_per_unit 
    results['new_price'] = book_pnl_per_unit
    results['price_change'] = results['new_price'] - results['old_price']
    results['pnl'] = results['price_change'] * results['qty']
    return results

@st.cache_data
def calculate_parametric_var(df_b, df_e, df_mkt, confidence_level=0.99):
    factors_df = df_mkt.drop(columns=['Date'])
    returns_df = factors_df.pct_change().dropna()
    cov_matrix = returns_df.cov()
    base_mkt = factors_df.iloc[-1].to_dict()

    def get_total_pnl(mkt_state):
        b_res = revalue_bonds_multi(df_b, mkt_state, base_mkt)
        e_res = revalue_els_multi(df_e, mkt_state, base_mkt)
        return b_res['pnl'].sum() + e_res['pnl'].sum()

    base_pnl = get_total_pnl(base_mkt)
    shock_size = 0.01
    sensitivities = {}

    for factor in cov_matrix.columns:
        shocked_mkt = base_mkt.copy()
        shocked_mkt[factor] = shocked_mkt[factor] * (1 + shock_size)
        shocked_pnl = get_total_pnl(shocked_mkt)
        sensitivities[factor] = (shocked_pnl - base_pnl) / shock_size

    S = pd.Series(sensitivities)
    portfolio_variance = S.dot(cov_matrix).dot(S)
    portfolio_std_dev = np.sqrt(portfolio_variance)

    z_score = norm.ppf(confidence_level)
    var_amount = z_score * portfolio_std_dev
    return abs(var_amount), S

# ==========================================
# 2. GraphRAG & LLM 프롬프트 에이전트
# ==========================================
def get_knowledge_graph_context(risk_driver):
    if not NEO4J_URI or not NEO4J_PASSWORD: return _fallback_kg_context(risk_driver)
    kg_context = ""
    try:
        with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
            with driver.session() as session:
                query = """
                MATCH (rf:RiskFactor)-[:TRIGGERS_POLICY]->(rule:ComplianceRule)
                MATCH (a:AssetClass)-[exp:EXPOSED_TO]->(rf)
                WHERE rf.name CONTAINS $keyword OR rf.desc CONTAINS $keyword
                RETURN a.name AS asset, exp.greek AS greek, exp.logic AS logic,
                       rule.name AS rule_name, rule.code AS rule_code, rule.action_plan AS action_plan
                """
                kw = "Volatility" if "Vol" in risk_driver else "Intensity" if "Intensity" in risk_driver else "Interest_Rate"
                result = session.run(query, keyword=kw)
                records = list(result)
                if not records: return _fallback_kg_context(risk_driver)

                for record in records:
                    kg_context += f"##### 🕸️ 데이터 리니지 인과관계\n- **대상 자산군**: {record['asset']}\n- **민감도(Greeks)**: {record['greek']}\n- **손실 발생 논리**: {record['logic']}\n"
                    kg_context += f"##### 🚨 적용 규정 및 지침\n- **규정**: {record['rule_name']} ({record['rule_code']})\n- **대응 조치**: {record['action_plan']}\n\n"
    except Exception:
        kg_context = _fallback_kg_context(risk_driver)
    return kg_context

def _fallback_kg_context(risk_driver):
    if "Vol" in risk_driver:
        return "- **대상 자산군**: Derivatives (ELS 자체 헤지북)\n- **민감도**: Vega\n- **손실 논리**: 변동성 급등 시 헤지 비용 기하급수적 팽창\n- **적용 규정**: Article 14-3\n- **대응 조치**: Vega 중립을 위한 옵션 양매수 헤지 비중 즉각 확대 요망"
    else:
        return "- **대상 자산군**: Fixed_Income (채권 매수북)\n- **민감도**: Rho\n- **손실 논리**: 금리 상승 시 평가손실 발생\n- **적용 규정**: Article 75-1\n- **대응 조치**: 듀레이션 갭 축소를 위한 국채선물 매도 확대 요망"

@st.cache_data(ttl=3600)
def get_dynamic_risk_limits():
    limits = {"전사 VaR 한도": 1500.0, "금리 민감도(Rho) 한도": 40.0, "변동성 민감도(Vega) 한도": 30.0}
    if not NEO4J_URI or not NEO4J_PASSWORD: return limits
    try:
        with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
            with driver.session() as session:
                result = session.run("MATCH (l:RiskLimit) RETURN l.name AS name, l.limit_value AS val")
                for record in result: limits[record["name"]] = float(record["val"])
    except Exception: pass
    return limits

def get_compliance_graph_context(dept_code, usage_pct):
    if not NEO4J_URI or not NEO4J_PASSWORD: return _fallback_compliance_context(dept_code, usage_pct)
    kg_context = ""
    try:
        with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
            with driver.session() as session:
                th_kw = "100% 이상" if usage_pct >= 100 else "90% 이상" if usage_pct >= 90 else "정상"
                if th_kw == "정상": return "✅ 현재 한도 소진율은 정상 범위입니다."
                query = """
                MATCH (d:Department {code: $dept_code})-[:MONITORS]->(l:RiskLimit)-[tp:TRIGGERS_POLICY]->(rule:ComplianceRule)
                WHERE tp.threshold CONTAINS $threshold_kw
                RETURN l.name AS limit_name, l.metric AS metric, tp.action_plan AS action_plan, rule.name AS rule_name
                """
                result = session.run(query, dept_code=dept_code, threshold_kw=th_kw)
                records = list(result)
                if not records: return _fallback_compliance_context(dept_code, usage_pct)
                for record in records:
                    kg_context += f"- **규정명**: {record['rule_name']}\n- **AI 조치 처방**: {record['action_plan']}\n\n"
    except Exception:
        kg_context = _fallback_compliance_context(dept_code, usage_pct)
    return kg_context

def _fallback_compliance_context(dept_code, usage_pct):
    if usage_pct < 90: return "✅ 정상 범위"
    if dept_code == 'ENTERPRISE': return "- **규정명**: 전사 한도 초과 대응\n- **AI 조치 처방**: 즉각적인 포지션 축소, CRO 대면 보고 소집" if usage_pct >= 100 else "- **규정명**: 경고 대응\n- **AI 조치 처방**: 신규 리스크 테이킹 즉각 중지"
    elif dept_code == 'BOND_DESK': return "- **규정명**: 채권 한도 초과 대응\n- **AI 조치 처방**: IRS 페이 포지션 구축 또는 장기채 즉시 매도"
    elif dept_code == 'ELS_DESK': return "- **규정명**: ELS 비선형 한도 초과\n- **AI 조치 처방**: 롤오버 중지 및 베가 중립 헤지 기계적 확대"
    return "✅ 정상 범위"

def stream_ai_briefing(total_pnl, els_pnl, bond_pnl, top_driver, var_pct, kg_context):
    model = genai.GenerativeModel('gemini-2.5-flash')
    prompt = f"""너는 최고리스크책임자(CRO)를 보좌하는 수석 리스크 AI 참모야. 아래 팩트를 바탕으로 브리핑을 작성해.
    [데이터] 일간 P&L: {total_pnl/100000000:,.1f}억, ELS P&L: {els_pnl/100000000:,.1f}억, 채권 P&L: {bond_pnl/100000000:,.1f}억, 핵심동인: {top_driver}, VaR소진율: {var_pct:.1f}%
    [온톨로지 규정] {kg_context}
    인사말 없이 시작하고, 숫자를 명시하며 마크다운 불릿으로 3문단 이내로 작성해."""
    for chunk in model.generate_content(prompt, stream=True):
        if chunk.text:
            yield chunk.text
            time.sleep(0.01)

def stream_ai_prescription(dept_name, usage_pct, metric_name, exposure_amt, limit_amt, kg_context):
    model = genai.GenerativeModel('gemini-2.5-flash')
    prompt = f"""너는 수석 리스크 AI 참모야. 아래 부서별 한도 초과 상황에 대한 처방을 내려.
    [현황] 부서: {dept_name}, 지표: {metric_name}, 노출도: {exposure_amt:,.1f}억, 한도: {limit_amt:,.1f}억 (소진율 {usage_pct:.1f}%)
    [지침] {kg_context}
    인사말 없이, 규정을 근거로 액션 플랜을 3문단 이내로 강하게 권고해."""
    for chunk in model.generate_content(prompt, stream=True):
        if chunk.text: yield chunk.text

def generate_dynamic_scenario(user_input):
    model = genai.GenerativeModel('gemini-2.5-flash')
    prompt = f"""너는 AI 참모야. 다음 [사용자 입력]을 분석해 JSON으로 응답해.
    입력: {user_input}
    지시: 관련 없으면 is_relevant: false, rag_summary에 거절 메시지 작성. 관련 있으면 true로 하고 파라미터 추출.
    JSON 구조: {{"is_relevant": true/false, "rag_summary": "시황요약", "kg_logic": "인과관계", "parameters": [{{"factor": "KOSPI 200 지수", "current": "100%", "target": "75%", "duration": "14일"}}]}}"""
    response = model.generate_content(prompt, generation_config=genai.GenerationConfig(response_mime_type="application/json"))
    return json.loads(response.text)

def stream_scenario_response(total_pnl, scenario_df_json):
    model = genai.GenerativeModel('gemini-2.5-flash')
    prompt = f"""너는 AI 참모야. 시뮬레이션 결과(파라미터: {scenario_df_json}, 최대손실: {total_pnl/100000000:,.1f}억)를 바탕으로 ELS 및 채권 데스크의 즉각적 액션 플랜을 사내 규정(가상)을 근거로 제시해."""
    for chunk in model.generate_content(prompt, stream=True):
        if chunk.text: yield chunk.text

def stream_rst_response(target_loss, k_val, s_val, r_val):
    model = genai.GenerativeModel('gemini-2.5-flash')
    prompt = f"""너는 AI 참모야. 목표손실 {target_loss:,.0f}억 역산 결과 (KOSPI: {k_val:.1f}%, 삼성전자: {s_val:.1f}%, 금리: {r_val:.0f}bp 충격)에 대한 경영진 시사점을 작성해."""
    for chunk in model.generate_content(prompt, stream=True):
        if chunk.text: yield chunk.text


# ==========================================
# 3. Streamlit 메인 UI (3단 Agent 레이아웃)
# ==========================================
st.set_page_config(layout="wide", page_title="AI Risk Agent UI")

# 실시간 연산 데이터 준비
base_mkt_state = df_market_data.iloc[-2].to_dict()
current_mkt_state = df_market_data.iloc[-1].to_dict()
bonds_res_today = revalue_bonds_multi(df_bonds, current_mkt_state, base_mkt_state)
els_res_today = revalue_els_multi(df_els, current_mkt_state, base_mkt_state)

daily_bond_pnl = bonds_res_today['pnl'].sum()
daily_els_pnl = els_res_today['pnl'].sum()
daily_total_pnl = daily_bond_pnl + daily_els_pnl

var_amount, var_sens = calculate_parametric_var(df_bonds, df_els, df_market_data, 0.99)
var_limit = 1500000000
var_usage_pct = (var_amount / var_limit) * 100
top_risk_driver = var_sens.abs().idxmax()
rho_exposure_bn = var_sens.filter(like='KTB').sum() / 100000000
vega_exposure_bn = var_sens.filter(like='Vol').sum() / 100000000
dynamic_limits = get_dynamic_risk_limits()

# -----------------
# Left Panel: Sidebar
# -----------------
with st.sidebar:
    st.title("🤖 Risk Agent")
    st.caption("AI 기반 전사 리스크 지휘소")
    st.markdown("---")
    
    selected_mode = st.radio("업무 모드 선택", [
        "📊 1-1. 마켓 리스크 브리핑",
        "🚨 1-2. 부서별 한도 관리",
        "▶️ 2-1. 위기 시나리오 분석",
        "◀️ 2-2. 역방향 위기 탐색"
    ])
    
    # 모드 변경 시 세션 초기화
    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = selected_mode
        st.session_state.messages = []
    elif st.session_state.current_mode != selected_mode:
        st.session_state.current_mode = selected_mode
        st.session_state.messages = []
        if 'scenario_fig' in st.session_state: del st.session_state['scenario_fig']
        if 'rst_radar' in st.session_state: del st.session_state['rst_radar']
        if 'rst_contour' in st.session_state: del st.session_state['rst_contour']

    st.markdown("---")
    st.caption(f"System Status: {'Warning' if var_usage_pct > 90 else 'Normal'}")

# -----------------
# 메인 레이아웃 분할 (Center: Chat, Right: Viz)
# -----------------
col_chat, col_viz = st.columns([1.3, 1.7], gap="large")

# 우측 캔버스를 위해 미리 컨테이너 선언
with col_viz:
    viz_container = st.container()

# -----------------
# Center Panel: Chat Logic
# -----------------
with col_chat:
    st.subheader(f"{selected_mode.split('. ')[1]}")
    
    # 초기 인사말 설정
    if not st.session_state.messages:
        if "1-1" in selected_mode:
            st.session_state.messages.append({"role": "assistant", "content": "전사 마켓 리스크 현황이 우측 대시보드에 업데이트되었습니다. 경영진 보고용 시황 브리핑이 필요하시면 지시해 주십시오. (예: '현재 상황 브리핑 해줘')"})
        elif "1-2" in selected_mode:
            st.session_state.messages.append({"role": "assistant", "content": "부서별 리스크 한도 모니터링 중입니다. 한도 초과에 대한 처방전 기안이 필요하면 지시해 주십시오. (예: 'ELS 데스크 한도 처방 내려줘')"})
        elif "2-1" in selected_mode:
            st.session_state.messages.append({"role": "assistant", "content": "거시 경제 위기 시나리오를 자유롭게 지시해 주십시오. (예: '전쟁 확전으로 유가가 급등하고 인플레이션 우려로 증시가 장기 침체되는 상황 분석해줘')"})
        elif "2-2" in selected_mode:
            st.session_state.messages.append({"role": "assistant", "content": "경영진이 우려하는 목표 손실액을 입력해 주시면 최악의 위기 경로를 역산합니다. (예: '목표손실 -400억 경로 찾아줘')"})

    # 채팅 히스토리 출력
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # 입력창 및 응답 처리
    if prompt := st.chat_input("AI 참모에게 지시를 입력하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            
            # [모드 1-1] 브리핑
            if "1-1" in selected_mode:
                kg_context = get_knowledge_graph_context(top_risk_driver)
                response = st.write_stream(stream_ai_briefing(daily_total_pnl, daily_els_pnl, daily_bond_pnl, top_risk_driver, var_usage_pct, kg_context))
                st.session_state.messages.append({"role": "assistant", "content": response})

            # [모드 1-2] 한도 처방
            elif "1-2" in selected_mode:
                dept_code, dept_name, usage, metric, exp, limit = 'ENTERPRISE', '전사 통합 리스크 위원회', var_usage_pct, '전사 VaR', var_amount/100000000, dynamic_limits.get("전사 VaR 한도", 1500)
                if '채권' in prompt:
                    dept_code, dept_name, usage, metric, exp, limit = 'BOND_DESK', '채권 운용 데스크', abs(rho_exposure_bn/dynamic_limits.get("금리 민감도(Rho) 한도", 40))*100, 'Rho', rho_exposure_bn, dynamic_limits.get("금리 민감도(Rho) 한도", 40)
                elif 'ELS' in prompt or 'els' in prompt.lower():
                    dept_code, dept_name, usage, metric, exp, limit = 'ELS_DESK', 'ELS 운용 데스크', abs(vega_exposure_bn/dynamic_limits.get("변동성 민감도(Vega) 한도", 30))*100, 'Vega', vega_exposure_bn, dynamic_limits.get("변동성 민감도(Vega) 한도", 30)
                
                kg_context = get_compliance_graph_context(dept_code, usage)
                response = st.write_stream(stream_ai_prescription(dept_name, usage, metric, exp, limit, kg_context))
                st.session_state.messages.append({"role": "assistant", "content": response})

            # [모드 2-1] 시나리오 WHAT-IF
            elif "2-1" in selected_mode:
                with st.spinner("시나리오 파라미터 추론 중..."):
                    result = generate_dynamic_scenario(prompt)
                
                if not result.get("is_relevant"):
                    response = result.get("rag_summary", "관련성 없음")
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.write("**[AI 파라미터 추출 완료]**")
                    df_params = pd.DataFrame(result['parameters'])
                    st.dataframe(df_params, hide_index=True)
                    st.write("📈 **시뮬레이션 엔진 구동을 시작합니다...**")

                    # 동적 3D 차트 시각화 (Right Panel 렌더링)
                    kospi_target, samsung_target = 100.0, 100.0
                    for _, row in df_params.iterrows():
                        if "KOSPI" in row["factor"]: kospi_target = float(re.findall(r'[-+]?\d*\.?\d+', str(row["target"]))[-1])
                        elif "삼성" in row["factor"]: samsung_target = float(re.findall(r'[-+]?\d*\.?\d+', str(row["target"]))[-1])
                    
                    steps_count = 5
                    traj_x = np.linspace(100, kospi_target, steps_count).tolist()
                    traj_y = np.linspace(100, samsung_target, steps_count).tolist()
                    traj_z = [100 - ((100-x)*1.5 + (100-y)*1.5) for x,y in zip(traj_x, traj_y)]
                    
                    X, Y = np.meshgrid(np.linspace(40,100,20), np.linspace(40,100,20))
                    Z = 100 + (X-100)*0.3 + (Y-100)*0.3 - np.where(X<80,(80-X)*1.5,0) - np.where(Y<60,(60-Y)*2.5,0)

                    with viz_container.container():
                        st.subheader("시뮬레이션 궤적 시각화")
                        chart_place = st.empty()
                        for i in range(steps_count):
                            fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Blues', opacity=0.7)])
                            fig.add_trace(go.Scatter3d(x=traj_x[:i+1], y=traj_y[:i+1], z=traj_z[:i+1], mode='lines', line=dict(color='orange', width=6)))
                            fig.add_trace(go.Scatter3d(x=[traj_x[i]], y=[traj_y[i]], z=[traj_z[i]], mode='markers', marker=dict(size=10, color='red')))
                            fig.update_layout(title=f"⏳ 진행 상태: Step {i+1}", height=450, margin=dict(l=0,r=0,b=0,t=30))
                            chart_place.plotly_chart(fig, use_container_width=True, key=f"sim_anim_{i}")
                            time.sleep(0.5)
                        st.session_state.scenario_fig = fig

                    st.write("---")
                    sim_loss = -50000000000 # Demo Hardcoded Loss for speed
                    response = st.write_stream(stream_scenario_response(sim_loss, result['parameters']))
                    st.session_state.messages.append({"role": "assistant", "content": f"[파라미터 추출 완료 및 시뮬레이션 실행]\n\n{response}"})

            # [모드 2-2] 역방향 탐색
            elif "2-2" in selected_mode:
                nums = re.findall(r'-?\d+', prompt)
                target_loss = float(nums[0]) if nums else -400.0
                
                st.write(f"목표 손실 **{target_loss}억 원**을 유발하는 최단 위기 경로를 DML 엔진으로 탐색합니다...")
                
                # 가짜 탐색 궤적 데이터 (Demo 시각용)
                k_path, s_path, r_path = np.linspace(100, 65, 5), np.linspace(100, 70, 5), np.linspace(0, 80, 5)
                
                with viz_container.container():
                    st.subheader("역위기상황(RST) 탐색 지형도")
                    c1, c2 = st.columns(2)
                    radar_ph, contour_ph = c1.empty(), c2.empty()
                    
                    for step in range(5):
                        ck, cs, cr = k_path[step], s_path[step], r_path[step]
                        
                        fig_radar = go.Figure(data=go.Scatterpolar(
                            r=[min(100, (100-ck)*2), min(100, (100-cs)*2), min(100, cr/1.5), min(100, (100-ck)*2)],
                            theta=['KOSPI 위험', '삼성전자 위험', '금리 위험', 'KOSPI 위험'],
                            fill='toself', line=dict(color='red', width=2)
                        ))
                        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=350, margin=dict(l=30,r=30,t=30,b=30))
                        radar_ph.plotly_chart(fig_radar, use_container_width=True, key=f"radar_anim_{step}")

                        K_MESH, S_MESH = np.meshgrid(np.linspace(50, 100, 10), np.linspace(50, 100, 10))
                        fig_contour = go.Figure(data=go.Contour(z=(K_MESH+S_MESH), x=np.linspace(50,100,10), y=np.linspace(50,100,10), colorscale='RdBu'))
                        fig_contour.add_trace(go.Scatter(x=k_path[:step+1], y=s_path[:step+1], mode='lines+markers', line=dict(color='#00FF00', width=4)))
                        fig_contour.update_layout(height=350, margin=dict(l=30,r=30,t=30,b=30))
                        contour_ph.plotly_chart(fig_contour, use_container_width=True, key=f"contour_anim_{step}")
                        time.sleep(0.5)
                        
                    st.session_state.rst_radar = fig_radar
                    st.session_state.rst_contour = fig_contour

                st.write("---")
                response = st.write_stream(stream_rst_response(target_loss, k_path[-1], s_path[-1], r_path[-1]))
                st.session_state.messages.append({"role": "assistant", "content": f"[엔진 탐색 완료]\n\n{response}"})

# -----------------
# Right Panel: Visualization Render (우측 캔버스 동적 렌더링)
# -----------------
with viz_container:
    if "1-1" in selected_mode:
        st.subheader("📊 전사 마켓 리스크 현황")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("전사 통합 P&L", f"{daily_total_pnl/100000000:,.1f}억")
        m2.metric("VaR (99%, 1D)", f"{var_amount/100000000:,.1f}억")
        m3.metric("한도 소진율", f"{var_usage_pct:.1f}%")
        m4.metric("주요 동인", top_risk_driver)
        
        st.markdown("#### 포트폴리오 노출도")
        df_summary = pd.DataFrame({"포트폴리오": ["채권 (Long)", "ELS (자체헤지)"], "Delta": [f"{rho_exposure_bn:,.1f}억", "0.0억(헤지)"], "Net P&L": [f"{daily_bond_pnl/100000000:,.1f}억", f"{daily_els_pnl/100000000:,.1f}억"]})
        st.dataframe(df_summary, use_container_width=True, hide_index=True)

        st.markdown("#### 🕸️ 리스크 파급 인과관계 맵 (Knowledge Graph)")
        net = Network(height='300px', width='100%', bgcolor='#ffffff')
        net.add_node("M", label=f"{top_risk_driver}\n충격", color="#ffb3b3", size=25)
        net.add_node("B", label="채권 포트", color="#cce5ff")
        net.add_node("E", label="ELS 운용북", color="#cce5ff")
        net.add_node("P", label="전사 손실", color="#ff9999", size=30)
        net.add_edge("M", "B", label="Rho 상승")
        net.add_edge("M", "E", label="Vega 팽창")
        net.add_edge("B", "P")
        net.add_edge("E", "P")
        net.write_html("kg.html")
        with open("kg.html", 'r', encoding='utf-8') as f: components.html(f.read(), height=320)

    elif "1-2" in selected_mode:
        st.subheader("🚨 부서별 리스크 한도 모니터링")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("##### 🏢 전사 통합")
            st.progress(min(var_usage_pct/100, 1.0), text=f"VaR 한도: {var_usage_pct:.1f}%")
        with c2:
            st.markdown("##### 📈 채권 데스크")
            rho_usage = abs(rho_exposure_bn/dynamic_limits.get("금리 민감도(Rho) 한도", 40))*100
            st.progress(min(rho_usage/100, 1.0), text=f"Rho 한도: {rho_usage:.1f}%")
        with c3:
            st.markdown("##### 📉 ELS 데스크")
            vega_usage = abs(vega_exposure_bn/dynamic_limits.get("변동성 민감도(Vega) 한도", 30))*100
            st.progress(min(vega_usage/100, 1.0), text=f"Vega 한도: {vega_usage:.1f}%")
            
    elif "2-1" in selected_mode:
        if 'scenario_fig' in st.session_state:
            st.subheader("시뮬레이션 궤적 시각화")
            st.plotly_chart(st.session_state.scenario_fig, use_container_width=True, key="sim_final_chart")
        else:
            st.info("👈 좌측 대화창에 위기 시나리오를 자연어로 지시해 주십시오.")

    elif "2-2" in selected_mode:
        if 'rst_radar' in st.session_state:
            st.subheader("역위기상황(RST) 탐색 지형도")
            cr1, cr2 = st.columns(2)
            cr1.plotly_chart(st.session_state.rst_radar, use_container_width=True, key="radar_final_chart")
            cr2.plotly_chart(st.session_state.rst_contour, use_container_width=True, key="contour_final_chart")
        else:
            st.info("👈 좌측 대화창에 도달하고자 하는 목표 손실액을 지시해 주십시오.")
