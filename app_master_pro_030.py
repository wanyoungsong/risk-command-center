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
import os

import json
import re

# --- 1. 세션 상태 관리 (페이지 이동 시 초기화 방지) ---
if 'scenario_step' not in st.session_state:
    st.session_state.scenario_step = 0
if 'rst_step' not in st.session_state:
    st.session_state.rst_step = 0
if 'batch_step' not in st.session_state:
    st.session_state.batch_step = 0
if 'incident_step' not in st.session_state:
    st.session_state.incident_step = 0

# --- 2. 사이드바 네비게이션 ---
st.set_page_config(layout="wide", page_title="Enterprise Risk Command Center")
with st.sidebar:
    st.title("🏦 Risk Command Center")
    st.markdown("---")

    # 메인 메뉴
    main_menu = st.radio("Main Navigation", [
        "1. 전사 리스크 대시보드",
        "2. 스트레스 테스트 데스크",
        "3. 시스템 오퍼레이션",
        "4. 장애 대응 가이드 에이전트"
    ])

    # 1번 메뉴 선택 시에만 나타나는 서브 메뉴
    if main_menu == "1. 전사 리스크 대시보드":
        sub_menu = st.radio("↳ 세부 메뉴", [
            "1-1. 전사 마켓 리스크 브리핑",
            "1-2. 부서별 한도 관리 및 처방"
        ])
    # 2번 메뉴 선택 시에만 나타나는 서브 메뉴
    elif main_menu == "2. 스트레스 테스트 데스크":
        sub_menu = st.radio("↳ 세부 메뉴", [
            "2-1. 순방향 WHAT-IF 시뮬레이션",
            "2-2. 역방향 위기 좌표 탐색 (RST)"
        ])
    else:
        sub_menu = None

    st.markdown("---")
    st.caption("System Status: Normal (Warning in Limit)")
    st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# --- 3. 공통 데이터 셋업 (다차원 리스크 팩터 & 포트폴리오) ---
# --- (시작) API 및 Neo4j DB 접속 설정 (Colab & Streamlit Cloud 동시 지원) ---

# 초기 변수 세팅
GOOGLE_API_KEY = None
NEO4J_URI = None
NEO4J_USER = None
NEO4J_PASSWORD = None

# 1. 구글 코랩(Colab) 환경인지 먼저 확인하고 userdata에서 열쇠를 찾습니다.
try:
    from google.colab import userdata
    GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")
    NEO4J_URI = userdata.get("NEO4J_URI")
    NEO4J_USER = userdata.get("NEO4J_USER")
    NEO4J_PASSWORD = userdata.get("NEO4J_PASSWORD")
except (ImportError, Exception):
    # 코랩 환경이 아니거나 모듈을 못 부르면 조용히 넘어갑니다.
    pass

# 2. 코랩에서 열쇠를 못 찾았다면(즉, 깃허브를 통해 스트림릿 클라우드에 배포된 상태라면) st.secrets를 뒤집니다.
if not GOOGLE_API_KEY:
    try:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
        NEO4J_URI = st.secrets["NEO4J_URI"]
        NEO4J_USER = st.secrets["NEO4J_USER"]
        NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]
    except (KeyError, FileNotFoundError, Exception):
        pass

# 3. 최종적으로 확보한 열쇠를 시스템에 꽂아 넣습니다.
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.error("⚠️ GOOGLE_API_KEY가 코랩(userdata)이나 스트림릿(secrets) 환경 모두에 설정되지 않았습니다. Fallback 모드로 동작합니다.")

if not NEO4J_URI or not NEO4J_PASSWORD:
    st.error("⚠️ Neo4j 접속 정보가 부족합니다. AI 브리핑 기능이 Fallback 모드로 동작합니다.")

# (이후 코드에서 GraphDatabase.driver 호출 시, 확보된 NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD 변수를 그대로 사용하면 됩니다.)
# --- (끝) API 및 Neo4j DB 접속 설정 (Colab & Streamlit Cloud 동시 지원) ---

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

# --- 4. 엔진 모듈 (Long/Short 및 자체 헤지북 로직 반영) ---

@st.cache_data
def revalue_bonds_multi(df, current_mkt, base_mkt):
    # [채권: 매수(Long) 포지션] -> 금리 상승 시 가격 하락 = 손실
    results = df.copy()
    results['base_rate'] = results['curve'].map(lambda x: base_mkt[x]) / 100.0
    results['current_rate'] = results['curve'].map(lambda x: current_mkt[x]) / 100.0

    results['old_price'] = results['face_value'] / ((1 + results['base_rate']) ** results['tenor'])
    results['new_price'] = results['face_value'] / ((1 + results['current_rate']) ** results['tenor'])

    # Long 포지션이므로 가격 변동이 곧 P&L
    results['price_change'] = results['new_price'] - results['old_price']
    results['pnl'] = results['price_change'] * results['qty']
    return results

@st.cache_data
def revalue_els_multi(df, current_mkt, base_mkt):
    # [ELS: 자체 헤지북 (Short ELS + Long Hedge)] -> 델타 중립 가정, 비선형 리스크만 Net P&L로 산출
    results = df.copy()

    r1 = (results['asset1'].map(lambda x: current_mkt[x]) / results['asset1'].map(lambda x: base_mkt[x])) * 100
    r2 = (results['asset2'].map(lambda x: current_mkt[x]) / results['asset2'].map(lambda x: base_mkt[x])) * 100
    worst_perf = np.minimum(r1, r2)
    distance_to_ki = worst_perf - results['ki_barrier']

    # 1. Delta (선형 리스크): 자체 델타 헤지로 100% 방어한다고 가정 -> P&L = 0 (제외)

    # 2. Gamma Bleed (감마 출혈): Short Gamma 상태. 배리어 근접 시 잦은 헤지(Buy High, Sell Low)로 인한 비용 팽창
    gamma_bleed = np.where(
        distance_to_ki > 0,
        2000 * np.exp(-0.15 * distance_to_ki),
        3000 + (results['ki_barrier'] - worst_perf) * 100  # KI 터치 이후 갭리스크 및 헤지 붕괴
    )

    # 3. Vega Loss (베가 손실): Short Vega 상태. 내재변동성 상승 시 매도한 옵션 가치(부채) 급등
    vol_asset1 = results['asset1'].apply(lambda x: x.replace('_Close', ''))
    vol_asset2 = results['asset2'].apply(lambda x: x.replace('_Close', ''))
    avg_vol = (vol_asset1.map(lambda x: current_mkt[f'Vol_{x}']) + vol_asset2.map(lambda x: current_mkt[f'Vol_{x}'])) / 2.0
    vega_loss = np.maximum(0, avg_vol - 0.20) * 50000

    # 4. Liquidity Slippage (유동성 슬리피지): 체결강도 하락 시 델타 헤지 과정에서 발생하는 호가 스프레드 손실
    avg_intensity = (vol_asset1.map(lambda x: current_mkt[f'{x}_Intensity']) + vol_asset2.map(lambda x: current_mkt[f'{x}_Intensity'])) / 2.0
    liquidity_loss = np.where(avg_intensity < 100, (100 - avg_intensity) * 20, 0)

    # ELS 운용북 Net P&L (헤지 비용 및 부채 증가의 누적) -> 마이너스(-)로 적용
    book_pnl_per_unit = - gamma_bleed - vega_loss - liquidity_loss

    # Base 시점의 펜딩된 초기 리스크 비용 (비교용)
    initial_gamma = np.where(100 - results['ki_barrier'] > 0, 2000 * np.exp(-0.15 * (100 - results['ki_barrier'])), 0)
    initial_vol = (vol_asset1.map(lambda x: base_mkt[f'Vol_{x}']) + vol_asset2.map(lambda x: base_mkt[f'Vol_{x}'])) / 2.0
    initial_vega = np.maximum(0, initial_vol - 0.20) * 50000
    initial_pnl_per_unit = - initial_gamma - initial_vega

    results['old_price'] = initial_pnl_per_unit # 컬럼 매핑 에러 방지용
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

# --- 4. 엔진 모듈 (계속) ---
# [신규] GraphRAG 실행 모듈 (Neo4j 지식 추출 및 Gemini 프롬프트 전송)
def get_knowledge_graph_context(risk_driver):
    """
    Neo4j에 접속하여 수학 엔진이 감지한 리스크 팩터가 유발하는 인과관계와 사내 규정을 가져옴.
    """
    kg_context = ""

    # DB 접속 정보가 없으면 즉시 Fallback 컨텍스트 반환
    if not NEO4J_URI or not NEO4J_PASSWORD:
        return _fallback_kg_context(risk_driver)

    try:
        # 실제 Neo4j 연결 및 쿼리 실행
        # with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
            with driver.session() as session:
                # 리스크 팩터 키워드 매칭 (Vol, Intensity 등)
                query = """
                MATCH (rf:RiskFactor)-[:TRIGGERS_POLICY]->(rule:ComplianceRule)
                MATCH (a:AssetClass)-[exp:EXPOSED_TO]->(rf)
                WHERE rf.name CONTAINS $keyword OR rf.desc CONTAINS $keyword
                RETURN a.name AS asset, exp.greek AS greek, exp.logic AS logic,
                       rule.name AS rule_name, rule.code AS rule_code,
                       rule.action_plan AS action_plan
                """
                # 변동성, 금리 등 키워드 매칭 로직 (엔진 변수명 -> 온톨로지 키워드)
                kw = "Volatility" if "Vol" in risk_driver else "Intensity" if "Intensity" in risk_driver else "Interest_Rate"
                result = session.run(query, keyword=kw)

                # 쿼리 결과를 LLM이 읽기 좋은 텍스트로 변환
                records = list(result)
                if not records:
                    return _fallback_kg_context(risk_driver) # 쿼리 결과 없을 시 Fallback

                for record in records:
                    kg_context += f"##### 🕸️ 데이터 리니지 인과관계\n"
                    kg_context += f"- **대상 자산군**: {record['asset']}\n"
                    kg_context += f"- **민감도(Greeks)**: {record['greek']}\n"
                    kg_context += f"- **손실 발생 논리(Logic)**: {record['logic']}\n"
                    kg_context += f"##### 🚨 적용 사내 규정 및 대응 지침\n"
                    kg_context += f"- **규정명/코드**: {record['rule_name']} ({record['rule_code']})\n"
                    kg_context += f"- **대응 조치 권고(Action Plan)**: {record['action_plan']}\n\n"

    except Exception as e:
        # [안전망] DB 연결 실패 시 Fallback 컨텍스트 반환 (시연 폭망 방지)
        kg_context = f"⚠️ Neo4j DB 연결 실패 (Error: {str(e)[:50]}...). 사내 규정 온톨로지 정보를 불러오지 못했습니다.\n\n"
        kg_context += _fallback_kg_context(risk_driver)

    return kg_context

def _fallback_kg_context(risk_driver):
    """
    Neo4j 연결 실패 시 시연을 위해 반환하는 하드코딩된 온톨로지 컨텍스트
    """
    if "Vol" in risk_driver:
        return """
        ##### 🕸️ 데이터 리니지 인과관계
        - **대상 자산군**: Derivatives (지수형 ELS 자체 헤지북)
        - **민감도(Greeks)**: Vega (베가)
        - **손실 발생 논리(Logic)**: 시장 변동성이 급등할수록 ELS 옵션 구조의 헤지 비용이 기하급수적으로 팽창하여 가치 하락을 주도합니다.
        ##### 🚨 적용 사내 규정 및 대응 지침
        - **규정명/코드**: 비선형 리스크 팽창 통제 규정 (Article 14-3)
        - **대응 조치 권고(Action Plan)**: ELS 포트폴리오의 Vega 중립(Neutral)을 위한 옵션 양매수(Straddle) 헤지 비중 즉각 확대 요망
        """
    else:
        return """
        ##### 🕸️ 데이터 리니지 인과관계
        - **대상 자산군**: Fixed_Income (국고채 및 회사채 매수북)
        - **민감도(Greeks)**: Rho (로)
        - **손실 발생 논리(Logic)**: 금리가 상승할수록 채권의 현재 가치(DCF)가 하락하여 평가손실이 발생합니다.
        ##### 🚨 적용 사내 규정 및 대응 지침
        - **규정명/코드**: 듀레이션 갭 관리 규정 (Article 75-1)
        - **대응 조치 권고(Action Plan)**: 채권 포트폴리오 듀레이션 갭 축소를 위한 국채선물 매도(Short) 규모 비례 확대 요망
        """

def stream_ai_briefing(total_pnl, els_pnl, bond_pnl, top_driver, var_pct, kg_context):
    """
    수학 엔진의 정량적 결과와 지식그래프의 온톨로지 지식을 결합하여
    gemini-2.5-flash 모델에게 경영진 브리핑 스트리밍 출력을 요청하는 제너레이터 함수
    """
    model = genai.GenerativeModel('gemini-2.5-flash')

    prompt = f"""
    너는 금융기관 최고리스크책임자(CRO)를 보좌하는 '수석 리스크 AI 참모'야.
    아래 [정량적 데이터]와 사내 지식그래프(Neo4j)에서 추출한 [온톨로지 규정 및 논리]를 바탕으로
    경영진을 위한 리스크 원인 규명 및 조치 권고 브리핑 리포트를 작성해.

    [정량적 데이터 (수학 엔진 산출 팩트)]
    - 전사 일간 통합 P&L: {total_pnl / 100000000:,.1f}억 원
    - ELS 자체 헤지북 P&L: {els_pnl / 100000000:,.1f}억 원 평가손실
    - 채권 운용북 P&L: {bond_pnl / 100000000:,.1f}억 원 평가손실
    - 금일 가장 크게 변동한 핵심 리스크 동인: {top_driver}
    - 전사 VaR 한도 소진율: {var_pct:.1f}% 도달 (⚠️ 경고 수준)

    [온톨로지 규정 및 논리 (지식그래프 기반 데이터 리니지)]
    {kg_context}

    [작성 가이드 (필독)]
    1. 불필요한 인사말("안녕하세요" 등) 없이 바로 핵심 브리핑을 시작할 것.
    2. 수학 엔진의 숫자 팩트와 지식그래프의 온톨로지 논리(Greeks, Logic)를 완벽하게 엮어서 인과관계를 설명할 것.
    3. 반드시 지식그래프에서 추출된 사내 규정(Article 등)과 구체적인 대응 액션 플랜(Action Plan)을 근거로 제시하며 권고할 것. 환각(Hallucination) 제로를 달성하라.
    4. 마크다운의 볼드체(**)와 글머리 기호(-)를 적절히 사용하여 경영진이 한눈에 파악할 수 있도록 3문단 이내로 가독성 높게 작성해.
    """

    try:
        # 스트리밍 활성화
        response = model.generate_content(prompt, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text
                time.sleep(0.01) # 스트리밍 시각적 효과를 위한 미세 딜레이
    except Exception as e:
        # [안전망] API 에러 시 시연 에러 방지용 메시지
        yield f"⚠️ AI 모델 통신 중 에러가 발생했습니다. (API 키 및 네트워크 확인 필요)\n\nError: {e}"

# --- 4. 엔진 모듈 (계속) ---
# [신규] GraphRAG 실행 모듈 2 (Neo4j 지식 추출 및 Gemini 프롬프트 전송)

@st.cache_data(ttl=3600) # 1시간마다 갱신 (DB 부하 방지)
def get_dynamic_risk_limits():
    """Neo4j 지식 그래프에서 부서별 최신 한도 금액을 동적으로 조회합니다."""
    # DB 연결 실패 시 사용할 기본 Fallback (기존 하드코딩 값)
    limits = {
        "전사 VaR 한도": 1500.0,
        "금리 민감도(Rho) 한도": 40.0,
        "변동성 민감도(Vega) 한도": 30.0
    }

    if not NEO4J_URI or not NEO4J_PASSWORD:
        return limits

    try:
        with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
            with driver.session() as session:
                # RiskLimit 노드에서 이름과 한도값을 가져오는 쿼리
                query = "MATCH (l:RiskLimit) RETURN l.name AS name, l.limit_value AS val"
                result = session.run(query)
                for record in result:
                    limits[record["name"]] = float(record["val"])
    except Exception:
        pass # 에러 시 Fallback 값 유지

    return limits

def get_compliance_graph_context(dept_code, usage_pct):
    """
    Neo4j에 접속하여 부서별 한도 소진율에 따른 사내 컴플라이언스 규정 및 액션 플랜을 가져옴.
    """
    kg_context = ""

    # DB 접속 정보가 없으면 즉시 Fallback 컨텍스트 반환
    if not NEO4J_URI or not NEO4J_PASSWORD:
        return _fallback_compliance_context(dept_code, usage_pct)

    try:
        # with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
            with driver.session() as session:
                # 부서 코드와 한도 소진율 조건(Threshold)을 매핑하는 Cypher
                query = """
                MATCH (d:Department {code: $dept_code})-[:MONITORS]->(l:RiskLimit)-[tp:TRIGGERS_POLICY]->(rule:ComplianceRule)
                WHERE tp.threshold CONTAINS $threshold_kw
                RETURN l.name AS limit_name, l.metric AS metric,
                       tp.threshold AS threshold, tp.action_plan AS action_plan,
                       rule.name AS rule_name, rule.code AS rule_code
                """
                # 소진율에 따른 키워드 매칭 로직
                th_kw = "100% 이상" if usage_pct >= 100 else "90% 이상" if usage_pct >= 90 else "정상"

                # 정상 상태일 때는 쿼리를 보내지 않고 조기 반환 (비용 절감)
                if th_kw == "정상": return "✅ 현재 한도 소진율은 정상 범위 내에 있으며, 추가적인 AI 처방이 필요하지 않습니다."

                result = session.run(query, dept_code=dept_code, threshold_kw=th_kw)

                records = list(result)
                if not records: return _fallback_compliance_context(dept_code, usage_pct)

                for record in records:
                    kg_context += f"##### 🚨 적용 사내 컴플라이언스 규정\n"
                    kg_context += f"- **규정명/코드**: {record['rule_name']} ({record['rule_code']})\n"
                    kg_context += f"- **위반 조건(Threshold)**: {record['threshold']}\n\n"
                    kg_context += f"##### 📝 AI 조치 처방 (Action Plan)\n"
                    kg_context += f"- **{record['metric']} 한도**: {record['action_plan']}\n"

    except Exception as e:
        # [안전망] DB 연결 실패 시 Fallback 컨텍스트 반환
        kg_context = f"⚠️ Neo4j DB 연결 실패 (Error: {str(e)[:50]}...). 사내 규정 온톨로지 정보를 불러오지 못했습니다.\n\n"
        kg_context += _fallback_compliance_context(dept_code, usage_pct)

    return kg_context

def _fallback_compliance_context(dept_code, usage_pct):
    """
    Neo4j 연결 실패 시 시연을 위해 반환하는 하드코딩된 온톨로지 컨텍스트
    """
    if usage_pct < 90: return "✅ 현재 한도 소진율은 정상 범위 내에 있으며, 추가적인 AI 처방이 필요하지 않습니다."

    if dept_code == 'ENTERPRISE':
        if usage_pct >= 100:
            return """
            ##### 🚨 적용 사내 컴플라이언스 규정
            - **규정명/코드**: 전사 한도 초과(Breach) 대응 (Article 12-2)
            ##### 📝 AI 조치 처방 (Action Plan)
            - **전사 VaR 한도**: 즉각적인 포지션 축소, CRO 대면 보고 및 위기상황 위원회 소집
            """
        else: # 90~100%
            return """
            ##### 🚨 적용 사내 컴플라이언스 규정
            - **규정명/코드**: 전사 한도 경고(Warning) 대응 (Article 12-1)
            ##### 📝 AI 조치 처방 (Action Plan)
            - **전사 VaR 한도**: 신규 리스크 테이킹 즉각 중지 및 포지션 동결
            """
    elif dept_code == 'BOND_DESK' and usage_pct >= 100:
        return """
        ##### 🚨 적용 사내 컴플라이언스 규정
        - **규정명/코드**: 채권 데스크 한도 초과 대응 (Article 75-2)
        ##### 📝 AI 조치 처방 (Action Plan)
        - **금리 민감도(Rho) 한도**: IRS 페이(Pay) 포지션 신규 구축 또는 장기채 즉시 매도하여 Rho 한도 이내로 축소
        """
    elif dept_code == 'ELS_DESK' and usage_pct >= 100:
        return """
        ##### 🚨 적용 사내 컴플라이언스 규정
        - **규정명/코드**: ELS 데스크 비선형 한도 초과 대응 (Article 14-1)
        ##### 📝 AI 조치 처방 (Action Plan)
        - **변동성 민감도(Vega) 한도**: 사내 규정 제14조에 의거, 신규 ELS 롤오버 중지 및 베가(Vega) 중립을 위한 헤지 비중 기계적 확대
        """
    return "✅ 현재 한도 소진율은 정상 범위 내에 있으며, 추가적인 AI 처방이 필요하지 않습니다."

def stream_ai_prescription(dept_name, usage_pct, metric_name, exposure_amt, limit_amt, kg_context):
    """
    수학 엔진의 한도 소진 결과와 지식그래프의 컴플라이언스 지식을 결합하여
    gemini-2.5-flash 모델에게 AI 처방전 스트리밍 출력을 요청하는 제너레이터 함수
    """
    model = genai.GenerativeModel('gemini-2.5-flash')

    prompt = f"""
    너는 금융기관 최고리스크책임자(CRO)를 보좌하는 '수석 리스크 AI 참모'야.
    아래 [정량적 한도 현황]과 사내 지식그래프(Neo4j)에서 추출한 [온톨로지 규정 및 조치 지침]을 바탕으로
    해당 부서를 위한 AI 조치 처방전(AI Prescription)을 작성해.

    [정량적 한도 현황 (수학 엔진 산출 팩트)]
    - 대상 부서: {dept_name}
    - 핵심 리스크 지표: {metric_name}
    - 현재 노출도: {exposure_amt:,.1f}억 원
    - 사내 컴플라이언스 한도: {limit_amt:,.1f}억 원
    - **한도 소진율: {usage_pct:.1f}% 도달**

    [온톨로지 규정 및 지침 (지식그래프 기반 데이터 리니지)]
    {kg_context}

    [작성 가이드 (필독)]
    1. 불필요한 인사말("안녕하세요" 등) 없이 바로 핵심 브리핑을 시작할 것.
    2. 수학 엔진의 숫자 팩트와 지식그래프의 온톨로지 논리(threshold, Logic)를 완벽하게 엮어서 인과관계를 설명할 것.
    3. 반드시 지식그래프에서 추출된 사내 규정(Article 등)과 구체적인 대응 액션 플랜(Action Plan)을 근거로 제시하며 권고할 것. 환각(Hallucination) 제로를 달성하라.
    4. 마크다운의 볼드체(**)와 글머리 기호(-)를 적절히 사용하여 경영진이 한눈에 파악할 수 있도록 3문단 이내로 가독성 높게 작성해.
    """

    try:
        # 스트리밍 활성화
        response = model.generate_content(prompt, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text
                time.sleep(0.01) # 스트리밍 시각적 효과를 위한 미세 딜레이
    except Exception as e:
        # [안전망] API 에러 시 시연 에러 방지용 메시지
        yield f"⚠️ AI 모델 통신 중 에러가 발생했습니다. (API 키 및 네트워크 확인 필요)\n\nError: {e}"

# --- 4. 엔진 모듈 (계속) ---
# [신규] AI 프롬프트 함수 추가

def generate_dynamic_scenario(user_input):
    """
    사용자의 자연어 시나리오를 받아 관련성을 판단하고, 
    유효한 경우에만 JSON 형태로 파라미터를 추출합니다.
    """
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = f"""
    너는 금융기관 최고경영진(Senior Management)에게 시장 상황을 보고하는 '수석 리스크 AI 참모'야.
    다음 [사용자 입력]을 분석해서 아래 지시사항에 따라 JSON 형태로만 응답해.

    [사용자 입력]
    {user_input}

    [지시사항]
    1. 먼저 이 입력이 거시 경제, 금융 시장, 또는 전사 리스크(포트폴리오, 손익 등)와 관련이 있는지 엄격하게 판단해. (예: 개인적인 일상, 날씨, 단순 장난 등은 관련 없음)
    2. 관련이 없다면 "is_relevant"를 false로 설정하고, "rag_summary"에 "⚠️ 입력하신 시나리오에서 전사 리스크 파급 경로와의 연관성을 찾지 못했습니다. 거시 경제나 금융 시장과 관련된 상황을 다시 입력해 주십시오."라고 작성해. (나머지 필드는 비워둠)
    3. 관련이 있다면 "is_relevant"를 true로 설정하고, 경영진 보고용 시황 요약(rag_summary), 인과관계(kg_logic), 시뮬레이션 파라미터(parameters)를 모두 작성해.

    [출력 JSON 구조]
    {{
        "is_relevant": true 혹은 false,
        "rag_summary": "시황 요약 텍스트 또는 거절 메시지",
        "kg_logic": "인과관계 추론 텍스트 (관련 없을 시 빈 문자열)",
        "parameters": [
            {{"factor": "국채/회사채 금리", "current": "Base Rate", "target": "+100 bp", "duration": "14일"}},
            {{"factor": "KOSPI 200 지수", "current": "100%", "target": "75%", "duration": "14일"}},
            {{"factor": "삼성전자 주가", "current": "100%", "target": "55%", "duration": "14일"}}
        ]
    }}
    """
    
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json"
        )
    )
    
    return json.loads(response.text)

def stream_scenario_response(total_pnl, scenario_df_json):
    """
    시뮬레이션 완료 후 최종 손익과 시나리오 파라미터를 바탕으로
    사내 규정 기반 대응 방안을 스트리밍합니다.
    """
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = f"""
    너는 금융기관 최고리스크책임자(CRO)를 보좌하는 '수석 리스크 AI 참모'야.
    방금 매크로 위기 시나리오 시뮬레이션이 종료되었어.
    아래 [시뮬레이션 결과]를 바탕으로 사내 규정에 기반한 임시 조치 가이드를 작성해.

    [시뮬레이션 최종 결과]
    - 적용된 시나리오 파라미터: {scenario_df_json}
    - 전사 통합 평가 손익 (최대 손실): {total_pnl / 100000000:,.1f}억 원

    [작성 가이드 (필독)]
    1. 불필요한 인사말 없이 바로 브리핑을 시작할 것.
    2. 가장 치명적인 영향을 준 리스크 팩터를 지목하고 그 이유(비선형 리스크 확대 등)를 설명할 것.
    3. 사내 리스크 관리 규정(가상의 Article 번호 포함)을 근거로 ELS 운용 데스크와 채권 운용 데스크가 즉각 취해야 할 액션 플랜을 3문단 이내로 제시할 것.
    """
    
    try:
        response = model.generate_content(prompt, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text
                time.sleep(0.01)
    except Exception as e:
        yield f"⚠️ AI 모델 통신 중 에러가 발생했습니다.\n\nError: {e}"

def save_stress_test_to_kg(scenario_json, total_pnl, prescription):
    """
    스트레스 테스트 결과를 Neo4j에 'StressTestEvent' 노드로 저장합니다.
    """
    # DB 연결 정보가 없으면 조용히 넘어감
    if not NEO4J_URI or not NEO4J_PASSWORD:
        return False, "Neo4j 접속 정보가 없어 저장할 수 없습니다. (Fallback)"
        
    try:
        with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
            with driver.session() as session:
                # StressTestEvent라는 새로운 노드를 생성하고 데이터를 속성으로 저장하는 Cypher 쿼리
                query = """
                CREATE (e:StressTestEvent {
                    timestamp: datetime(),
                    scenario: $scenario,
                    total_pnl: $pnl,
                    prescription: $prescription
                })
                RETURN e
                """
                session.run(query, scenario=str(scenario_json), pnl=float(total_pnl), prescription=str(prescription))
        return True, "성공적으로 지식 그래프에 적재되었습니다."
    except Exception as e:
        return False, f"저장 중 에러 발생: {e}"


# --- 5. 실시간 연산 (T-1 vs T) ---
# 어제(Day 28) 대비 오늘(Day 29)의 일간 변화를 측정
base_mkt_state = df_market_data.iloc[-2].to_dict()
current_mkt_state = df_market_data.iloc[-1].to_dict()

bonds_res_today = revalue_bonds_multi(df_bonds, current_mkt_state, base_mkt_state)
els_res_today = revalue_els_multi(df_els, current_mkt_state, base_mkt_state)

daily_bond_pnl = bonds_res_today['pnl'].sum()
daily_els_pnl = els_res_today['pnl'].sum()
daily_total_pnl = daily_bond_pnl + daily_els_pnl

# 오늘 기준 VaR (99%) 산출
var_amount, var_sens = calculate_parametric_var(df_bonds, df_els, df_market_data, 0.99)
var_limit = 1500000000  # 가상의 통합 VaR 한도 (1500억)
var_usage_pct = (var_amount / var_limit) * 100

# 어제 기준 VaR (증감량 표시용)
df_mkt_yesterday = df_market_data.iloc[:-1]
var_amount_y, _ = calculate_parametric_var(df_bonds, df_els, df_mkt_yesterday, 0.99)
var_change = var_amount - var_amount_y

# ==========================================
# [페이지 1] 전사 리스크 브리핑 & 한도 관리
# ==========================================
if main_menu == "1. 전사 리스크 대시보드" and sub_menu == "1-1. 전사 마켓 리스크 브리핑":
    st.subheader("전사 마켓 리스크 현황 및 AI 원인 규명 브리핑")
    st.markdown("---")

    # Top Tier: 핵심 지표 (실시간 연산 데이터 바인딩)
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    # P&L 포맷팅 (단위: 억 원)
    pnl_str = f"{daily_total_pnl / 100000000:,.1f}억 원"

    # VaR 포맷팅 (단위: 억 원)
    var_str = f"{var_amount / 100000000:,.1f}억 원"
    var_delta_str = f"{var_change / 100000000:+,.1f}억"

    col_m1.metric("전사 통합 P&L (일간)", pnl_str, f"{daily_total_pnl / 100000000:+,.1f}억")
    col_m2.metric("전사 통합 VaR (99%, 1D)", var_str, var_delta_str, delta_color="inverse")
    col_m3.metric("리스크 한도 소진율", f"{var_usage_pct:.1f}%", f"{(var_usage_pct - (var_amount_y/var_limit)*100):+.1f}%p", delta_color="inverse")

    # 가장 민감도가 높은 리스크 동인 추출 (VaR Sensitivities 기반)
    top_risk_driver = var_sens.abs().idxmax()
    col_m4.metric("주요 리스크 동인", f"{top_risk_driver}", "모니터링 강화")

    st.markdown("<br>", unsafe_allow_html=True)

    # Middle Tier: 전사 포트폴리오 요약 및 드릴다운
    st.markdown("#### 📊 전사 포트폴리오 마켓 리스크 노출도 (Drill-down)")
    st.caption("※ 실시간 프라이싱 엔진 연산 결과입니다. ELS 포트폴리오는 기초자산 자체 헤지(Delta-Neutral)가 완료된 Net 운용북 기준입니다.")

    # 시연용 그릭스 (ELS 델타는 헤지 완료로 0에 수렴함을 표현)
    df_summary = pd.DataFrame({
        "포트폴리오 (Depth 1)": ["Bond_Portfolio (매수 10종목)", "ELS 자체 헤지 운용북 (매도 10종목)"],
        "Delta (델타 노출도)": [f"{var_sens.filter(like='KTB').sum() / 100000000:,.1f}억", "0.0억 (헤지 완료)"],
        "Gamma (감마 리스크)": ["-", "-14.5억 (Short Gamma)"],
        "Vega (베가 리스크)": ["-", f"{var_sens.filter(like='Vol').sum() / 100000000:,.1f}억 (Short Vega)"],
        "Rho (금리 민감도)": [f"{var_sens.filter(like='KTB').sum() / 100000000:,.1f}억", "-"],
        "운용북 Net P&L (일간)": [f"{daily_bond_pnl / 100000000:,.1f}억", f"{daily_els_pnl / 100000000:,.1f}억"]
    })

    st.dataframe(df_summary, use_container_width=True, hide_index=True)

    col_ex1, col_ex2 = st.columns(2)
    with col_ex1:
        with st.expander("📂 채권 포트폴리오 세부 10종목 보기 (Depth 2)"):
            st.dataframe(bonds_res_today[['name', 'curve', 'old_price', 'new_price', 'pnl']].style.format({"old_price": "{:,.0f}", "new_price": "{:,.0f}", "pnl": "{:,.0f}"}), use_container_width=True, hide_index=True)
    with col_ex2:
        with st.expander("📂 ELS 포트폴리오 세부 10종목 보기 (Depth 2)"):
            st.dataframe(els_res_today[['name', 'asset1', 'asset2', 'old_price', 'new_price', 'pnl']].style.format({"old_price": "{:,.0f}", "new_price": "{:,.0f}", "pnl": "{:,.0f}"}), use_container_width=True, hide_index=True)

    # 30일 추이 보여주기
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📈 최근 20일 포트폴리오 누적 손익 및 VaR 추이")
    st.caption("※ 과거 데이터(체결강도, 변동성, 금리)를 프라이싱 엔진에 통과시켜 산출한 누적 P&L과 1-Day VaR(99%) 궤적입니다.")

    # 1. 시계열 데이터 연산 (캐싱으로 속도 최적화)
    @st.cache_data
    def calculate_historical_trend(df_b, df_e, df_mkt):
        # 기관 포트폴리오 현실성 및 시각적 밸런스를 위해 채권 수량(Scale) 100배 보정
        df_b_scaled = df_b.copy()
        df_b_scaled['qty'] = df_b_scaled['qty'] * 100

        base_mkt_hist = df_mkt.iloc[0].to_dict() # Day 0 기준 시점
        history = []

        # 공분산(Covariance) 계산을 위한 최소 기간 10일을 제외하고 최근 20일만 시뮬레이션
        for i in range(10, len(df_mkt)):
            curr_mkt = df_mkt.iloc[i].to_dict()
            date_val = curr_mkt['Date']

            # 누적 P&L 연산
            b_res = revalue_bonds_multi(df_b_scaled, curr_mkt, base_mkt_hist)
            e_res = revalue_els_multi(df_e, curr_mkt, base_mkt_hist)

            bond_pnl = b_res['pnl'].sum() / 100000000  # 억 원 단위 변환
            els_pnl = e_res['pnl'].sum() / 100000000
            total_pnl = bond_pnl + els_pnl

            # 시점별 Parametric VaR 연산 (과거 ~ 해당 시점까지의 시장 데이터 사용)
            var_amt, _ = calculate_parametric_var(df_b_scaled, df_e, df_mkt.iloc[:i+1], confidence_level=0.99)
            var_amt_bn = var_amt / 100000000

            history.append({
                'Date': date_val,
                'Bond_PnL': bond_pnl,
                'ELS_PnL': els_pnl,
                'Total_PnL': total_pnl,
                'VaR_99': -var_amt_bn, # P&L 차트와 비교하기 위해 음수로 변환
                'Market_Vol': curr_mkt['Vol_KOSPI200'] * 100
            })
        return pd.DataFrame(history)

    df_trend = calculate_historical_trend(df_bonds, df_els, df_market_data)

    # 2. Plotly 이중 축(Dual-Axis) 차트 생성
    from plotly.subplots import make_subplots

    fig_trend = make_subplots(specs=[[{"secondary_y": True}]])

    # 막대그래프: 채권과 ELS의 개별 누적 P&L
    fig_trend.add_trace(go.Bar(x=df_trend['Date'], y=df_trend['Bond_PnL'], name='채권 누적 P&L (Scaled)', marker_color='#87CEFA', opacity=0.8), secondary_y=False)
    fig_trend.add_trace(go.Bar(x=df_trend['Date'], y=df_trend['ELS_PnL'], name='ELS 누적 P&L', marker_color='#FFA07A', opacity=0.8), secondary_y=False)

    # 꺾은선 그래프: 전사 통합 누적 P&L
    fig_trend.add_trace(go.Scatter(x=df_trend['Date'], y=df_trend['Total_PnL'], mode='lines+markers', name='전사 통합 누적 P&L', line=dict(color='#DC143C', width=3)), secondary_y=False)

    # 면적 꺾은선: VaR 99% 한계선 (리스크 경계)
    fig_trend.add_trace(go.Scatter(x=df_trend['Date'], y=df_trend['VaR_99'], mode='lines', name='VaR (99%, 1D) 하한선', fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.05)', line=dict(color='red', width=2, dash='dash')), secondary_y=False)

    # 보조 축 꺾은선: 시장 변동성 (주요 리스크 동인)
    fig_trend.add_trace(go.Scatter(x=df_trend['Date'], y=df_trend['Market_Vol'], mode='lines', name='KOSPI 변동성(우축)', line=dict(color='#808080', width=2, dash='dot')), secondary_y=True)

    # 차트 레이아웃 디자인
    fig_trend.update_layout(
        barmode='relative',
        hovermode='x unified',
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_trend.update_yaxes(title_text="평가 손익 및 VaR (억 원)", secondary_y=False)
    fig_trend.update_yaxes(title_text="내재 변동성 (%)", showgrid=False, secondary_y=True)

    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("---")

    # Bottom Tier: AI 지식 그래프 & 서술형 리포트
    st.markdown("---")
    st.markdown("#### 📝 AI 참모 원인 규명 및 대응 권고 리포트 (GraphRAG 기반)")
    st.caption("수학 엔진의 정량적 연산 결과(Fact)와 사내 지식 그래프(Neo4j Aura)의 온톨로지 지식(사내 규정 및 인과관계)을 결합하여 gemini-2.5-flash 모델이 실시간으로 경영진 보고서를 작성합니다.")

    # UI 레이아웃 분리 (리포트 텍스트 vs 리니지 맵)
    col_b1, col_b2 = st.columns([1.5, 1.5])

    with col_b1:
        # AI 브리핑 생성 버튼
        if st.button("✨ 경영진 보고용 실시간 AI 브리핑 생성 (gemini-2.5-flash)", type="primary", use_container_width=True):
            with st.spinner("지식 그래프 쿼리 및 AI 참모 추론 중..."):
                # 1. Neo4j 지식 그래프에서 사내 규정 컨텍스트 조회
                kg_context = get_knowledge_graph_context(top_risk_driver)

                # 2. UI 컨테이너 스타일링 (경고 느낌의 박스)
                with st.container(border=True):
                    st.markdown(f"**[Warning] 전사 리스크 한도 소진율 {var_usage_pct:.1f}% 도달**", help="사내 리스크 관리 규정에 의거하여 경영진 즉시 보고가 필요한 단계입니다.")

                    # 3. Streamlit Native 스트리밍 함수를 통해 Gemini 답변 타이핑 효과 연출
                    st.write_stream(
                        stream_ai_briefing(
                            daily_total_pnl,
                            daily_els_pnl,
                            daily_bond_pnl,
                            top_risk_driver,
                            var_usage_pct,
                            kg_context
                        )
                    )
        else:
            # 버튼을 누르기 전 대기 화면
            st.info("👆 상단의 'AI 브리핑 생성' 버튼을 클릭하면 수학 엔진의 산출 결과와 사내 지식 그래프(Neo4j)를 결합하여 AI 참모가 실시간 보고서를 작성합니다.")

    with col_b2:
        st.markdown("#### 🕸️ 리스크 파급 인과관계 맵 (Knowledge Graph)")
        st.caption("수학적 팩트 기반 데이터 리니지 (위험 요인 ➡ 자산군 ➡ P&L)")

        # [업데이트] 더미 데이터가 아닌 실제 엔진/온톨로지 데이터를 기반으로 노드/엣지 생성
        net_rca = Network(height='400px', width='100%', bgcolor='#ffffff', font_color='black')

        # 1. Macro 노드 (top_risk_driver 동적 반영)
        macro_label = f"{top_risk_driver}\n변동성/충격 확대"
        net_rca.add_node("Macro_Driver", label=macro_label, color="#ffb3b3", size=25, shape="dot")

        # 2. Port 노드 (Scaled P&L 동적 반영)
        port_bond_label = f"Bond_Portfolio\n({daily_bond_pnl / 100000000:,.0f}억)"
        port_els_label = f"ELS 자체헤지북\n({daily_els_pnl / 100000000:,.0f}억)"
        net_rca.add_node("Port_Bond", label=port_bond_label, color="#cce5ff", size=20, shape="database")
        net_rca.add_node("Port_ELS", label=port_els_label, color="#cce5ff", size=25, shape="database")

        # 3. PnL 노드 (Total PnL 동적 반영)
        pnl_label = f"전사 일간 P&L\n({daily_total_pnl / 100000000:,.0f}억)"
        net_rca.add_node("PnL", label=pnl_label, color="#ff9999", size=35, shape="star")

        # 4. 엣지 연결 및 라벨링 (온톨로지 기반)
        net_rca.add_edge("Macro_Driver", "Port_Bond", label="금리/Rho 상승", arrows="to", color="#cccccc")
        net_rca.add_edge("Macro_Driver", "Port_ELS", label="변동성/Vega 폭등", arrows="to", color="#cccccc")

        # 손실 규모에 따라 엣지 굵기 조절 (가독성 증대)
        net_rca.add_edge("Port_Bond", "PnL", arrows="to", color="#ff9999", width=abs(daily_bond_pnl/daily_total_pnl)*5 if daily_total_pnl !=0 else 1)
        net_rca.add_edge("Port_ELS", "PnL", label="주요 원인", arrows="to", color="#ff9999", width=abs(daily_els_pnl/daily_total_pnl)*10 if daily_total_pnl !=0 else 2)

        net_rca.set_options('{"physics": {"solver": "forceAtlas2Based"}, "edges": {"font": {"size": 11, "color": "#555555"}}}')
        net_rca.write_html("kg_rca.html")
        with open("kg_rca.html", 'r', encoding='utf-8') as f:
            components.html(f.read(), height=420)

# ==========================================
# [페이지 1-2] 부서별 한도 관리 및 처방
# ==========================================
elif main_menu == "1. 전사 리스크 대시보드" and sub_menu == "1-2. 부서별 한도 관리 및 처방":
    # --- [추가] Neo4j에서 동적 한도 불러오기 ---
    dynamic_limits = get_dynamic_risk_limits()

    st.subheader("🚨 부서별 리스크 한도(Limits) 실시간 모니터링")
    st.caption("프라이싱 엔진에서 산출된 실시간 그릭스(Greeks) 및 VaR를 사내 컴플라이언스 한도와 대조합니다.")
    st.markdown("---")

    # --- 1. Top Tier: 실시간 한도 모니터링 표 ---
    st.markdown("##### 1. 실시간 한도 모니터링 (Top Tier)")
    col_t1, col_t2, col_t3 = st.columns(3)

    # 앞서 계산한 엔진 결과값을 그대로 활용합니다!
    # var_usage_pct (전사 VaR 소진율), daily_bond_pnl (채권 P&L), daily_els_pnl (ELS P&L), var_sens (그릭스)
    # var_amount (전사 VaR 총액), rho_exposure (채권 Rho 총액), vega_exposure (ELS Vega 총액)

    with col_t1:
        st.markdown("##### 🏢 전사 통합 한도")
        with st.container(border=True):
            # 하드코딩 대신 동적 변수 사용
            var_limit_bn = dynamic_limits.get("전사 VaR 한도", 1500.0)
            st.markdown(f"**Parametric VaR 한도 (99%, 1D)**: {var_limit_bn:,.0f}억")
            # st.markdown("**Parametric VaR 한도 (99%, 1D)**: 1,500억")

            if var_usage_pct >= 100:
                st.progress(1.0, text=f"🚨 {var_usage_pct:.1f}% (한도 초과)")
                st.error("Action: 즉각적인 포지션 축소 및 위기상황 보고 요망")
            elif var_usage_pct >= 90:
                st.progress(var_usage_pct / 100.0, text=f"⚠️ {var_usage_pct:.1f}% (경고)")
                st.warning("Action: 신규 리스크 테이킹 중지")
            else:
                st.progress(var_usage_pct / 100.0, text=f"✅ {var_usage_pct:.1f}% (정상)")
                st.success("Action: 특이사항 없음")

    with col_t2:
        st.markdown("##### 📈 채권 운용 데스크")
        with st.container(border=True):
            rho_exposure = var_sens.filter(like='KTB').sum()
            rho_exposure_bn = rho_exposure / 100000000 # 억 원
            # rho_limit_bn = 40.0 # 억 원
            rho_limit_bn = dynamic_limits.get("금리 민감도(Rho) 한도", 40.0)
            rho_usage = abs(rho_exposure_bn / rho_limit_bn) * 100

            # st.markdown("**Rho (금리 민감도) 한도**: 40억 / 1bp")
            st.markdown(f"**Rho (금리 민감도) 한도**: {rho_limit_bn:,.0f}억 / 1bp")
            if rho_usage >= 100:
                st.progress(1.0, text=f"🚨 {rho_usage:.1f}% (한도 초과)")
            else:
                st.progress(min(rho_usage / 100.0, 1.0), text=f"✅ {rho_usage:.1f}% (정상)")
                st.success("Action: 특이사항 없음")

            st.markdown(f"**현재 누적 P&L**: {daily_bond_pnl / 100000000:,.1f}억 원")

    with col_t3:
        st.markdown("##### 📉 ELS 운용 데스크 (자체 헤지북)")
        with st.container(border=True):
            vega_exposure = var_sens.filter(like='Vol').sum()
            vega_exposure_bn = vega_exposure / 100000000 # 억 원
            # vega_limit_bn = 30.0 # 억 원
            vega_limit_bn = dynamic_limits.get("변동성 민감도(Vega) 한도", 30.0)
            vega_usage = abs(vega_exposure_bn / vega_limit_bn) * 100

            # st.markdown("**Vega (변동성 민감도) 한도**: 30억 / 1%p")
            st.markdown(f"**Vega (변동성 민감도) 한도**: {vega_limit_bn:,.0f}억 / 1%p")
            if vega_usage >= 100:
                st.progress(1.0, text=f"🚨 {vega_usage:.1f}% (한도 초과)")
            else:
                st.progress(min(vega_usage / 100.0, 1.0), text=f"⚠️ {vega_usage:.1f}% (초과 임박)")

            st.warning("Action: 비선형 리스크 확대 구간 진입")
            st.markdown(f"**현재 Net P&L (헤지비용 포함)**: {daily_els_pnl / 100000000:,.1f}억 원")

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("💡 **시스템 알림**: 한도 모니터링은 장중 실시간 매매 데이터 및 기초자산 가격을 반영하여 1시간 주기로 갱신됩니다. 현재 전사 VaR 한도가 사내 리스크 관리 규정(제12조)의 임계치를 상회하고 있습니다.")
    st.markdown("---")

    # --- 2. Bottom Tier: AI 한도 처방 리포트 (GraphRAG 기반) ---
    st.markdown("#### 📝 AI 참모 실시간 한도 처방 리포트")
    st.caption("수학 엔진의 정량적 소진 결과(Fact)와 사내 지식 그래프(Neo4j Aura)의 온톨로지 지식(사내 규정 및 조치 지침)을 결합하여 Gemini 1.5 Pro 모델이 실시간으로 처방전을 작성합니다.")

    with st.container(border=True):
        # 레이아웃 분리 없이 넓고 시원하게 구성
        selected_dept = st.selectbox("처방전을 생성할 부서를 선택하십시오.", ["전사 통합 리스크 위원회", "채권 운용 데스크", "ELS 운용 데스크"], index=0)

        # 선택된 부서에 맞는 데이터 매핑
        if selected_dept == "전사 통합 리스크 위원회":
            dept_code, dept_name, usage_pct, metric_name, exposure_amt, limit_amt = 'ENTERPRISE', selected_dept, var_usage_pct, '전사 VaR 한도', var_amount / 100000000, 1500.0
        elif selected_dept == "채권 운용 데스크":
            dept_code, dept_name, usage_pct, metric_name, exposure_amt, limit_amt = 'BOND_DESK', selected_dept, rho_usage, '금리 민감도(Rho) 한도', rho_exposure_bn, rho_limit_bn
        else: # ELS 운용 데스크
            dept_code, dept_name, usage_pct, metric_name, exposure_amt, limit_amt = 'ELS_DESK', selected_dept, vega_usage, '변동성 민감도(Vega) 한도', vega_exposure_bn, vega_limit_bn

        # AI 처방 생성 버튼
        if st.button(f"✨ {dept_name} AI 한도 처방 생성", type="primary", use_container_width=True):
            with st.spinner(f"{dept_name} 컴플라이언스 규정 검토 및 AI 참모 추론 중..."):
                # 1. Neo4j 지식 그래프에서 사내 규정 컨텍스트 조회
                kg_context = get_compliance_graph_context(dept_code, usage_pct)

                st.markdown("---")
                st.markdown(f"##### 🤖 AI 참모의 {dept_name} 처방전")

                # 2. Streamlit Native 스트리밍 함수를 통해 Gemini 답변 타이핑 효과 연출
                st.write_stream(
                    stream_ai_prescription(
                        dept_name,
                        usage_pct,
                        metric_name,
                        exposure_amt,
                        limit_amt,
                        kg_context
                    )
                )
        else:
            # 버튼을 누르기 전 대기 화면
            st.info(f"👆 위 셀렉트박스에서 부서를 선택한 후 버튼을 클릭하면, AI 참모가 사내 규정을 검토하여 실시간으로 {dept_name}을 위한 처방전을 작성합니다.")


# ==========================================
# [페이지 2] 스트레스 테스트 데스크
# ==========================================
# ==========================================
# [페이지 2-1] 순방향 WHAT-IF 시뮬레이션
# ==========================================
elif main_menu == "2. 스트레스 테스트 데스크" and sub_menu == "2-1. 순방향 WHAT-IF 시뮬레이션":
    st.subheader("▶ 매크로 위기 시나리오 분석 (Forward WHAT-IF)")
    st.markdown("---")

    col_left, col_right = st.columns([1.2, 1.8])

    with col_left:
        st.markdown("#### a. 시나리오 설정 (자연어 지시)")
        scenario_prompt = st.text_area(
            "매크로 위기 시나리오 입력:",
            value="전쟁 확전으로 인해 유가가 급등하고 인플레이션 우려로 금리가 오르며 반도체 섹터와 아시아 증시가 장기 침체될 것 같다. 회사 손익과 포트폴리오에 미치는 영향을 분석해 보자."
        )
        
        if st.button("AI 시나리오 파라미터 기안 생성", key="btn_fw", use_container_width=True):
            with st.spinner("경영진 보고용 시나리오 파라미터를 추론 중입니다..."):
                result = generate_dynamic_scenario(scenario_prompt)
                
                # AI가 금융 리스크와 관련이 있다고 판단한 경우만 다음 스텝으로 진행
                if result.get("is_relevant", False):
                    st.session_state.scenario_data = result
                    st.session_state.scenario_step = 1
                else:
                    # 관련이 없으면 스텝을 초기화하고 AI의 거절 메시지 출력
                    st.session_state.scenario_step = 0
                    st.warning(result.get("rag_summary", "회사 리스크 파급 경로와 연관성을 찾지 못했습니다. 다시 입력해 주세요."))

        # scenario_step이 1 이상일 때만 아래 결과를 보여줌
        if st.session_state.scenario_step >= 1 and 'scenario_data' in st.session_state:
            s_data = st.session_state.scenario_data
            
            st.markdown("---")
            st.markdown("#### b. AI 시나리오 추론 및 근거 (동적 생성)")

            with st.container(border=True):
                st.markdown("**🔍 AI 시황 분석 요약**")
                st.write(s_data.get('rag_summary', ''))

                st.markdown("**🕸️ 지식 그래프 인과관계 변환**")
                st.markdown(s_data.get('kg_logic', ''))
                st.success("💡 위 추론을 바탕으로 시스템 시뮬레이션을 위한 최적의 파라미터를 도출했습니다.")

            st.markdown("---")
            st.markdown("#### c. 시나리오 파라미터 검토 및 승인")
            st.info("AI가 도출한 리스크 팩터별 최대 충격량과 도달 기간을 검토하고, 필요시 표에서 직접 수정(Edit)한 후 승인하십시오.")

            # 파라미터가 정상적으로 들어왔을 경우 데이터프레임 렌더링
            if 'parameters' in s_data and s_data['parameters']:
                df_params = pd.DataFrame(s_data['parameters'])
                df_params.columns = ["리스크 팩터", "현재 수준", "최대 충격 (Target)", "충격 도달 기간"]
                
                edited_df = st.data_editor(df_params, use_container_width=True, hide_index=True)

                if st.button("✅ 기안 승인 및 Full Revaluation 실행", key="btn_fw_run", type="primary", use_container_width=True):
                    st.session_state.final_scenario_df = edited_df
                    st.session_state.scenario_step = 2

                    # --- [추가된 부분] 새로운 시뮬레이션을 위해 이전 처방전 텍스트 초기화 ---
                    if 'generated_prescription' in st.session_state:
                        del st.session_state['generated_prescription']

    with col_right:
        if st.session_state.scenario_step == 2:
            st.markdown("#### d. 시계열 파급 분석 (Time-Step Full Revaluation)")

            chart_placeholder = st.empty()
            metrics_placeholder = st.empty()
            status_text = st.empty()

            # --- [동적 파싱] 왼쪽 표에서 데이터 추출 및 궤적 생성 ---
            df = st.session_state.final_scenario_df
            
            # 텍스트에서 마지막 숫자를 추출하는 헬퍼 함수 ("25% 하락(75%)" -> 75.0)
            def get_target_num(text, default=100.0):
                nums = re.findall(r'[-+]?\d*\.?\d+', str(text))
                return float(nums[-1]) if nums else default

            # 팩터별 목표 수치 초기화
            rate_target = 0.0
            kospi_target = 100.0
            samsung_target = 100.0
            duration_days = 14
            
            for _, row in df.iterrows():
                factor = row["리스크 팩터"]
                target_val = get_target_num(row["최대 충격 (Target)"])
                days_val = int(get_target_num(row["충격 도달 기간"], 14))
                
                if "금리" in factor: rate_target = target_val
                elif "KOSPI" in factor: kospi_target = target_val
                elif "삼성전자" in factor: samsung_target = target_val
                duration_days = max(duration_days, days_val) # 가장 긴 기간 기준

            # 동적 궤적(Array) 생성 (7단계 스텝으로 쪼개기)
            steps_count = 7
            traj_x = np.linspace(100, kospi_target, steps_count).tolist()
            traj_y = np.linspace(100, samsung_target, steps_count).tolist()
            rate_shocks = np.linspace(0, rate_target, steps_count).tolist()
            
            # Z축(차트 높이)은 X, Y 하락분에 비례해서 대략적으로 떨어지도록 동적 연산
            traj_z = [100 - ((100 - x) * 1.5 + (100 - y) * 1.5) for x, y in zip(traj_x, traj_y)]
            
            # 동적 시간 라벨 생성
            time_labels = [f"Day {int(d)}" for d in np.linspace(0, duration_days, steps_count)]
            time_labels[0] = "Day 0 (정상)"
            time_labels[-1] = f"Day {duration_days} (최대 손실)"

            # --- 엔진 연산 준비 ---
            # 3D 표면도(Surface) 생성을 위한 기본 데이터 (기존과 동일)
            x = np.linspace(40, 100, 40)
            y = np.linspace(40, 100, 40)
            X, Y = np.meshgrid(x, y)
            Z = 100 + (X - 100)*0.3 + (Y - 100)*0.3 - np.where(X < 80, (80 - X) * 1.5, 0) - np.where(Y < 60, (60 - Y) * 2.5, 0) - 35 * np.exp(-0.03 * ((X - 75)**2 + (Y - 55)**2))

            base_mkt_state = df_market_data.iloc[-1].to_dict()
            df_bonds_scaled = df_bonds.copy()
            df_bonds_scaled['qty'] = df_bonds_scaled['qty'] * 100

            final_total_pnl = 0 # 최종 P&L 저장용 변수

            # --- 동적 시뮬레이션 루프 실행 ---
            for i in range(steps_count):
                # 1. 차트 업데이트
                fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Blues', opacity=0.7)])
                fig.add_trace(go.Scatter3d(x=traj_x[:i+1], y=traj_y[:i+1], z=traj_z[:i+1], mode='lines', line=dict(color='orange', width=6), name='과거 궤적'))
                fig.add_trace(go.Scatter3d(x=[traj_x[i]], y=[traj_y[i]], z=[traj_z[i]], mode='markers', marker=dict(size=10, color='red'), name='현재 지표'))
                fig.update_layout(title=f"⏳ 진행 상태: {time_labels[i]}", scene=dict(xaxis_title='KOSPI 200 (%)', yaxis_title='삼성전자 (%)', zaxis_title='포트폴리오 가치', camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.2))), margin=dict(l=0, r=0, b=0, t=30), height=350)
                chart_placeholder.plotly_chart(fig, use_container_width=True)

                # 2. 다차원 리스크 팩터 동적 생성
                curr_mkt = base_mkt_state.copy()

                k_shock_pct = traj_x[i] / 100.0
                s_shock_pct = traj_y[i] / 100.0
                curr_mkt['KOSPI200_Close'] *= k_shock_pct
                curr_mkt['Samsung_Close'] *= s_shock_pct
                curr_mkt['SKHynix_Close'] *= k_shock_pct
                curr_mkt['Naver_Close'] *= k_shock_pct

                for tenor in ['KTB_6M', 'KTB_1Y', 'KTB_3Y', 'KTB_5Y', 'Corp_6M', 'Corp_1Y']:
                    curr_mkt[tenor] += (rate_shocks[i] / 100.0)

                vol_bump = (100 - traj_x[i]) * 0.005
                liq_drop = (100 - traj_x[i]) * 1.5

                for key in curr_mkt.keys():
                    if key.startswith('Vol_'): curr_mkt[key] += vol_bump
                    elif key.endswith('_Intensity'): curr_mkt[key] -= liq_drop

                # 3. 프라이싱 엔진 연산
                sim_bonds = revalue_bonds_multi(df_bonds_scaled, curr_mkt, base_mkt_state)
                sim_els = revalue_els_multi(df_els, curr_mkt, base_mkt_state)

                bond_pnl = sim_bonds['pnl'].sum()
                els_pnl = sim_els['pnl'].sum()
                final_total_pnl = bond_pnl + els_pnl

                # 4. 결과 출력
                with metrics_placeholder.container():
                    st.markdown(f"**실시간 통합 평가 손익 (Full Revaluation):** `<span style='color:red;'>{final_total_pnl/100000000:,.1f}억 원</span>`", unsafe_allow_html=True)
                    m1, m2 = st.columns(2)
                    m1.metric("채권 포트폴리오 P&L (Scaled)", f"{bond_pnl/100000000:,.1f}억")
                    m2.metric("ELS 자체 헤지북 Net P&L", f"{els_pnl/100000000:,.1f}억")

                status_text.warning(f"다차원 엔진 연산 중... 현재 단계: {time_labels[i]}")
                time.sleep(0.8)

            status_text.error(f"🚨 시뮬레이션 종료: {duration_days}일 차 최대 충격 구간에 도달했습니다.")

            # --- [동적 생성] AI 상황 판단 및 사내 규정 기반 대응 방안 ---
            st.markdown("---")
            st.markdown("#### e. AI 상황 판단 및 사내 규정 기반 대응 방안")

            with st.container(border=True):
                scenario_json_str = df.to_dict(orient="records")
                
                # 버튼 클릭 시마다 AI가 다시 스트리밍하는 것을 방지하기 위해 세션에 텍스트 저장
                if 'generated_prescription' not in st.session_state:
                    # AI 스트리밍 실행 후 완성된 전체 텍스트를 변수에 캡처!
                    full_text = st.write_stream(stream_scenario_response(final_total_pnl, scenario_json_str))
                    st.session_state.generated_prescription = full_text
                else:
                    # 이미 생성된 처방전이 있다면 스트리밍 없이 바로 텍스트 출력
                    st.markdown(st.session_state.generated_prescription)

            # --- [신규 추가] 지식 그래프(Neo4j) 아카이빙 버튼 ---
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("💾 이 분석 결과 및 AI 처방을 지식 그래프(Neo4j)에 아카이빙", type="primary", use_container_width=True):
                with st.spinner("Neo4j Aura DB에 스트레스 테스트 이력을 적재 중입니다..."):
                    success, msg = save_stress_test_to_kg(
                        scenario_json_str, 
                        final_total_pnl, 
                        st.session_state.generated_prescription
                    )
                    
                    if success:
                        st.success(f"✅ {msg}")
                        st.info("💡 향후 '장애 대응 가이드 에이전트'가 유사한 위기 상황 발생 시 이 이력을 참조하여 더 스마트한 처방을 내리게 됩니다.")
                    else:
                        st.error(f"⚠️ {msg}")


# ==========================================
# [페이지 2-2] 역방향 위기 좌표 탐색 (RST)
# ==========================================
elif main_menu == "2. 스트레스 테스트 데스크" and sub_menu == "2-2. 역방향 위기 좌표 탐색 (RST)":
    st.subheader("◀ 역방향: 위기 좌표 탐색 (Reverse Stress Test)")
    st.markdown("---")
    st.markdown("#### 1. 목표 손실(Target Loss) 기반 최단 위기 경로 탐색")
    st.caption("사전에 정의된 타겟 손실을 유발하는 최악의 다차원 리스크 팩터 조합(KOSPI, 삼성전자, 국채 금리)을 DML 엔진의 경사하강법으로 역산합니다.")

    col_input, col_empty = st.columns([1, 2])
    with col_input:
        target_loss_input = st.number_input("목표 손실액 (단위: 억 원, 예: -400):", min_value=-2000, max_value=-10, value=-400, step=10)
        if st.button("▶ 다차원 역탐색 애니메이션 실행", key="btn_rev_run", type="primary"):
            st.session_state.rst_step = 1

    if st.session_state.rst_step == 1:
        st.success("✅ DML 프록시 엔진 탐색 완료. 타겟 손실에 도달하는 최단 위기 좌표 궤적을 도출했습니다.")

        col_r1, col_r2 = st.columns(2)

        radar_placeholder = col_r1.empty()
        contour_placeholder = col_r2.empty()
        status_placeholder = st.empty()

        # --- 기준 상태 및 스케일링 세팅 ---
        base_mkt_state = df_market_data.iloc[-1].to_dict()
        df_bonds_scaled = df_bonds.copy()
        df_bonds_scaled['qty'] = df_bonds_scaled['qty'] * 100

        # --- 시뮬레이션 경로 사전 세팅 (Target Loss 도달 경로) ---
        steps = 10
        k_path = np.linspace(100, 70, steps) # KOSPI 100% -> 70%
        s_path = np.linspace(100, 60, steps) # 삼성전자 100% -> 60%
        r_path = np.linspace(0, 100, steps)  # 금리 0 -> +100bp

        # --- 배경 지형도(Contour) 데이터 사전 연산 ---
        grid_size = 15 # 렌더링 속도 최적화
        k_grid = np.linspace(50, 100, grid_size)
        s_grid = np.linspace(50, 100, grid_size)
        K_MESH, S_MESH = np.meshgrid(k_grid, s_grid)
        Z_PNL = np.zeros((grid_size, grid_size))

        # 충격 생성 헬퍼 함수
        def get_shocked_mkt(k_pct, s_pct, r_bp):
            mkt = base_mkt_state.copy()
            mkt['KOSPI200_Close'] *= (k_pct / 100.0)
            mkt['Samsung_Close'] *= (s_pct / 100.0)
            mkt['SKHynix_Close'] *= (k_pct / 100.0)
            mkt['Naver_Close'] *= (k_pct / 100.0)

            for tenor in ['KTB_6M', 'KTB_1Y', 'KTB_3Y', 'KTB_5Y', 'Corp_6M', 'Corp_1Y']:
                mkt[tenor] += (r_bp / 100.0) # 퍼센트 포인트 단위 상승

            vol_bump = (100 - k_pct) * 0.005
            liq_drop = (100 - k_pct) * 1.5
            for key in mkt.keys():
                if key.startswith('Vol_'): mkt[key] += vol_bump
                elif key.endswith('_Intensity'): mkt[key] -= liq_drop
            return mkt

        with st.spinner("손실 지형도(Contour Map) 생성 중..."):
            for i in range(grid_size):
                for j in range(grid_size):
                    # 임의의 연관 금리 충격 부여
                    temp_mkt = get_shocked_mkt(K_MESH[i, j], S_MESH[i, j], (100 - K_MESH[i, j]) * 1.5)
                    b_res = revalue_bonds_multi(df_bonds_scaled, temp_mkt, base_mkt_state)
                    e_res = revalue_els_multi(df_els, temp_mkt, base_mkt_state)
                    Z_PNL[i, j] = (b_res['pnl'].sum() + e_res['pnl'].sum()) / 100000000

        history_data = []

        # --- 애니메이션 루프 ---
        for step in range(steps):
            current_k = k_path[step]
            current_s = s_path[step]
            current_r = r_path[step]

            curr_mkt = get_shocked_mkt(current_k, current_s, current_r)
            sim_bonds = revalue_bonds_multi(df_bonds_scaled, curr_mkt, base_mkt_state)
            sim_els = revalue_els_multi(df_els, curr_mkt, base_mkt_state)

            step_bond_pnl = sim_bonds['pnl'].sum()
            step_els_pnl = sim_els['pnl'].sum()
            current_total_pnl = (step_bond_pnl + step_els_pnl) / 100000000

            # 현재 스텝 기록
            row_data = {
                "탐색 단계": f"Step {step+1}",
                "KOSPI 잔존가치": f"{current_k:.1f}%",
                "삼성전자 잔존가치": f"{current_s:.1f}%",
                "국채금리 충격": f"+{current_r:.0f} bp"
            }
            for _, r in sim_bonds.iterrows():
                row_data[f"B_{r['name']}"] = r['price_change'] * r['qty']
            for _, r in sim_els.iterrows():
                row_data[f"E_{r['name']}"] = r['price_change'] * r['qty']
            history_data.append(row_data)

            # [화면 1] 다차원 리스크 팽창 방사형 차트 (Radar)
            risk_k = max(0, (100 - current_k) / 50 * 100)
            risk_s = max(0, (100 - current_s) / 50 * 100)
            risk_r = max(0, current_r / 150 * 100)

            fig_radar = go.Figure(data=go.Scatterpolar(
                r=[risk_k, risk_s, risk_r, risk_k],
                theta=['KOSPI 하락 위험', '삼성전자 하락 위험', '국채금리 상승 위험', 'KOSPI 하락 위험'],
                fill='toself', fillcolor='rgba(255, 75, 75, 0.4)', line=dict(color='red', width=2)
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False, title=f"다차원 리스크 팩터 팽창 (Iteration {step+1})",
                height=420, margin=dict(l=40, r=40, t=50, b=40)
            )
            radar_placeholder.plotly_chart(fig_radar, use_container_width=True)

            # [화면 2] Top-2 팩터 투영 손실 지형도 (Contour)
            fig_contour = go.Figure(data=go.Contour(
                z=Z_PNL, x=k_grid, y=s_grid, colorscale='RdBu',
                contours=dict(showlabels=True, labelfont=dict(color='white'))
            ))
            fig_contour.add_trace(go.Scatter(
                x=k_path[:step+1], y=s_path[:step+1],
                mode='lines+markers', line=dict(color='#00FF00', width=4, dash='dot'),
                marker=dict(size=6, color='black'), name='탐색 궤적'
            ))
            fig_contour.add_trace(go.Scatter(
                x=[current_k], y=[current_s],
                mode='markers', marker=dict(size=14, color='yellow', symbol='star'),
                name='현재 탐색 좌표'
            ))
            fig_contour.update_layout(
                title=f"핵심 팩터(Top-2) 투영 손실 지형도<br>현재 추정 Net P&L: <span style='color:red;'>{current_total_pnl:,.0f}억 원</span>",
                xaxis_title="KOSPI 200 잔존가치 (%)", yaxis_title="삼성전자 잔존가치 (%)",
                height=420, margin=dict(l=30, r=30, t=60, b=30)
            )
            contour_placeholder.plotly_chart(fig_contour, use_container_width=True)

            time.sleep(0.5)

        # --- 탐색 완료 후 경영진 보고용 시사점 출력 ---
        status_placeholder.error(f'''
        **🚨 역위기상황 탐색 완료: 타겟 손실 도달 최악의 팩터 조합 산출**

        * **KOSPI 200 지수:** {100 - current_k:.1f}% 하락 시 ELS 포트폴리오 비선형 리스크 확대 구간 진입
        * **삼성전자 주가:** {100 - current_s:.1f}% 하락 (가장 가파른 손실 기울기 편미분 값을 형성하는 핵심 개별주식 동인)
        * **국채 금리:** {current_r:.0f}bp 상승 시 채권 운용북 포지션 한도 도달

        **[경영진 보고용 시사점]** 현재 전사 포트폴리오는 단일 팩터의 충격보다 **'대형주(삼성전자) 폭락'과 '금리 급등'이 결합된 복합 위기 상황**에서 자체 헤지북의 감마/베가 출혈 속도가 기하급수적으로 빨라집니다. 스트레스 테스트 시나리오 설정 시 팩터 간의 교차 민감도(Cross-Greeks)를 최우선으로 고려해야 합니다.
        ''')

        # --- 시계열 상세 데이터 테이블(Audit Trail) 표출 ---
        st.markdown("---")
        st.markdown("#### 3. 역위기상황 탐색 시계열 상세 데이터 (Audit Trail)")
        st.caption("각 탐색 단계별 리스크 팩터의 변화와 구성 상품 20종의 누적 가격 변화(평가손익, 단위: 원) 추적 내역입니다. 가로로 스크롤하여 전체 포트폴리오를 확인할 수 있습니다.")

        df_history = pd.DataFrame(history_data)

        for col in df_history.columns:
            if col not in ["탐색 단계", "KOSPI 잔존가치", "삼성전자 잔존가치", "국채금리 충격"]:
                df_history[col] = df_history[col].apply(lambda x: f"{x:,.0f}")

        st.dataframe(df_history, use_container_width=True, hide_index=True)


# ==========================================
# [페이지 3] 시스템 오퍼레이션 (자연어 배치 구동)
# ==========================================
elif main_menu == "3. 시스템 오퍼레이션":
    st.subheader("🤖 AI 기반 복합 배치(Batch) 파이프라인 기안 에이전트")
    st.markdown("단순 반복적인 매뉴얼 세팅과 휴먼 에러를 제거하고, **인력을 고부가가치 분석 업무로 재배치**하기 위한 지능형 오퍼레이션 시연입니다.")
    st.markdown("---")

    col1, col2 = st.columns([1.2, 1.8])

    # [좌측] 기존 방식 (AS-IS)
    with col1:
        st.header("📉 AS-IS: 레거시 매뉴얼 설정")
        st.caption("작업자가 모든 종속성과 파라미터를 수동으로 확인하고 세팅해야 함")

        with st.container(border=True):
            st.selectbox("1. 대상 포트폴리오 선택", ["선택하세요", "포트폴리오 A", "포트폴리오 B", "포트폴리오 C", "포트폴리오 D"])
            st.date_input("2. 시작일 설정")
            st.date_input("3. 종료일 설정")
            st.multiselect("4. 실행 엔진 선택", ["Market Data Fetch", "Pricing Engine", "VaR Engine", "Stress Test Engine", "Report Gen"])
            st.checkbox("5. 에러 발생 시 롤백 활성화")
            st.checkbox("6. 선행 작업(Data Fetch) 완료 대기")
            st.button("수동 배치 실행 (위험: 의존성 체크 누락 가능성)")

            st.error("⚠️ **문제점:** 월말 재배치 시 작업자가 10여 개의 화면을 오가며 세팅. 순서 실수 시 8시간짜리 야간 배치 실패 및 익일 보고 누락 발생.")

    # [우측] AI 에이전트 방식 (TO-BE)
    with col2:
        st.header("✨ TO-BE: AI 에이전트 오케스트레이션")
        st.caption("자연어 의도 파악 -> 의존성 분석 -> 파이프라인 자동 기안 -> 사용자 승인")

        user_input = st.text_area(
            "💬 오퍼레이션 지시 (자연어 입력):",
            value="26.01.15부터 26.02.14까지 포트폴리오 D에 대한 재배치를 돌리고, 각 날짜별 보고서 생성 작업도 다시 돌려줘."
        )

        if st.button("AI 파이프라인 기안 (Draft) 생성 🚀"):
            with st.spinner("자연어 분석 및 종속성(Dependency) 파악 중..."):
                time.sleep(1.2)
                st.session_state.batch_step = 1

        if st.session_state.batch_step == 1:
            st.success("✅ 시스템 파라미터 매핑 및 작업 종속성 검증 완료")

            with st.expander("🔍 AI 파라미터 추출 결과 (시스템 매핑)", expanded=True):
                st.code('''
                {
                  "intent": "RE-RUN_BATCH_AND_REPORT",
                  "target_portfolio": "PORT_D",
                  "date_range": {"start": "2026-01-15", "end": "2026-02-14"},
                  "required_engines": ["Pricing", "VaR", "Report_Gen"],
                  "days_count": 31
                }
                ''', language='json')

            st.markdown("### 📋 AI 기안: 자동 생성된 작업 파이프라인(DAG)")
            st.info('''
            💡 **AI 시스템 알림:** 요청하신 '보고서 생성'을 위해서는 필수 선행 작업이 필요합니다.
            AI가 시스템 종속성을 파악하여 **누락된 선행 엔진 구동을 자동으로 파이프라인에 추가**했습니다.
            ''')

            st.markdown("""
            1. 🔄 **[Data Fetch]** 26.01.15 ~ 26.02.14 구간 'PORT_D' 기초자산 및 시장 데이터 검증 (자동 추가됨)
            2. 🧮 **[Pricing Engine]** 31일치 개별 상품 민감도(Greeks) 재산출
            3. 📈 **[VaR Engine]** 31일치 포트폴리오 통합 VaR 및 P&L 재산출
            4. 📊 **[Report Gen]** 31개 일일 리스크 브리핑 보고서 PDF 생성 및 DB 아카이빙
            """)

            st.warning("⏱️ **예상 소요 시간:** 약 45분 / 🖥️ **시스템 부하 예상:** 중간(Medium)")

            st.markdown("---")
            st.markdown("#### 👨‍⚖️ 최종 승인 (Human-in-the-loop)")
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                if st.button("✅ 기안 승인 및 배치 실행"):
                    st.toast("배치 작업이 안전하게 스케줄링 되었습니다.")
            with col2_2:
                if st.button("❌ 기안 반려 (재입력)"):
                    st.session_state.batch_step = 0
                    st.rerun()

# ==========================================
# [페이지 4] 지능형 장애 대응 및 시스템 복구 에이전트")
# ==========================================
elif main_menu == "4. 장애 대응 가이드 에이전트":
    st.title("🛠️ AI 장애 대응 및 유지보수 에이전트")
    st.markdown("지식 그래프에 축적된 과거 장애 조치 이력과 시스템 로그를 분석하여, **1차 해결 가이드라인을 제공**하고 필요시 유지보수팀(L2)에 **컨텍스트 기반 호출**을 수행합니다.")
    st.markdown("---")

    col1, col2 = st.columns([1, 1.5])

    # [좌측] 에러 로그 모니터링 및 입력 창
    with col1:
        st.subheader("🚨 실시간 시스템 알림 (Alert)")

        # 가상의 에러 발생 상황 제시
        st.error("**[CRITICAL] 배치 작업 실패 (Pricing Engine)**\n\n발생 시간: 03:15 AM\n대상: Portfolio_ELS_Book")

        error_log_input = st.text_area(
            "상세 에러 로그 확인 및 분석 요청:",
            value="""[ERROR] MemoryError: Out of memory calculating Pricing Surface for 10,000 ELS instruments.
[TRACE] File "pricing_engine.py", line 428, in revalue_els_portfolio
[TRACE] File "numpy/core/numeric.py", line 330, in full
[INFO] Container memory limit (32GB) exceeded.""",
            height=150
        )

        if st.button("🔍 AI 원인 분석 및 해결 가이드 요청", type="primary"):
            st.session_state.incident_step = 1

    # [우측] AI 분석 및 가이드 제공 영역
    with col2:
        if st.session_state.incident_step >= 1:
            st.subheader("🤖 AI 에이전트 분석 결과")

            with st.status("지식 그래프 탐색 중...", expanded=True) as status:
                st.write("1. 로그 텍스트에서 Entity 추출 중... (Pricing Engine, ELS, Out of memory)")
                time.sleep(0.5)
                st.write("2. 사내 조치 이력(Jira/Wiki) Ontology 매핑 중...")
                time.sleep(0.5)
                st.write("3. 유사 장애 패턴 (Similarity > 92%) 검색 완료.")
                time.sleep(0.5)
                status.update(label="분석 완료: 유사 장애 이력 3건 발견", state="complete", expanded=False)

            # 지식 그래프 추론 결과 매핑
            with st.expander("🔗 지식 그래프 추론 경로 (Traceability)", expanded=False):
                st.code("""
(ErrorLog: OOM) -[OCCURRED_IN]-> (Component: Pricing Engine)
(Component: Pricing Engine) -[HAS_HISTORY]-> (Ticket: INC-2025-08)
(Ticket: INC-2025-08) -[RESOLVED_BY]-> (Action: Increase Container Memory Limit)
(Ticket: INC-2025-08) -[RESOLVED_BY]-> (Action: Enable Chunk Processing)
                """, language="text")

            # 1차 조치 가이드라인
            st.info("💡 **1차 대응 가이드라인 (추천도: 높음)**\n\n과거 `INC-2025-08` (25년 8월 ELS 대규모 재평가 장애) 사례와 동일한 패턴입니다. 다음 조치를 순서대로 수행해 보십시오.")

            st.markdown("""
            **조치 1.** `config/pricing_env.yaml` 파일에서 컨테이너 메모리 제한을 임시 상향 (32GB $\\rightarrow$ 64GB)

            **조치 2.** 배치 파라미터 중 `CHUNK_SIZE`를 10,000에서 2,000으로 분할 설정

            **조치 3.** [시스템 오퍼레이션] 메뉴에서 부분 재배치(Partial Re-run) 실행
            """)

            st.markdown("---")
            st.markdown("#### 🛠️ 조치 결과 입력")

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("✅ 1차 조치로 해결됨 (DB 이력 저장)"):
                    st.success("해결 이력이 지식 그래프에 업데이트되었습니다. 시스템이 정상화되었습니다.")
                    st.session_state.incident_step = 0 # 초기화

            with col_b:
                if st.button("🚨 1차 대응 실패 (유지보수팀 호출)"):
                    st.session_state.incident_step = 2

        # 1차 대응 실패 시: 유지보수팀 에스컬레이션 화면
        if st.session_state.incident_step == 2:
            st.warning("⚠️ 1차 가이드라인으로 해결되지 않았습니다. 유지보수팀(L2/L3)으로 에스컬레이션합니다.")

            st.markdown("**✉️ 자동 생성된 호출 리포트 (Draft)**")
            with st.container(border=True):
                st.markdown("""
                **To:** 리스크 솔루션 개발팀 (dev_risk@company.com)
                **CC:** 시스템 운영 파트장
                **Subject:** [URGENT] Pricing Engine OOM 장애 1차 대응 실패 - 지원 요청

                **1. 장애 개요**
                * **발생 일시:** 2026-03-23 03:15 AM
                * **증상:** ELS Portfolio 평가 중 MemoryError 발생 (Container Limit 32GB Exceeded)

                **2. 1차 대응 내역 (운영팀 조치사항)**
                * 가이드에 따라 메모리 Limit 64GB 상향 및 Chunk Size 2000 분할을 적용 후 재시도 하였으나,
                * **추가 에러:** `CUDA Out of Memory` 로 전이되며 다시 실패함.

                **3. 요청 사항**
                * 로직 내 메모리 누수(Leak) 또는 GPU 메모리 해제 로직 점검 요망.
                * 즉각적인 L3 엔지니어 투입 및 디버깅을 요청합니다.
                """)

            if st.button("🚀 리포트 전송 및 엔지니어 Paging (Slack/Email 발송)"):
                st.toast("유지보수팀 호출이 완료되었습니다. Ticket #INC-2026-03 이 생성되었습니다.")
                time.sleep(1)
                st.session_state.incident_step = 0
