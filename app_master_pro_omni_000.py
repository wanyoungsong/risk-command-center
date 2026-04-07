import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pyvis.network import Network
import time
import re
import google.generativeai as genai
import json

# ==========================================
# 0. API 세팅 및 AI 에이전트 도구(Tools) 정의
# ==========================================
try:
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", None)
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
except Exception:
    pass

# [Omni-Agent 핵심] AI가 스스로 선택할 '행동(Function)'들을 정의합니다.
def route_market_briefing():
    """전사 마켓 리스크 현황과 P&L, VaR를 조회하고 브리핑 대시보드를 엽니다."""
    return "view_briefing"

def route_limit_management(department: str):
    """부서별(전사, 채권데스크, ELS데스크) 리스크 한도 소진율을 확인하고 한도 모니터링 화면을 엽니다."""
    return "view_limits"

def route_what_if_scenario(scenario_text: str):
    """자연어 거시경제 시나리오를 바탕으로 파라미터를 추출하고 스트레스 테스트(What-If) 화면을 엽니다."""
    return "view_what_if"

def route_reverse_stress_test(target_loss: float):
    """경영진이 제시한 목표 손실액(예: -400)에 도달하는 최단 위기 경로(RST) 역탐색 화면을 엽니다."""
    return "view_rst"

# 도구 목록
omni_tools = [route_market_briefing, route_limit_management, route_what_if_scenario, route_reverse_stress_test]

# ==========================================
# 1. 수학 및 프라이싱 엔진 (기존 증권사 코드 100% 동일 축약)
# ==========================================
# (※ 시연을 위해 기존 엔진 로직을 간단히 구조화했습니다. 실제로는 기존 코드를 그대로 쓰시면 됩니다.)
base_mkt_state = {"KOSPI200_Close": 330, "Samsung_Close": 71000, "Vol_KOSPI200": 0.28, "KTB_3Y": 3.7}
daily_total_pnl = -12000000000  # -120억
var_usage_pct = 95.5
top_risk_driver = "Vol_KOSPI200 (변동성)"

# ==========================================
# 2. GraphRAG 및 환각(Hallucination) 방지 프롬프트
# ==========================================
def get_neo4j_context_mock(intent_keyword):
    """실제 Neo4j 쿼리를 대체하는 Mock 함수 (환각 테스트용)"""
    # 실제로는 이 부분에서 GraphDatabase.driver(...) 로 AURA에서 데이터를 끌어옵니다.
    if "briefing" in intent_keyword:
        return "사내 규정 Article 12-1: 전사 VaR 소진율 90% 초과 시 신규 리스크 테이킹 즉각 중지."
    elif "limits" in intent_keyword:
        return "사내 규정 Article 14-1: ELS 데스크 변동성 한도 초과 시 베가 중립을 위한 헤지 비중 기계적 확대."
    return "조회된 지식 그래프 데이터가 없습니다."

def generate_omni_response(prompt_text, view_state):
    """Neo4j 데이터에만 의존하도록 가드레일이 쳐진 LLM 생성"""
    kg_context = get_neo4j_context_mock(view_state)
    
    # [가드레일 핵심] 시스템 프롬프트로 강제 지시
    system_instruction = f"""
    너는 리스크 통합 에이전트(Omni-Agent)야.
    반드시 아래 제공된 [Neo4j 지식 그래프] 데이터에만 근거하여 경영진에게 보고해.
    지식 그래프에 없는 외부 정보나 너의 자체적인 금융 지식을 덧붙이는 것은 '엄격히 금지(Strictly Prohibited)'된다.
    만약 지식 그래프 내용이 부족하다면 "지식 그래프에 해당 규정이 없습니다"라고 솔직하게 답해.
    
    [Neo4j 지식 그래프]
    {kg_context}
    """
    
    model = genai.GenerativeModel(
        model_name='gemini-3.1-flash-lite-preview',
        system_instruction=system_instruction
    )
    return model.generate_content(prompt_text).text

# ==========================================
# 3. Streamlit 메인 UI (Omni-Agent 레이아웃)
# ==========================================
st.set_page_config(layout="wide", page_title="Omni Risk Agent")

# 세션 상태 초기화 (메뉴 대신 view 상태로 관리)
if "current_view" not in st.session_state:
    st.session_state.current_view = "home"
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요. 통합 리스크 에이전트입니다. 마켓 브리핑, 한도 조회, 시나리오 분석 중 무엇을 도와드릴까요?"}]

# 좌측 사이드바는 이제 메뉴가 없고 단순 타이틀만 존재!
with st.sidebar:
    st.title("🤖 Omni Agent")
    st.caption("메뉴 없는 대화형 리스크 지휘소")
    st.info("💡 **Try it:**\n- '오늘 마켓 브리핑 해줘'\n- 'ELS 부서 한도 어때?'\n- '유가 급등 시나리오 돌려봐'")

col_chat, col_viz = st.columns([1.3, 1.7], gap="large")

# -----------------
# Center Panel: Chat & Function Calling
# -----------------
with col_chat:
    st.subheader("💬 통합 커맨드 센터")
    
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("에이전트에게 지시하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("의도 파악 및 도구 선택 중..."):
                # [Function Calling 핵심] AI에게 사용자의 말을 던지고 도구를 고르게 함
                model = genai.GenerativeModel(model_name='gemini-3.1-flash-lite-preview', tools=omni_tools)
                response = model.generate_content(prompt)
                
                # AI가 함수 호출(Function Call)을 결정했는지 확인
                function_called = False
                for part in response.parts:
                    if part.function_call:
                        function_called = True
                        func_name = part.function_call.name
                        args = part.function_call.args
                        
                        # AI가 선택한 함수 이름에 따라 View 상태 변경
                        if func_name == "route_market_briefing":
                            st.session_state.current_view = "view_briefing"
                        elif func_name == "route_limit_management":
                            st.session_state.current_view = "view_limits"
                        elif func_name == "route_what_if_scenario":
                            st.session_state.current_view = "view_what_if"
                            # 시나리오 텍스트 저장 로직 추가 가능
                        elif func_name == "route_reverse_stress_test":
                            st.session_state.current_view = "view_rst"
                            
                        st.markdown(f"*(시스템 로드 완료: 우측 화면에 `{st.session_state.current_view}`을 띄웁니다.)*")
                        break
                
                # View가 변경되었으므로 GraphRAG에 기반한 답변 생성
                safe_response = generate_omni_response(f"방금 사용자가 '{prompt}'라고 지시했어. 우측에 화면을 띄웠으니 관련 브리핑을 해줘.", st.session_state.current_view)
                st.write(safe_response)
                st.session_state.messages.append({"role": "assistant", "content": safe_response})
                time.sleep(1)
                st.rerun() # 우측 캔버스 업데이트!

# -----------------
# Right Panel: Dynamic View Rendering
# -----------------
with col_viz:
    view = st.session_state.current_view
    
    if view == "home":
        st.empty() # 초기 빈 화면
        
    elif view == "view_briefing":
        st.subheader("📊 전사 마켓 리스크 브리핑")
        c1, c2, c3 = st.columns(3)
        c1.metric("통합 P&L", f"{daily_total_pnl/100000000:,.1f}억")
        c2.metric("VaR 소진율", f"{var_usage_pct}%")
        c3.metric("핵심 동인", top_risk_driver)
        st.info("여기에 기존의 듀얼 액시스 차트와 Pyvis 지식 그래프가 렌더링됩니다.")
        
    elif view == "view_limits":
        st.subheader("🚨 부서별 한도 모니터링")
        st.progress(0.95, text="전사 VaR 한도 (95%)")
        st.progress(1.0, text="ELS 데스크 Vega 한도 (100% 초과!)")
        st.info("여기에 기존 부서별 한도 모니터링 표가 렌더링됩니다.")
        
    elif view == "view_what_if":
        st.subheader("📈 위기 시나리오 분석 (What-If)")
        st.warning("사용자가 지시한 시나리오를 바탕으로 파라미터를 추출했습니다.")
        st.info("여기에 기존 3D 표면도와 시뮬레이션 애니메이션이 렌더링됩니다.")
        
    elif view == "view_rst":
        st.subheader("◀ 역방향 위기 좌표 탐색 (RST)")
        st.error("지정된 목표 손실에 도달하는 최악의 경로를 찾습니다.")
        st.info("여기에 기존 레이더 차트와 손실 등고선 애니메이션이 렌더링됩니다.")
