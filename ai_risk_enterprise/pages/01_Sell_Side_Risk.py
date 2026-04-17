import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import re

# 🛠️ 우리가 만든 엔터프라이즈 모듈들 임포트
from data_access.loaders import get_market_data, get_sell_side_portfolio
from quant_engines.pricing import calculate_parametric_var, revalue_bonds_multi, revalue_els_multi
from quant_engines.scenario_engine import ScenarioEngine
from ai_agents.orchestrator import ai_client
from ai_agents.knowledge_graph import kg_client

st.set_page_config(layout="wide")

# ==========================================
# 1. 데이터 및 엔진 초기화
# ==========================================
@st.cache_data
def load_data():
    df_m = get_market_data(30)
    df_b, df_e = get_sell_side_portfolio()
    return df_m, df_b, df_e

df_market, df_bonds, df_els = load_data()
base_mkt = df_market.iloc[-2].to_dict()
curr_mkt = df_market.iloc[-1].to_dict()

# 엔진 인스턴스 생성
engine = ScenarioEngine(base_mkt)

# ==========================================
# 2. 사이드바 및 UI 상태 관리
# ==========================================
modes = [
    "📊 1. 마켓 리스크 브리핑", 
    "🚨 2. 부서별 한도 관리", 
    "▶️ 3. 위기 시나리오 분석", 
    "◀️ 4. 역방향 위기 탐색 (RST)"
]

with st.sidebar:
    st.title("📈 Sell-Side 참모")
    selected_mode = st.radio("업무 모드 선택", modes)

# 세션 상태 초기화
if f"msgs_{selected_mode}" not in st.session_state:
    st.session_state[f"msgs_{selected_mode}"] = []

curr_msgs = st.session_state[f"msgs_{selected_mode}"]
col_chat, col_viz = st.columns([1.3, 1.7], gap="large")

# ==========================================
# 3. AI 참모 대화창 (Left Column)
# ==========================================
with col_chat:
    st.subheader(f"{selected_mode.split('. ')[1]}")
    
    if not curr_msgs:
        intro = "무엇을 도와드릴까요?"
        if "1" in selected_mode: intro = "전사 마켓 리스크 브리핑을 준비할까요?"
        elif "2" in selected_mode: intro = "부서별 한도 모니터링 중입니다. 처방이 필요하시면 지시하세요."
        elif "3" in selected_mode: intro = "거시 경제 위기 시나리오를 지시해 주십시오."
        elif "4" in selected_mode: intro = "목표 손실액(예: -400)을 입력하시면 최악의 위기 경로를 역산합니다."
        curr_msgs.append({"role": "assistant", "content": intro})

    for msg in curr_msgs:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("AI 참모에게 지시하세요..."):
        curr_msgs.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            if "1" in selected_mode:
                # 1. 마켓 리스크 브리핑 로직
                var_amt, var_sens = calculate_parametric_var(df_bonds, df_els, df_market, 0.99)
                top_driver = var_sens.abs().idxmax()
                
                # 지식 그래프에서 규정 가져오기
                kg_context = kg_client.get_sell_side_kg_context(top_driver)
                
                # 임시 P&L 연산
                bonds_res = revalue_bonds_multi(df_bonds, curr_mkt, base_mkt)
                els_res = revalue_els_multi(df_els, curr_mkt, base_mkt)
                bond_pnl = bonds_res['pnl'].sum()
                els_pnl = els_res['pnl'].sum()
                total_pnl = bond_pnl + els_pnl

                res = st.write_stream(ai_client.stream_sell_side_briefing(
                    total_pnl, els_pnl, bond_pnl, top_driver, 85.0, kg_context
                ))
                curr_msgs.append({"role": "assistant", "content": res})

                st.session_state.sell_dashboard = (total_pnl, var_amt, top_driver, bond_pnl, els_pnl)
                st.rerun()

            elif "2" in selected_mode:
                st.write("부서별 한도 관리 로직이 실행됩니다. (기존 구현체 활용)")
                curr_msgs.append({"role": "assistant", "content": "부서별 한도 모니터링 결과를 확인하세요."})

            elif "3" in selected_mode:
                # 3. 시나리오 분석 (JSON 추출 후 엔진 연산)
                res = ai_client.generate_dynamic_scenario(prompt)
                if res.get("intent") in ["new", "tuning"]:
                    st.write(f"🔍 {res.get('rag_summary')}\n\n우측 화면에서 시뮬레이션을 실행합니다.")
                    curr_msgs.append({"role": "assistant", "content": "시나리오 분석 완료. 우측 결과를 확인하세요."})
                    
                    # ScenarioEngine 호출!
                    target_params = {'kospi': 80.0, 'samsung': 75.0, 'rate': 50.0} # 임시 하드코딩 (파싱 로직 추가 가능)
                    st.session_state.scenario_history = engine.generate_simulation_path(df_bonds, df_els, target_params)
                    st.rerun()
                else:
                    st.write(res.get("rag_summary"))

            elif "4" in selected_mode:
                # 4. 역방향 탐색 (엔진 호출)
                nums = re.findall(r'-?\d+', prompt)
                target_loss = float(nums[0]) if nums else -400.0
                st.write(f"🔍 목표 손실 {target_loss}억 원 도달 경로를 탐색합니다.")
                
                # ScenarioEngine 호출!
                st.session_state.rst_path = engine.find_worst_case_path(df_bonds, df_els, target_loss)
                curr_msgs.append({"role": "assistant", "content": f"탐색 완료. 우측 화면을 확인하세요."})
                st.rerun()

# ==========================================
# 4. 시각화 캔버스 (Right Column)
# ==========================================
with col_viz:
    if "1" in selected_mode and "sell_dashboard" in st.session_state:
        st.subheader("📊 리스크 통합 대시보드")
        total_pnl, var_amt, top_driver, bond_pnl, els_pnl = st.session_state.sell_dashboard
        
        m1, m2, m3 = st.columns(3)
        m1.metric("통합 Net P&L", f"{total_pnl/100000000:,.1f}억")
        m2.metric("전사 VaR (99%, 1D)", f"{var_amt/100000000:,.1f}억")
        m3.metric("최대 리스크 동인", top_driver)
        
        st.markdown("#### 📁 데스크별 P&L 기여도")
        df_summary = pd.DataFrame({
            "포트폴리오 (Desk)": ["채권 운용 (Fixed Income)", "ELS 자체헤지 (Derivatives)"],
            "일간 손익 (P&L)": [f"{bond_pnl/100000000:,.1f}억", f"{els_pnl/100000000:,.1f}억"],
            "포지션 비중": ["Long Bias", "Delta Neutral"]
        })
        st.dataframe(df_summary, use_container_width=True, hide_index=True)

    elif "3" in selected_mode and "scenario_history" in st.session_state:
        st.subheader("📈 시뮬레이션 경로")
        history = st.session_state.scenario_history
        df_hist = pd.DataFrame(history)
        st.dataframe(df_hist, use_container_width=True)
        st.info("이곳에 시계열 궤적 차트가 렌더링 됩니다.")

    elif "4" in selected_mode and "rst_path" in st.session_state:
        st.subheader("◀ 역방향: 위기 좌표 탐색 완료")
        path = st.session_state.rst_path
        final_state = path[-1]
        
        st.error(f"🚨 타격점 발견: KOSPI {final_state['kospi']:.1f}%, 삼성전자 {final_state['samsung']:.1f}%, 금리 +{final_state['rate']:.0f}bp")
        
        fig = go.Figure(data=go.Scatterpolar(
            r=[100-final_state['kospi'], 100-final_state['samsung'], final_state['rate']/1.5, 100-final_state['kospi']],
            theta=['KOSPI 하락', '삼성전자 하락', '금리 상승', 'KOSPI 하락'],
            fill='toself', line=dict(color='red', width=2)
        ))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👈 좌측 대화창에서 AI 참모에게 분석을 지시해 주세요.")