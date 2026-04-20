import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import re
import json

# 🛠️ 우리가 만든 엔터프라이즈 모듈들 임포트
from data_access.loaders import get_market_data, get_sell_side_portfolio
from quant_engines.pricing import calculate_parametric_var, revalue_bonds_multi, revalue_els_multi
from quant_engines.scenario_engine import ScenarioEngine
from ai_agents.orchestrator import ai_client
from ai_agents.knowledge_graph import kg_client

st.set_page_config(layout="wide")

# ==========================================
# 1. 데이터 및 엔진 초기화 (5만배 증폭 삭제, 순정 복구)
# ==========================================
@st.cache_data
def load_data():
    df_m = get_market_data(30)
    df_b, df_e = get_sell_side_portfolio()
    return df_m, df_b, df_e

df_market, df_bonds, df_els = load_data()
base_mkt = df_market.iloc[-2].to_dict()
curr_mkt = df_market.iloc[-1].to_dict()

engine = ScenarioEngine(base_mkt)

# ==========================================
# 2. 사이드바 및 UI 상태 관리 (Step 복구)
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

# 대화와 화면 단계를 분리하여 저장
if f"msgs_{selected_mode}" not in st.session_state:
    st.session_state[f"msgs_{selected_mode}"] = []
if f"step_{selected_mode}" not in st.session_state:
    st.session_state[f"step_{selected_mode}"] = 0

curr_msgs = st.session_state[f"msgs_{selected_mode}"]
curr_step_key = f"step_{selected_mode}"

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
                # 1. 마켓 리스크 브리핑
                var_amt, var_sens = calculate_parametric_var(df_bonds, df_els, df_market, 0.99)
                top_driver = var_sens.abs().idxmax()
                kg_context = kg_client.get_sell_side_kg_context(top_driver)
                
                bonds_res = revalue_bonds_multi(df_bonds, curr_mkt, base_mkt)
                els_res = revalue_els_multi(df_els, curr_mkt, base_mkt)
                bond_pnl = bonds_res['pnl'].sum()
                els_pnl = els_res['pnl'].sum()
                total_pnl = bond_pnl + els_pnl

                res = st.write_stream(ai_client.stream_sell_side_briefing(
                    prompt, total_pnl, els_pnl, bond_pnl, top_driver, 85.0, kg_context
                ))
                curr_msgs.append({"role": "assistant", "content": res})

                st.session_state.sell_dashboard = (total_pnl, var_amt, top_driver, bond_pnl, els_pnl)
                st.session_state[curr_step_key] = 1
                st.rerun()

            elif "3" in selected_mode:
                # 3. 시나리오 분석: JSON 파싱 후 승인 대기(Step 1) 상태로 진입
                res = ai_client.generate_dynamic_scenario(prompt)
                if res.get("intent") in ["new", "tuning"]:
                    # JSON 응답에서 파라미터 추출
                    st.session_state.scenario_params = res.get('parameters', [])
                    st.session_state[curr_step_key] = 1 # 승인 대기 상태로 변경
                    
                    msg = f"🔍 {res.get('rag_summary')}\n\n우측 화면에서 파라미터를 검토하시고 **승인 버튼**을 눌러 시뮬레이션을 실행해 주십시오."
                    st.write(msg)
                    curr_msgs.append({"role": "assistant", "content": msg})
                    st.rerun()
                else:
                    st.write(res.get("rag_summary"))
                    curr_msgs.append({"role": "assistant", "content": res.get("rag_summary")})

            elif "4" in selected_mode:
                # 4. 역방향 탐색: 엔진 호출 후 시각화(Step 1) 상태로 진입
                nums = re.findall(r'-?\d+', prompt)
                target_loss = float(nums[0]) if nums else -400.0
                st.write(f"🔍 목표 손실 {target_loss}억 원 도달 최단 경로 탐색을 시작합니다.")
                
                with st.spinner("RST 엔진 경사하강법 탐색 중..."):
                    path_result = engine.find_worst_case_path(df_bonds, df_els, target_loss)
                    st.session_state.rst_path = path_result
                    st.session_state[curr_step_key] = 1
                
                curr_msgs.append({"role": "assistant", "content": "탐색이 완료되었습니다. 우측 캔버스의 경로 데이터를 확인하십시오."})
                st.rerun()

# ==========================================
# 4. 시각화 캔버스 (Right Column)
# ==========================================
with col_viz:
    step = st.session_state[curr_step_key]
    
    if "1" in selected_mode and step == 1:
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

    # ----------------------------------------------------
    # 모드 3: 시나리오 분석 (승인 -> 실행 흐름 복구)
    # ----------------------------------------------------
    elif "3" in selected_mode:
        if step == 1:
            st.subheader("📝 시나리오 파라미터 검토 및 승인")
            # AI가 뽑아준 파라미터를 표 형태로 변환하여 편집기 띄우기
            df_params = pd.DataFrame(st.session_state.scenario_params)
            edited_df = st.data_editor(df_params, use_container_width=True, hide_index=True)
            
            if st.button("✅ 기안 승인 및 Full Revaluation 실행", type="primary"):
                with st.spinner("시계열 파급 시뮬레이션 중..."):
                    # 예시를 위해 데이터프레임의 값들을 하드코딩 수치로 단순 파싱하여 넘김
                    target_params = {'kospi': 80.0, 'samsung': 75.0, 'rate': 50.0} 
                    history = engine.generate_simulation_path(df_bonds, df_els, target_params)
                    st.session_state.scenario_history = history
                    st.session_state[curr_step_key] = 2 # 결과 보기 단계로 변경
                    st.rerun()
                    
        elif step == 2:
            st.subheader("📈 시뮬레이션 시계열 파급 결과")
            history = st.session_state.scenario_history
            df_hist = pd.DataFrame(history)
            
            # 최종 손익 강조
            final_pnl = df_hist.iloc[-1]['total_pnl']
            st.markdown(f"**최종 통합 평가 손익:** `<span style='color:red;'>{final_pnl/100000000:,.1f}억 원</span>`", unsafe_allow_html=True)
            
            st.markdown("#### 🔍 시뮬레이션 상세 데이터 (Audit Trail)")
            # 보기 좋게 포맷팅 (P&L은 억 단위로 변환)
            df_display = df_hist.copy()
            df_display['kospi'] = df_display['kospi'].map("{:.1f}%".format)
            df_display['samsung'] = df_display['samsung'].map("{:.1f}%".format)
            df_display['rate_shock'] = df_display['rate_shock'].map("+{:.0f}bp".format)
            df_display['total_pnl'] = (df_display['total_pnl']/100000000).map("{:,.1f}억".format)
            df_display['bond_pnl'] = (df_display['bond_pnl']/100000000).map("{:,.1f}억".format)
            df_display['els_pnl'] = (df_display['els_pnl']/100000000).map("{:,.1f}억".format)
            st.dataframe(df_display, use_container_width=True, hide_index=True)

    # ----------------------------------------------------
    # 모드 4: 역위기상황 탐색 (RST 경로 렌더링 복구)
    # ----------------------------------------------------
    elif "4" in selected_mode:
        if step == 1 and "rst_path" in st.session_state:
            st.subheader("◀ 역방향: 위기 좌표 탐색 완료")
            path = st.session_state.rst_path
            
            if not path:
                st.error("경로 탐색에 실패했습니다. 엔진 파라미터를 확인하세요.")
            else:
                final_state = path[-1]
                st.error(f"🚨 타격점 발견: KOSPI {final_state['kospi']:.1f}%, 삼성전자 {final_state['samsung']:.1f}%, 금리 +{final_state['rate']:.0f}bp")
                
                col_r1, col_r2 = st.columns(2)
                
                # 좌측: 최종 도달 레이더 차트
                fig = go.Figure(data=go.Scatterpolar(
                    r=[max(0, 100-final_state['kospi']), max(0, 100-final_state['samsung']), max(0, final_state['rate']/1.5), max(0, 100-final_state['kospi'])],
                    theta=['KOSPI 하락', '삼성전자 하락', '금리 상승', 'KOSPI 하락'],
                    fill='toself', line=dict(color='red', width=2)
                ))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, height=350, margin=dict(t=30, b=30))
                col_r1.plotly_chart(fig, use_container_width=True)

                # 우측: 탐색 궤적 상세 데이터 표
                with col_r2:
                    st.markdown("#### 🔍 경사하강법 탐색 궤적")
                    df_path = pd.DataFrame(path)
                    df_path['step'] = range(1, len(df_path) + 1)
                    df_path['kospi'] = df_path['kospi'].map("{:.1f}%".format)
                    df_path['samsung'] = df_path['samsung'].map("{:.1f}%".format)
                    df_path['rate'] = df_path['rate'].map("+{:.0f}bp".format)
                    df_path['pnl'] = (df_path['pnl']/100000000).map("{:,.1f}억".format)
                    st.dataframe(df_path[['step', 'kospi', 'samsung', 'rate', 'pnl']], use_container_width=True, hide_index=True)

    else:
        st.info("👈 좌측 대화창에서 AI 참모에게 분석을 지시해 주세요.")
