import sys
import os
# 모듈 인식을 위한 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import plotly.graph_objects as go
from pyvis.network import Network
import time
import pandas as pd
import re

# 분리된 모듈들 임포트
from data_access.loaders import generate_daily_risk_factors, get_bond_portfolio, get_els_portfolio
from quant_engines.pricing import revalue_bonds_multi, revalue_els_multi, calculate_parametric_var
from ai_agents.knowledge_graph import kg_client
from ai_agents.orchestrator import ai_client

# ==========================================
# 3. Streamlit 메인 UI (3단 Agent 레이아웃)
# ==========================================
st.set_page_config(layout="wide", page_title="AI Risk Agent UI")

# 실시간 연산 데이터 준비
@st.cache_data
def load_data_and_calc_base():
    df_market_data = generate_daily_risk_factors(30)
    df_bonds = get_bond_portfolio()
    df_els = get_els_portfolio()
    return df_market_data, df_bonds, df_els

df_market_data, df_bonds, df_els = load_data_and_calc_base()

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
dynamic_limits = kg_client.get_dynamic_risk_limits()

# ---------------------------------------------------------
# 상단 세션 초기화 부분
# ---------------------------------------------------------
modes = [
    "📊 1-1. 마켓 리스크 브리핑", 
    "🚨 1-2. 부서별 한도 관리", 
    "▶️ 2-1. 위기 시나리오 분석", 
    "◀️ 2-2. 역방향 위기 탐색"
]

for m in modes:
    if f"msgs_{m}" not in st.session_state: 
        st.session_state[f"msgs_{m}"] = []
    if f"step_{m}" not in st.session_state: 
        st.session_state[f"step_{m}"] = 0

if 'target_loss' not in st.session_state: 
    st.session_state.target_loss = -400.0

# ---------------------------------------------------------
# Left Panel: Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.title("🤖 Risk Agent")
    st.caption("AI 기반 전사 리스크 지휘소")
    st.markdown("---")
    
    selected_mode = st.radio("업무 모드 선택", modes)
    
    st.markdown("---")
    st.caption(f"System Status: {'Warning' if var_usage_pct > 90 else 'Normal'}")

# -----------------
# 메인 레이아웃 분할 (Center: Chat, Right: Viz)
# -----------------
col_chat, col_viz = st.columns([1.3, 1.7], gap="large")

# -----------------
# Center Panel: Chat Logic
# -----------------
curr_msgs = st.session_state[f"msgs_{selected_mode}"]
curr_step_key = f"step_{selected_mode}"

with col_chat:
    st.subheader(f"{selected_mode.split('. ')[1]}")
    
    if not curr_msgs:
        intro = "무엇을 도와드릴까요?"
        if "1-1" in selected_mode: intro = "전사 마켓 리스크 현황이 우측 대시보드에 업데이트되었습니다. 경영진 보고용 시황 브리핑이 필요하시면 지시해 주십시오."
        elif "1-2" in selected_mode: intro = "부서별 리스크 한도 모니터링 중입니다. 한도 초과에 대한 처방전 기안이 필요하면 지시해 주십시오."
        elif "2-1" in selected_mode: intro = "거시 경제 위기 시나리오를 지시해 주십시오. (예: '전쟁 확전으로 유가가 급등하는 상황 분석해줘')"
        elif "2-2" in selected_mode: intro = "경영진이 우려하는 목표 손실액을 입력해 주시면 최악의 위기 경로를 역산합니다. (예: '목표손실 -400억 경로 찾아줘')"
        curr_msgs.append({"role": "assistant", "content": intro})

    for msg in curr_msgs:
        st.chat_message(msg["role"]).write(msg["content"])

    if "2-1" in selected_mode and st.session_state[curr_step_key] == 3:
        with st.chat_message("assistant"):
            st.markdown("🚨 **시뮬레이션 분석 결과를 브리핑합니다.**")
            df_json = st.session_state.final_scenario_df.to_dict(orient="records")
            response = st.write_stream(ai_client.stream_scenario_response(st.session_state.final_sim_pnl, df_json))
            curr_msgs.append({"role": "assistant", "content": f"🚨 [시뮬레이션 분석 결과]\n\n{response}"})
            st.session_state[curr_step_key] = 4

    if "2-2" in selected_mode and st.session_state[curr_step_key] == 2:
        with st.chat_message("assistant"):
            st.markdown("🚨 **역방향 위기 탐색 결과를 브리핑합니다.**")
            fk, fs, fr = st.session_state.rst_final_factors
            response = st.write_stream(ai_client.stream_rst_response(st.session_state.target_loss, fk, fs, fr))
            curr_msgs.append({"role": "assistant", "content": f"🚨 [RST 분석 결과]\n\n{response}"})
            st.session_state[curr_step_key] = 3

    if prompt := st.chat_input("AI 참모에게 지시를 입력하세요..."):
        curr_msgs.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            if "1-1" in selected_mode:
                kg_context = kg_client.get_knowledge_graph_context(top_risk_driver)
                res = st.write_stream(ai_client.stream_ai_briefing(daily_total_pnl, daily_els_pnl, daily_bond_pnl, top_risk_driver, var_usage_pct, kg_context))
                curr_msgs.append({"role": "assistant", "content": res})

            elif "1-2" in selected_mode:
                dept_code, dept_name, usage, metric, exp, limit = 'ENTERPRISE', '전사 통합 리스크 위원회', var_usage_pct, '전사 VaR', var_amount/100000000, dynamic_limits.get("전사 VaR 한도", 1500)
                if '채권' in prompt:
                    dept_code, dept_name, usage, metric, exp, limit = 'BOND_DESK', '채권 운용 데스크', abs(rho_exposure_bn/dynamic_limits.get("금리 민감도(Rho) 한도", 40))*100, 'Rho', rho_exposure_bn, dynamic_limits.get("금리 민감도(Rho) 한도", 40)
                elif 'ELS' in prompt or 'els' in prompt.lower():
                    dept_code, dept_name, usage, metric, exp, limit = 'ELS_DESK', 'ELS 운용 데스크', abs(vega_exposure_bn/dynamic_limits.get("변동성 민감도(Vega) 한도", 30))*100, 'Vega', vega_exposure_bn, dynamic_limits.get("변동성 민감도(Vega) 한도", 30)
                
                kg_context = kg_client.get_compliance_graph_context(dept_code, usage)
                res = st.write_stream(ai_client.stream_ai_prescription(dept_name, usage, metric, exp, limit, kg_context))
                curr_msgs.append({"role": "assistant", "content": res})

            elif "2-1" in selected_mode:
                with st.spinner("AI 참모가 의도를 분석하고 답변을 준비 중입니다..."):
                    current_params = None
                    if st.session_state[curr_step_key] >= 1 and 'scenario_data' in st.session_state:
                        current_params = st.session_state.scenario_data.get('parameters')
                        
                    res = ai_client.generate_dynamic_scenario(prompt, current_params)
                
                intent = res.get("intent", "irrelevant")
                
                if intent == "irrelevant":
                    st.write(res.get("rag_summary"))
                    curr_msgs.append({"role": "assistant", "content": res.get("rag_summary")})
                
                elif intent == "explain":
                    st.write(res.get("rag_summary"))
                    curr_msgs.append({"role": "assistant", "content": res.get("rag_summary")})
                    
                elif intent in ["new", "tuning"]:
                    st.session_state.scenario_data = res
                    df_params = pd.DataFrame(res['parameters'])
                    if len(df_params.columns) == 4:
                        df_params.columns = ["리스크 팩터", "현재 수준", "최대 충격 (Target)", "충격 도달 기간"]
                    
                    st.session_state.final_scenario_df = df_params
                    st.session_state[curr_step_key] = 1 
                    
                    msg = f"🔍 {res.get('rag_summary')}\n\n우측 화면에서 파라미터를 검토하시고 승인 버튼을 눌러 시뮬레이션을 실행해 주십시오."
                    st.write(msg)
                    curr_msgs.append({"role": "assistant", "content": msg})
                    st.rerun()

            elif "2-2" in selected_mode:
                nums = re.findall(r'-?\d+', prompt)
                st.session_state.target_loss = float(nums[0]) if nums else -400.0
                st.session_state[curr_step_key] = 1
                msg = f"🔍 목표 손실 **{st.session_state.target_loss}억 원**을 유발하는 최단 위기 경로 탐색을 시작합니다. 우측 화면을 확인해 주십시오."
                st.write(msg)
                curr_msgs.append({"role": "assistant", "content": msg})
                st.rerun()

# -----------------
# Right Panel: Visualization Render (우측 캔버스 동적 렌더링)
# -----------------
with col_viz:
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
        step = st.session_state[curr_step_key]
        if step == 0:
            st.info("👈 좌측 대화창에 위기 시나리오를 자연어로 지시해 주십시오.")
        
        elif step == 1:
            s_data = st.session_state.scenario_data
            st.subheader("📝 시나리오 파라미터 검토 및 승인")
            
            df_params = pd.DataFrame(s_data['parameters'])
            if len(df_params.columns) == 4:
                df_params.columns = ["리스크 팩터", "현재 수준", "최대 충격 (Target)", "충격 도달 기간"]
            edited_df = st.data_editor(df_params, use_container_width=True, hide_index=True)
            
            if st.button("✅ 기안 승인 및 Full Revaluation 실행", type="primary"):
                st.session_state.final_scenario_df = edited_df
                st.session_state[curr_step_key] = 2
                st.rerun()
                
        elif step == 2:
            st.markdown("---")
            st.subheader("📈 시계열 파급 분석 (Time-Step Full Revaluation)")
            chart_placeholder = st.empty()
            metrics_placeholder = st.empty()
            
            df = st.session_state.final_scenario_df
            def get_target_num(text, default=100.0):
                nums = re.findall(r'[-+]?\d*\.?\d+', str(text))
                return float(nums[-1]) if nums else default

            rate_target, kospi_target, samsung_target, duration_days = 0.0, 100.0, 100.0, 14
            for _, row in df.iterrows():
                factor = row.iloc[0]
                target_val = get_target_num(row.iloc[2])
                if "금리" in factor: rate_target = target_val
                elif "KOSPI" in factor: kospi_target = target_val
                elif "삼성전자" in factor: samsung_target = target_val
            
            steps_count = 7
            traj_x = np.linspace(100, kospi_target, steps_count).tolist()
            traj_y = np.linspace(100, samsung_target, steps_count).tolist()
            rate_shocks = np.linspace(0, rate_target, steps_count).tolist()
            traj_z = [100 - ((100 - x)*1.5 + (100 - y)*1.5) for x, y in zip(traj_x, traj_y)]
            time_labels = [f"Day {int(d)}" for d in np.linspace(0, duration_days, steps_count)]
            
            x, y = np.linspace(40, 100, 40), np.linspace(40, 100, 40)
            X, Y = np.meshgrid(x, y)
            Z = 100 + (X - 100)*0.3 + (Y - 100)*0.3 - np.where(X < 80, (80 - X)*1.5, 0) - np.where(Y < 60, (60 - Y)*2.5, 0) - 35*np.exp(-0.03*((X - 75)**2 + (Y - 55)**2))
            
            df_bonds_scaled = df_bonds.copy()
            df_bonds_scaled['qty'] = df_bonds_scaled['qty'] * 100
            final_total_pnl = 0
            
            scen_history = [] 

            for i in range(steps_count):
                fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Blues', opacity=0.7)])
                fig.add_trace(go.Scatter3d(x=traj_x[:i+1], y=traj_y[:i+1], z=traj_z[:i+1], mode='lines', line=dict(color='orange', width=6)))
                fig.add_trace(go.Scatter3d(x=[traj_x[i]], y=[traj_y[i]], z=[traj_z[i]], mode='markers', marker=dict(size=10, color='red')))
                fig.update_layout(title=f"⏳ 진행 상태: {time_labels[i]}", scene=dict(xaxis_title='KOSPI 200', yaxis_title='삼성전자', camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.2))), margin=dict(l=0, r=0, b=0, t=30), height=350)
                chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"sim_anim_{i}")

                curr_mkt = base_mkt_state.copy()
                curr_mkt['KOSPI200_Close'] *= (traj_x[i]/100.0)
                curr_mkt['Samsung_Close'] *= (traj_y[i]/100.0)
                for tenor in ['KTB_6M', 'KTB_1Y', 'KTB_3Y', 'KTB_5Y', 'Corp_6M', 'Corp_1Y']: curr_mkt[tenor] += (rate_shocks[i] / 100.0)
                for key in curr_mkt.keys():
                    if key.startswith('Vol_'): curr_mkt[key] += (100 - traj_x[i]) * 0.005

                sim_bonds = revalue_bonds_multi(df_bonds_scaled, curr_mkt, base_mkt_state)
                sim_els = revalue_els_multi(df_els, curr_mkt, base_mkt_state)
                final_total_pnl = sim_bonds['pnl'].sum() + sim_els['pnl'].sum()

                row_data = {
                    "단계": time_labels[i],
                    "KOSPI 잔존": f"{traj_x[i]:.1f}%",
                    "삼성전자 잔존": f"{traj_y[i]:.1f}%",
                    "금리 충격": f"+{rate_shocks[i]:.0f} bp"
                }
                for _, r in sim_bonds.iterrows(): row_data[f"B_{r['name']}"] = f"{r['pnl']:,.0f}"
                for _, r in sim_els.iterrows(): row_data[f"E_{r['name']}"] = f"{r['pnl']:,.0f}"
                scen_history.append(row_data)

                with metrics_placeholder.container():
                    st.markdown(f"**실시간 통합 평가 손익:** `<span style='color:red;'>{final_total_pnl/100000000:,.1f}억 원</span>`", unsafe_allow_html=True)
                time.sleep(0.5)

            st.session_state.scen_history = scen_history
            st.session_state.scenario_fig = fig
            st.session_state.final_sim_pnl = final_total_pnl
            st.session_state[curr_step_key] = 3
            st.rerun()

        elif step >= 3:
            st.markdown("---")
            st.subheader("📈 시계열 파급 분석 (종료)")
            if 'scenario_fig' in st.session_state:
                st.plotly_chart(st.session_state.scenario_fig, use_container_width=True, key="sim_final")
            if 'final_sim_pnl' in st.session_state:
                st.markdown(f"**최종 통합 평가 손익:** `<span style='color:red;'>{st.session_state.final_sim_pnl/100000000:,.1f}억 원</span>`", unsafe_allow_html=True)
            st.markdown("#### 🔍 시뮬레이션 상세 데이터 (Audit Trail)")
            if 'scen_history' in st.session_state:
                st.dataframe(pd.DataFrame(st.session_state.scen_history), use_container_width=True, hide_index=True)

    elif "2-2" in selected_mode:
        step = st.session_state[curr_step_key]
        if step == 0:
            st.info("👈 좌측 대화창에 도달하고자 하는 목표 손실액을 지시해 주십시오.")
        elif step == 1:
            st.subheader("◀ 역방향: 위기 좌표 탐색 (Reverse Stress Test)")
            col_r1, col_r2 = st.columns(2)
            radar_ph, contour_ph = col_r1.empty(), col_r2.empty()
            
            df_bonds_scaled = df_bonds.copy()
            df_bonds_scaled['qty'] = df_bonds_scaled['qty'] * 100
            
            def eval_pnl(k, s, r):
                mkt = base_mkt_state.copy()
                mkt['KOSPI200_Close'] *= (k / 100.0)
                mkt['Samsung_Close'] *= (s / 100.0)
                for tenor in ['KTB_6M', 'KTB_1Y', 'KTB_3Y', 'KTB_5Y', 'Corp_6M', 'Corp_1Y']: mkt[tenor] += (r / 100.0)
                for key in mkt.keys():
                    if key.startswith('Vol_'): mkt[key] += (100 - min(k,s)) * 0.005
                tb = revalue_bonds_multi(df_bonds_scaled, mkt, base_mkt_state)
                te = revalue_els_multi(df_els, mkt, base_mkt_state)
                return tb['pnl'].sum() + te['pnl'].sum()

            target_pnl_raw = st.session_state.target_loss * 100000000
            search_k, search_s, search_r = 100.0, 100.0, 0.0
            path_k, path_s, path_r = [100.0], [100.0], [0.0]
            current_pnl = eval_pnl(search_k, search_s, search_r)

            with st.spinner("최적화 알고리즘 역탐색 중..."):
                for _ in range(50):
                    if current_pnl <= target_pnl_raw: break
                    eps = 1.0
                    grad_k_raw = max(0, current_pnl - eval_pnl(search_k - eps, search_s, search_r))
                    grad_s_raw = max(0, current_pnl - eval_pnl(search_k, search_s - eps, search_r))
                    grad_r_raw = max(0, current_pnl - eval_pnl(search_k, search_s, search_r + eps))

                    grad_k, grad_s, grad_r = grad_k_raw + (grad_s_raw * 0.3) + 1.0, grad_s_raw + (grad_k_raw * 0.5) + 1.5, grad_r_raw + 2.0
                    total_grad = grad_k + grad_s + grad_r
                    search_k = max(10.0, search_k - (grad_k / total_grad) * 2.0)
                    search_s = max(10.0, search_s - (grad_s / total_grad) * 2.0)
                    search_r += (grad_r / total_grad) * 5.0
                    current_pnl = eval_pnl(search_k, search_s, search_r)
                    path_k.append(search_k); path_s.append(search_s); path_r.append(search_r)
            
            steps = 10
            idx = np.linspace(0, len(path_k)-1, steps).astype(int)
            k_path, s_path, r_path = [path_k[i] for i in idx], [path_s[i] for i in idx], [path_r[i] for i in idx]

            k_grid, s_grid = np.linspace(max(0, int(search_k)-10), 100, 10), np.linspace(max(0, int(search_s)-10), 100, 10)
            K_MESH, S_MESH = np.meshgrid(k_grid, s_grid)
            Z_PNL = np.zeros((10, 10))
            
            history_data = [] 
            for step in range(steps):
                ck, cs, cr = k_path[step], s_path[step], r_path[step]
                
                mkt_step = base_mkt_state.copy()
                mkt_step['KOSPI200_Close'] *= (ck / 100.0)
                mkt_step['Samsung_Close'] *= (cs / 100.0)
                for tenor in ['KTB_6M', 'KTB_1Y', 'KTB_3Y', 'KTB_5Y', 'Corp_6M', 'Corp_1Y']: mkt_step[tenor] += (cr / 100.0)
                for key in mkt_step.keys():
                    if key.startswith('Vol_'): mkt_step[key] += (100 - min(ck,cs)) * 0.005
                
                sim_b_step = revalue_bonds_multi(df_bonds_scaled, mkt_step, base_mkt_state)
                sim_e_step = revalue_els_multi(df_els, mkt_step, base_mkt_state)

                row_data = {
                    "탐색 단계": f"Step {step+1}",
                    "KOSPI 잔존가치": f"{ck:.1f}%",
                    "삼성전자 잔존가치": f"{cs:.1f}%",
                    "국채금리 충격": f"+{cr:.0f} bp"
                }
                for _, r in sim_b_step.iterrows(): row_data[f"B_{r['name']}"] = f"{r['pnl']:,.0f}"
                for _, r in sim_e_step.iterrows(): row_data[f"E_{r['name']}"] = f"{r['pnl']:,.0f}"
                history_data.append(row_data)

                fig_radar = go.Figure(data=go.Scatterpolar(
                    r=[min(100, max(0, (100-ck)/50*100)), min(100, max(0, (100-cs)/50*100)), min(100, max(0, cr/150*100)), min(100, max(0, (100-ck)/50*100))],
                    theta=['KOSPI 하락', '삼성전자 하락', '금리 상승', 'KOSPI 하락'], fill='toself', line=dict(color='red', width=2)
                ))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, height=350, margin=dict(l=30,r=30,t=30,b=30))
                radar_ph.plotly_chart(fig_radar, use_container_width=True, key=f"radar_{step}")

                fig_contour = go.Figure(data=go.Contour(z=(K_MESH+S_MESH), x=k_grid, y=s_grid, colorscale='RdBu'))
                fig_contour.add_trace(go.Scatter(x=k_path[:step+1], y=s_path[:step+1], mode='lines+markers', line=dict(color='#00FF00', width=4, dash='dot'), marker=dict(size=6, color='black')))
                fig_contour.add_trace(go.Scatter(x=[ck], y=[cs], mode='markers', marker=dict(size=14, color='yellow', symbol='star')))
                fig_contour.update_layout(height=350, margin=dict(l=30,r=30,t=30,b=30), showlegend=False)
                contour_ph.plotly_chart(fig_contour, use_container_width=True, key=f"contour_{step}")
                time.sleep(0.4)

            st.session_state.rst_radar = fig_radar
            st.session_state.rst_contour = fig_contour
            st.session_state.rst_history = history_data
            st.session_state.rst_final_factors = (k_path[-1], s_path[-1], r_path[-1])
            st.session_state[curr_step_key] = 2 
            st.rerun()

        elif step >= 2:
            st.subheader("◀ 역방향: 위기 좌표 탐색 (종료)")
            cr1, cr2 = st.columns(2)
            if 'rst_radar' in st.session_state:
                cr1.plotly_chart(st.session_state.rst_radar, use_container_width=True, key="radar_final")
            if 'rst_contour' in st.session_state:
                cr2.plotly_chart(st.session_state.rst_contour, use_container_width=True, key="contour_final")
            
            st.markdown("---")
            st.markdown("#### 🔍 역위기상황 탐색 시계열 상세 데이터 (Audit Trail)")
            if 'rst_history' in st.session_state:
                st.dataframe(pd.DataFrame(st.session_state.rst_history), use_container_width=True, hide_index=True)
