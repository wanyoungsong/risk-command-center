import sys
import os
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import re
import time

# [중요] pages 폴더는 루트보다 한 단계 더 깊으므로 두 단계 위(..)를 경로에 추가합니다.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 분리된 엔터프라이즈 모듈들 임포트
from data_access.loaders import generate_daily_risk_factors, get_bond_portfolio, get_els_portfolio
from quant_engines.pricing import revalue_bonds_multi, revalue_els_multi, calculate_parametric_var
from ai_agents.knowledge_graph import kg_client
from ai_agents.orchestrator import ai_client

st.set_page_config(layout="wide", page_title="Sell-Side Risk Agent")

# ==========================================
# 1. 데이터 및 실시간 연산 초기화 (원본 로직)
# ==========================================
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

# VaR 및 민감도 계산
var_amount, var_sens = calculate_parametric_var(df_bonds, df_els, df_market_data, 0.99)
var_limit = 1500000000
var_usage_pct = (var_amount / var_limit) * 100
top_risk_driver = var_sens.abs().idxmax()
rho_exposure_bn = var_sens.filter(like='KTB').sum() / 100000000
vega_exposure_bn = var_sens.filter(like='Vol').sum() / 100000000
dynamic_limits = kg_client.get_dynamic_risk_limits()

# ==========================================
# 2. 사이드바 및 업무 모드 선택
# ==========================================
modes = [
    "📊 1-1. 마켓 리스크 브리핑", 
    "🚨 1-2. 부서별 한도 관리", 
    "▶️ 2-1. 위기 시나리오 분석", 
    "◀️ 2-2. 역방향 위기 탐색"
]

with st.sidebar:
    st.title("🤖 Risk Agent")
    st.caption("Sell-Side 전사 리스크 지휘소")
    st.markdown("---")
    selected_mode = st.radio("업무 모드 선택", modes)

# 모드별 세션 상태 초기화
for m in modes:
    if f"msgs_{m}" not in st.session_state: st.session_state[f"msgs_{m}"] = []
    if f"step_{m}" not in st.session_state: st.session_state[f"step_{m}"] = 0

curr_msgs = st.session_state[f"msgs_{selected_mode}"]
curr_step_key = f"step_{selected_mode}"

col_chat, col_viz = st.columns([1.3, 1.7], gap="large")

# ==========================================
# 3. Center Panel: Chat Logic (AI 참모)
# ==========================================
with col_chat:
    st.subheader(f"{selected_mode.split('. ')[1]}")
    
    if not curr_msgs:
        intro = "무엇을 도와드릴까요?"
        if "1-1" in selected_mode: intro = "전사 마켓 리스크 현황이 우측 대시보드에 업데이트되었습니다. 경영진 보고용 시황 브리핑이 필요하시면 지시해 주십시오."
        elif "1-2" in selected_mode: intro = "부서별 리스크 한도 모니터링 중입니다. 한도 초과에 대한 처방전 기안이 필요하면 지시해 주십시오."
        elif "2-1" in selected_mode: intro = "거시 경제 위기 시나리오를 지시해 주십시오. (예: '유가가 급등하는 상황 분석해줘')"
        elif "2-2" in selected_mode: intro = "경영진이 우려하는 목표 손실액을 입력해 주시면 최악의 위기 경로를 역산합니다."
        curr_msgs.append({"role": "assistant", "content": intro})

    for msg in curr_msgs:
        st.chat_message(msg["role"]).write(msg["content"])

    # 시뮬레이션 종료 후 자동 브리핑 (Step 3 -> 4)
    if "2-1" in selected_mode and st.session_state[curr_step_key] == 3:
        with st.chat_message("assistant"):
            st.markdown("🚨 **시뮬레이션 분석 결과를 브리핑합니다.**")
            df_json = st.session_state.final_scenario_df.to_dict(orient="records")
            response = st.write_stream(ai_client.stream_scenario_response(st.session_state.final_sim_pnl, df_json))
            curr_msgs.append({"role": "assistant", "content": f"🚨 [시뮬레이션 분석 결과]\n\n{response}"})
            st.session_state[curr_step_key] = 4

    # RST 종료 후 자동 브리핑 (Step 2 -> 3)
    if "2-2" in selected_mode and st.session_state[curr_step_key] == 2:
        with st.chat_message("assistant"):
            st.markdown("🚨 **역방향 위기 탐색 결과를 브리핑합니다.**")
            fk, fs, fr = st.session_state.rst_final_factors
            response = st.write_stream(ai_client.stream_rst_response(st.session_state.target_loss, fk, fs, fr))
            curr_msgs.append({"role": "assistant", "content": f"🚨 [RST 분석 결과]\n\n{response}"})
            st.session_state[curr_step_key] = 3

    if prompt := st.chat_input("AI 참모에게 지시하세요..."):
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
                current_params = st.session_state.scenario_data.get('parameters') if st.session_state[curr_step_key] >= 1 and 'scenario_data' in st.session_state else None
                res = ai_client.generate_dynamic_scenario(prompt, current_params)
                intent = res.get("intent", "irrelevant")
                
                if intent == "irrelevant" or intent == "explain":
                    st.write(res.get("rag_summary"))
                    curr_msgs.append({"role": "assistant", "content": res.get("rag_summary")})
                elif intent in ["new", "tuning"]:
                    st.session_state.scenario_data = res
                    df_params = pd.DataFrame(res['parameters'])
                    if len(df_params.columns) == 4: df_params.columns = ["리스크 팩터", "현재 수준", "최대 충격 (Target)", "충격 도달 기간"]
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
                msg = f"🔍 목표 손실 **{st.session_state.target_loss}억 원** 위기 경로 탐색을 시작합니다. 우측 화면을 확인해 주십시오."
                st.write(msg)
                curr_msgs.append({"role": "assistant", "content": msg})
                st.rerun()

# ==========================================
# 4. Right Panel: Visualization (우측 캔버스)
# ==========================================
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
        if step == 0: st.info("👈 좌측 대화창에 위기 시나리오를 지시해 주십시오.")
        elif step == 1:
            st.subheader("📝 시나리오 파라미터 검토 및 승인")
            edited_df = st.data_editor(st.session_state.final_scenario_df, use_container_width=True, hide_index=True)
            if st.button("✅ 기안 승인 및 Full Revaluation 실행", type="primary"):
                st.session_state.final_scenario_df = edited_df
                st.session_state[curr_step_key] = 2
                st.rerun()
        elif step == 2:
            st.subheader("📈 시계열 파급 분석")
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
            
            x, y = np.linspace(40, 100, 40), np.linspace(40, 100, 40)
            X, Y = np.meshgrid(x, y)
            Z = 100 + (X - 100)*0.3 + (Y - 100)*0.3 - np.where(X < 80, (80 - X)*1.5, 0) - np.where(Y < 60, (60 - Y)*2.5, 0) - 35*np.exp(-0.03*((X - 75)**2 + (Y - 55)**2))
            
            df_bonds_scaled = df_bonds.copy(); df_bonds_scaled['qty'] *= 100
            scen_history = []

            for i in range(steps_count):
                fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Blues', opacity=0.7)])
                fig.add_trace(go.Scatter3d(x=traj_x[:i+1], y=traj_y[:i+1], z=traj_z[:i+1], mode='lines', line=dict(color='orange', width=6)))
                fig.add_trace(go.Scatter3d(x=[traj_x[i]], y=[traj_y[i]], z=[traj_z[i]], mode='markers', marker=dict(size=10, color='red')))
                fig.update_layout(title=f"⏳ Day {int(i*2)}", scene=dict(xaxis_title='KOSPI 200', yaxis_title='삼성전자', camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.2))), margin=dict(l=0,r=0,b=0,t=30), height=350)
                chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"sim_anim_{i}")

                curr_mkt = base_mkt_state.copy()
                curr_mkt['KOSPI200_Close'] *= (traj_x[i]/100.0); curr_mkt['Samsung_Close'] *= (traj_y[i]/100.0)
                for tenor in ['KTB_6M', 'KTB_1Y', 'KTB_3Y', 'KTB_5Y', 'Corp_6M', 'Corp_1Y']: curr_mkt[tenor] += (rate_shocks[i] / 100.0)
                
                sim_bonds = revalue_bonds_multi(df_bonds_scaled, curr_mkt, base_mkt_state)
                sim_els = revalue_els_multi(df_els, curr_mkt, base_mkt_state)
                final_total_pnl = sim_bonds['pnl'].sum() + sim_els['pnl'].sum()

                row_data = {"단계": f"Day {i*2}", "KOSPI": f"{traj_x[i]:.1f}%", "삼성": f"{traj_y[i]:.1f}%", "금리": f"+{rate_shocks[i]:.0f}bp"}
                for _, r in sim_bonds.iterrows(): row_data[f"B_{r['name']}"] = f"{r['pnl']:,.0f}"
                for _, r in sim_els.iterrows(): row_data[f"E_{r['name']}"] = f"{r['pnl']:,.0f}"
                scen_history.append(row_data)

                metrics_placeholder.markdown(f"**실시간 통합 평가 손익:** `<span style='color:red;'>{final_total_pnl/100000000:,.1f}억 원</span>`", unsafe_allow_html=True)
                time.sleep(0.4)

            st.session_state.scen_history = scen_history; st.session_state.final_sim_pnl = final_total_pnl; st.session_state.scenario_fig = fig
            st.session_state[curr_step_key] = 3; st.rerun()
        elif step >= 3:
            st.plotly_chart(st.session_state.scenario_fig, use_container_width=True)
            st.markdown(f"**최종 통합 평가 손익:** `<span style='color:red;'>{st.session_state.final_sim_pnl/100000000:,.1f}억 원</span>`", unsafe_allow_html=True)
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
            df_bonds_scaled['qty'] *= 100
            
            def eval_pnl(k, s, r):
                mkt = base_mkt_state.copy()
                mkt['KOSPI200_Close'] *= (k / 100.0)
                mkt['Samsung_Close'] *= (s / 100.0)
                for tenor in ['KTB_6M', 'KTB_1Y', 'KTB_3Y', 'KTB_5Y', 'Corp_6M', 'Corp_1Y']: 
                    mkt[tenor] += (r / 100.0)
                tb = revalue_bonds_multi(df_bonds_scaled, mkt, base_mkt_state)
                te = revalue_els_multi(df_els, mkt, base_mkt_state)
                return tb['pnl'].sum() + te['pnl'].sum()

            target_pnl_raw = st.session_state.target_loss * 100000000
            ck, cs, cr = 100.0, 100.0, 0.0
            pk, ps, pr = [100.0], [100.0], [0.0]
            
            with st.spinner("최적화 알고리즘 역탐색 중..."):
                for _ in range(40):
                    curr_pnl = eval_pnl(ck, cs, cr)
                    if curr_pnl <= target_pnl_raw: break
                    
                    # Gradient(기울기) 계산 시 금리의 스케일을 주가와 맞춤
                    gk = max(0, curr_pnl - eval_pnl(ck - 1.0, cs, cr))
                    gs = max(0, curr_pnl - eval_pnl(ck, cs - 1.0, cr))
                    gr = max(0, curr_pnl - eval_pnl(ck, cs, cr + 10.0))
                    
                    total_g = gk + gs + gr + 1e-5
                    
                    # 편식 방지: 주가와 금리가 골고루 움직이도록 강제 스텝(-0.5, +2.0) 부여
                    ck = max(10.0, ck - (gk / total_g) * 3.0 - 0.5)
                    cs = max(10.0, cs - (gs / total_g) * 3.0 - 0.5)
                    cr += (gr / total_g) * 10.0 + 2.0
                    
                    pk.append(ck)
                    ps.append(cs)
                    pr.append(cr)
            
            idx = np.linspace(0, len(pk)-1, 10).astype(int)
            k_path, s_path, r_path = [pk[i] for i in idx], [ps[i] for i in idx], [pr[i] for i in idx]
            
            # 등고선(Contour) 차트 베이스
            k_grid, s_grid = np.linspace(50, 100, 10), np.linspace(50, 100, 10)
            K_MESH, S_MESH = np.meshgrid(k_grid, s_grid)
            history_data = []
            
            for s in range(10):
                ck, cs, cr = k_path[s], s_path[s], r_path[s]
                
                # 1. 레이더 차트 스케일 정상화 (40% 하락 시 100, 300bp 상승 시 100으로 정규화)
                r_vals = [
                    min(100, max(0, (100 - ck) * 2.5)), 
                    min(100, max(0, (100 - cs) * 2.5)), 
                    min(100, max(0, cr / 3.0)), 
                    min(100, max(0, (100 - ck) * 2.5))
                ]
                
                fig_radar = go.Figure(data=go.Scatterpolar(
                    r=r_vals, theta=['KOSPI 하락', '삼성전자 하락', '금리 상승', 'KOSPI 하락'], 
                    fill='toself', line=dict(color='red', width=2)
                ))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, height=350, margin=dict(l=30,r=30,t=30,b=30))
                radar_ph.plotly_chart(fig_radar, use_container_width=True, key=f"radar_{s}")
                
                # 2. 등고선 차트 복구
                fig_contour = go.Figure(data=go.Contour(z=(K_MESH+S_MESH), x=k_grid, y=s_grid, colorscale='RdBu'))
                fig_contour.add_trace(go.Scatter(x=k_path[:s+1], y=s_path[:s+1], mode='lines+markers', line=dict(color='#00FF00', width=4), marker=dict(size=8, color='black')))
                fig_contour.update_layout(height=350, margin=dict(l=30,r=30,t=30,b=30), showlegend=False)
                contour_ph.plotly_chart(fig_contour, use_container_width=True, key=f"contour_{s}")
                
                history_data.append({"Step": s+1, "KOSPI": f"{ck:.1f}%", "삼성전자": f"{cs:.1f}%", "금리": f"+{cr:.0f}bp"})
                time.sleep(0.3)

            # [중요] 애니메이션 종료 시 차트 객체를 저장하고 Step 2로 이동
            st.session_state.rst_radar = fig_radar
            st.session_state.rst_contour = fig_contour
            st.session_state.rst_history = history_data
            st.session_state.rst_final_factors = (ck, cs, cr)
            st.session_state[curr_step_key] = 2 
            st.rerun()

        elif step >= 2:
            st.subheader("◀ 역방향: 위기 좌표 탐색 (종료)")
            
            # [중요] 사라졌던 차트를 화면에 다시 고정시키는 코드 복구!
            c1, c2 = st.columns(2)
            if 'rst_radar' in st.session_state:
                c1.plotly_chart(st.session_state.rst_radar, use_container_width=True)
            if 'rst_contour' in st.session_state:
                c2.plotly_chart(st.session_state.rst_contour, use_container_width=True)
            
            st.info(f"🚨 최종 타격점 발견: KOSPI {st.session_state.rst_final_factors[0]:.1f}%, 삼성전자 {st.session_state.rst_final_factors[1]:.1f}%, 금리 +{st.session_state.rst_final_factors[2]:.0f}bp")
            
            st.markdown("#### 🔍 탐색 궤적 상세 데이터")
            if 'rst_history' in st.session_state:
                st.dataframe(pd.DataFrame(st.session_state.rst_history), use_container_width=True, hide_index=True)
