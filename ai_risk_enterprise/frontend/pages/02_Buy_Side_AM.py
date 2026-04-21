import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
import time

# 엔터프라이즈 모듈 호출
from data_access.loaders import get_buy_side_funds
from quant_engines.pricing import run_am_stress_test
from ai_agents.knowledge_graph import kg_client
from ai_agents.orchestrator import ai_client

st.set_page_config(layout="wide", page_title="AM Risk Agent")

# 1. 초기 데이터 로드
df_base, fixed_costs = get_buy_side_funds()

modes = ["▶️ 1. 위기 시나리오 분석 (What-If)", "◀️ 2. 역방향 위기 탐색 (RST)"]
for m in modes:
    if f"msgs_{m}" not in st.session_state: st.session_state[f"msgs_{m}"] = []
    if f"step_{m}" not in st.session_state: st.session_state[f"step_{m}"] = 0

with st.sidebar:
    st.title("🏦 Asset Mgmt Agent")
    st.caption("자산운용사 포트폴리오 AI 리스크")
    selected_mode = st.radio("메뉴 선택", modes)

col_chat, col_viz = st.columns([1.3, 1.7], gap="large")
curr_msgs = st.session_state[f"msgs_{selected_mode}"]
curr_step_key = f"step_{selected_mode}"

with col_chat:
    st.subheader(f"{selected_mode.split('. ')[1]}")
    if not curr_msgs:
        intro = "운용사 AUM 방어 AI 참모입니다. 퍼펙트 스톰 시나리오를 지시하세요. (예: '주가 25% 폭락, 환율 15% 상승, 금리 2%p 인상 시나리오 돌려봐')" if "1" in selected_mode else "영업이익 적자 전환점(-800억 이하)을 유발하는 최악의 주가/환율 궤적을 역산합니다. (예: '영업손실 -100억 찍히는 경로 찾아줘')"
        curr_msgs.append({"role": "assistant", "content": intro})

    for msg in curr_msgs:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("AI에게 지시하세요..."):
        curr_msgs.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            if "1" in selected_mode:
                with st.spinner("시나리오 추출 중..."):
                    params = ai_client.extract_am_scenario(prompt)
                st.session_state.am_params = params
                st.session_state[curr_step_key] = 1
                msg = f"🔍 파라미터 추출: 주가 {params['stock']*100}%, 환율 {params['fx']*100}%, 금리 {params['rate']*100}%p. 우측에서 결과를 확인하세요."
                st.write(msg); curr_msgs.append({"role": "assistant", "content": msg})
                
                # [모듈화 반영] df_base와 fixed_costs를 인자로 넘김
                df_res, t_curr, t_final, c_op, n_op = run_am_stress_test(df_base, fixed_costs, params['stock'], params['fx'], params['rate'])
                st.session_state.am_results = (df_res, t_curr, t_final, c_op, n_op)
                
                st.markdown("🚨 **AI 경영진 브리핑**")
                kg_context = kg_client.get_am_kg_context(params['stock'], params['fx'])
                res = st.write_stream(ai_client.stream_am_briefing(t_curr - t_final, c_op, n_op, df_res, kg_context))
                
                curr_msgs.append({"role": "assistant", "content": f"🚨 [AI 브리핑]\n\n{res}"})
                st.rerun()
                
            elif "2" in selected_mode:
                nums = re.findall(r'-?\d+', prompt)
                target_op = float(nums[0]) if nums else -100.0
                st.session_state.target_op = target_op
                st.session_state[curr_step_key] = 1
                msg = f"🔍 목표 영업이익 **{target_op}억 원**에 도달하는 최단 경로를 엔진으로 탐색합니다."
                st.write(msg); curr_msgs.append({"role": "assistant", "content": msg})
                st.rerun()

with col_viz:
    if "1" in selected_mode and st.session_state[curr_step_key] >= 1:
        df_res, t_curr, t_final, c_op, n_op = st.session_state.am_results
        
        st.subheader("🏢 전사 AUM 및 영업이익 임팩트")
        m1, m2, m3 = st.columns(3)
        m1.metric("총 수탁고 (AUM)", f"{t_final:,.0f}억", f"{t_final - t_curr:,.0f}억 (펀드런 포함)")
        m2.metric("대규모 환매 (Fund Run)", f"-{df_res['Outflow'].sum():,.0f}억", "유동성 위기")
        m3.metric("예상 영업이익 (OP)", f"{n_op:,.0f}억", f"{n_op - c_op:,.0f}억 (고정비 {fixed_costs}억)", delta_color="inverse")
        
        fig = go.Figure(go.Waterfall(
            orientation="v", measure=["absolute", "relative", "relative", "total"],
            x=["현재 AUM", "시장가치 하락", "고객 대규모 환매", "최종 AUM"],
            y=[t_curr, -(t_curr - df_res['AUM_MTM'].sum()), -df_res['Outflow'].sum(), t_final],
            textposition="outside", text=[f"{t_curr:,.0f}", f"-{t_curr - df_res['AUM_MTM'].sum():,.0f}", f"-{df_res['Outflow'].sum():,.0f}", f"{t_final:,.0f}"]
        ))
        fig.update_layout(title="시나리오별 AUM 증감 폭포수 차트", height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### 📊 펀드별 스트레스 테스트 상세 내역")
        df_disp = df_res[['Hedge_Type', 'Stress_Return', 'Run_Rate', 'Outflow', 'Final_AUM']].copy()
        df_disp['Stress_Return'] = (df_disp['Stress_Return'] * 100).map("{:.1f}%".format)
        df_disp['Run_Rate'] = (df_disp['Run_Rate'] * 100).map("{:.1f}%".format)
        df_disp['Outflow'] = df_disp['Outflow'].map("{:,.0f}억".format)
        df_disp['Final_AUM'] = df_disp['Final_AUM'].map("{:,.0f}억".format)
        st.dataframe(df_disp, use_container_width=True)

    elif "2" in selected_mode and st.session_state[curr_step_key] >= 1:
        st.subheader("◀ 역방향: 영업이익 목표 역탐색 (RST)")
        target_op = st.session_state.target_op
        
        with st.spinner("비선형 유동성 리스크를 반영한 최악의 조합 탐색 중..."):
            stock_grid = np.linspace(0, -0.60, 30) 
            fx_grid = np.linspace(0, 0.50, 30)
            S_MESH, F_MESH = np.meshgrid(stock_grid, fx_grid)
            OP_MESH = np.zeros((30, 30))
            
            for i in range(30):
                for j in range(30):
                    # [모듈화 반영] df_base와 fixed_costs를 인자로 넘김
                    _, _, _, _, op = run_am_stress_test(df_base, fixed_costs, S_MESH[i, j], F_MESH[i, j], 0.0) 
                    OP_MESH[i, j] = op
            
            idx = np.unravel_index(np.argmin(np.abs(OP_MESH - target_op)), OP_MESH.shape)
            worst_s, worst_f, worst_op = S_MESH[idx], F_MESH[idx], OP_MESH[idx]
            time.sleep(1) 
            
        fig_contour = go.Figure(data=go.Contour(z=OP_MESH, x=stock_grid*100, y=fx_grid*100, colorscale='RdBu', colorbar=dict(title='영업이익')))
        fig_contour.add_trace(go.Scatter(x=[worst_s*100], y=[worst_f*100], mode='markers+text', marker=dict(size=14, color='yellow', symbol='star'), text=[f"목표 타격점<br>({worst_op:,.0f}억)"], textposition="top right"))
        fig_contour.update_layout(title="영업이익 손실 지형도 (주가 vs 환율)", xaxis_title="주가 하락률 (%)", yaxis_title="환율 상승률 (%)", height=400)
        
        st.plotly_chart(fig_contour, use_container_width=True)
        st.error(f"🚨 **엔진 탐색 결과:** 주가 **{worst_s*100:.1f}% 폭락** 및 환율 **{worst_f*100:+.1f}% 급등** 시 영업이익 **{worst_op:,.0f}억 원** 도달")
