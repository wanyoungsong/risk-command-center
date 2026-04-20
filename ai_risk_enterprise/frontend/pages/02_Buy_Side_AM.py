import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np

# 우리가 만든 엔터프라이즈 모듈들 임포트
from data_access.loaders import get_buy_side_funds
from quant_engines.pricing import run_am_stress_test
from ai_agents.knowledge_graph import kg_client
from ai_agents.orchestrator import ai_client

st.set_page_config(layout="wide", page_title="Buy-Side AM 참모")

st.header("🏦 자산운용사(Buy-Side) AUM 방어 AI 참모")

# 1. 데이터 로드
df_base, fixed_costs = get_buy_side_funds()

# 2. UI 레이아웃 분할
col_chat, col_viz = st.columns([1.3, 1.7], gap="large")

with col_chat:
    st.subheader("🤖 AI 참모 대화창")
    if "am_messages" not in st.session_state:
        st.session_state.am_messages = [{"role": "assistant", "content": "운용사 AUM 방어 모드입니다. 스트레스 테스트 시나리오를 자연어로 입력해 주십시오. (예: 주가 20% 하락하고 환율 10% 급등하면 어떻게 돼?)"}]
    
    for msg in st.session_state.am_messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("시나리오를 지시하세요..."):
        st.session_state.am_messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.spinner("AI가 시나리오 분석 및 규정 매핑 중..."):
            # 1단계: AI 파라미터 추출
            params = ai_client.extract_am_scenario(prompt)
            
            # 2단계: Quant Engine 연산
            df_res, t_curr, t_final, c_op, n_op = run_am_stress_test(
                df_base, fixed_costs, params['stock'], params['fx'], params['rate']
            )
            
            # 3단계: 지식 그래프 조회
            kg_context = kg_client.get_am_kg_context(params['stock'], params['fx'])
            
            # 4단계: AI 브리핑 생성
            with st.chat_message("assistant"):
                response = st.write_stream(ai_client.stream_am_briefing(
                    t_curr - t_final, c_op, n_op, df_res['Outflow'].sum(), kg_context
                ))
                st.session_state.am_messages.append({"role": "assistant", "content": response})
                
                # 우측 캔버스를 위한 결과 저장
                st.session_state.am_last_result = (df_res, t_curr, t_final, c_op, n_op, params)
                st.rerun()

with col_viz:
    st.subheader("📊 AUM 및 영업이익 시뮬레이션 결과")
    if "am_last_result" in st.session_state:
        df_res, t_curr, t_final, c_op, n_op, params = st.session_state.am_last_result
        
        st.info(f"**적용 파라미터:** 주가 {params['stock']*100:.1f}%, 환율 {params['fx']*100:.1f}%, 금리 {params['rate']*100:.1f}%")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("최종 통합 AUM", f"{t_final:,.0f}억", f"{t_final - t_curr:,.0f}억 증감")
        
        # 영업이익이 마이너스(적자)로 전환되었는지 강조
        delta_op = n_op - c_op
        op_color = "normal" if n_op > 0 else "inverse"
        m2.metric("운용사 영업이익", f"{n_op:,.0f}억", f"{delta_op:,.0f}억", delta_color=op_color)
        
        m3.metric("펀드런 환매액", f"{df_res['Outflow'].sum():,.0f}억", delta_color="inverse")
        
        st.markdown("#### 📁 개별 펀드 스트레스 데이터")
        df_display = df_res[['Name', 'Current_AUM', 'Stress_Return', 'Outflow', 'Final_AUM', 'New_Rev']].copy()
        df_display['Stress_Return'] = (df_display['Stress_Return'] * 100).map("{:.2f}%".format)
        df_display['Outflow'] = df_display['Outflow'].map("{:,.1f}억".format)
        df_display['Final_AUM'] = df_display['Final_AUM'].map("{:,.0f}억".format)
        df_display['New_Rev'] = df_display['New_Rev'].map("{:,.1f}억".format)
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        st.info("👈 좌측 대화창에서 시나리오를 지시하시면 분석 결과가 렌더링 됩니다.")
