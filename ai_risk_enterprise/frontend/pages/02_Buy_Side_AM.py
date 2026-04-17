import streamlit as st
import pandas as pd
# 우리가 만든 엔터프라이즈 모듈들만 임포트합니다.
from data_access.loaders import get_buy_side_funds
from quant_engines.pricing import run_am_stress_test
from ai_agents.orchestrator import ai_client
from ai_agents.knowledge_graph import kg_client

st.set_page_config(layout="wide")

st.header("🏦 자산운용사 AUM 방어 AI 참모")

# 1. 데이터 로드 (Data Access Layer 호출)
df_base, fixed_costs = get_buy_side_funds()

# 2. UI 레이아웃
col_chat, col_viz = st.columns([1.3, 1.7], gap="large")

with col_chat:
    st.subheader("AI 참모 대화창")
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "시나리오를 입력하세요."}]
    
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("주가 20% 하락 시나리오 분석해줘"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # AI Agent 호출 (Orchestrator 레이어)
        with st.spinner("AI가 분석 중..."):
            params = ai_client.extract_am_scenario(prompt)
            
            # 엔진 연산 (Quant Engine 레이어)
            df_res, t_curr, t_final, c_op, n_op = run_am_stress_test(
                df_base, fixed_costs, params['stock'], params['fx'], params['rate']
            )
            
            # 지식 그래프 조회 (Knowledge Graph 레이어)
            kg_context = kg_client.get_am_kg_context(params['stock'], params['fx'])
            
            # 브리핑 생성
            with st.chat_message("assistant"):
                response = st.write_stream(ai_client.stream_am_briefing(
                    t_curr - t_final, c_op, n_op, df_res['Outflow'].sum(), kg_context
                ))
                st.session_state.messages.append({"role": "assistant", "content": response})
                # 시연용 결과 저장
                st.session_state.last_result = (df_res, t_curr, t_final, c_op, n_op)

with col_viz:
    st.subheader("시각화 캔버스")
    if "last_result" in st.session_state:
        df_res, t_curr, t_final, c_op, n_op = st.session_state.last_result
        m1, m2, m3 = st.columns(3)
        m1.metric("최종 AUM", f"{t_final:,.0f}억")
        m2.metric("영업이익", f"{n_op:,.0f}억")
        m3.metric("환매액", f"{df_res['Outflow'].sum():,.0f}억")
        st.dataframe(df_res, use_container_width=True)
    else:
        st.info("시나리오를 실행하면 분석 결과가 여기에 표시됩니다.")