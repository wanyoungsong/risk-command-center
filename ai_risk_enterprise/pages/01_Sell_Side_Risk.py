import streamlit as st
import pandas as pd

# 🛠️ 우리가 만든 엔터프라이즈 모듈 3대장 임포트
from data_access.loaders import get_market_data, get_sell_side_portfolio
from quant_engines.pricing import calculate_parametric_var, revalue_bonds_multi, revalue_els_multi
from ai_agents.orchestrator import ai_client
from ai_agents.knowledge_graph import kg_client

st.set_page_config(layout="wide")

st.header("📈 증권사(Sell-Side) 전사 마켓 리스크 AI 참모")

# ==========================================
# 1. Data Load (데이터 계층)
# ==========================================
with st.spinner("Market Data & Position 원장 로딩 중..."):
    df_market = get_market_data(30)
    df_bonds, df_els = get_sell_side_portfolio()
    
    base_mkt_state = df_market.iloc[-2].to_dict()
    current_mkt_state = df_market.iloc[-1].to_dict()

# ==========================================
# 2. UI Layout (프론트엔드 계층)
# ==========================================
col_chat, col_viz = st.columns([1.3, 1.7], gap="large")

with col_chat:
    st.subheader("🤖 AI 참모 대화창")
    if "sell_msgs" not in st.session_state:
        st.session_state.sell_msgs = [{"role": "assistant", "content": "전사 마켓 리스크 현황이 우측 대시보드에 업데이트되었습니다. 시나리오 분석이나 경영진 브리핑을 지시해 주십시오."}]

    # 기존 대화 내역 출력
    for msg in st.session_state.sell_msgs:
        st.chat_message(msg["role"]).write(msg["content"])

    # 사용자 입력 처리
    if prompt := st.chat_input("예: 오늘자 전사 리스크 브리핑 해줘"):
        st.session_state.sell_msgs.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.spinner("수학 엔진 연산 및 AI 규정 맵핑 중..."):
            # 🧮 1단계: 엔진 연산 (Quant Engines)
            bonds_res = revalue_bonds_multi(df_bonds, current_mkt_state, base_mkt_state)
            els_res = revalue_els_multi(df_els, current_mkt_state, base_mkt_state)
            
            daily_bond_pnl = bonds_res['pnl'].sum()
            daily_els_pnl = els_res['pnl'].sum()
            total_pnl = daily_bond_pnl + daily_els_pnl
            
            var_amt, var_sens = calculate_parametric_var(df_bonds, df_els, df_market, 0.99)
            top_driver = var_sens.abs().idxmax()

            # 🕸️ 2단계: 지식 그래프 조회 (Knowledge Graph)
            # (참고: knowledge_graph.py에 증권사용 메서드를 추가했다고 가정)
            # kg_context = kg_client.get_sell_side_kg_context(top_driver) 
            kg_context = f"- **적용 규정**: 리스크 관리 규정 제75조\n- **AI 처방**: {top_driver} 변동성 확대에 따른 즉각적인 헤지 포지션(장내파생) 진입 요망."

            # 🧠 3단계: AI 브리핑 스트리밍 (Orchestrator)
            with st.chat_message("assistant"):
                # 시연을 위해 orchestrator에 증권사용 프롬프트 함수를 연결하거나 직접 출력
                response = f"🚨 **전사 리스크 통합 브리핑**\n\n현재 일간 통합 P&L은 **{total_pnl/100000000:,.1f}억 원**이며, 99% 신뢰수준 VaR는 **{var_amt/100000000:,.1f}억 원**입니다. 포트폴리오의 가장 큰 취약점은 **{top_driver}**입니다.\n\n{kg_context}"
                st.write(response)
                st.session_state.sell_msgs.append({"role": "assistant", "content": response})

            # 시각화 캔버스를 위한 상태 저장
            st.session_state.sell_dashboard = (total_pnl, var_amt, top_driver, daily_bond_pnl, daily_els_pnl)
            st.rerun()

with col_viz:
    st.subheader("📊 리스크 통합 대시보드")
    if "sell_dashboard" in st.session_state:
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

        st.markdown("#### 🔍 상세 포지션 원장 (Audit)")
        st.write("채권(Bonds) Book")
        st.dataframe(df_bonds[['name', 'tenor', 'curve', 'qty']], use_container_width=True, height=150)
        st.write("파생(ELS) Book")
        st.dataframe(df_els[['name', 'asset1', 'asset2', 'ki_barrier', 'qty']], use_container_width=True, height=150)
    else:
        st.info("👈 좌측 대화창에서 AI 참모에게 분석을 지시해 주세요.")