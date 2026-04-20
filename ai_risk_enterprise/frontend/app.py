import sys
import os
# 모듈 인식을 위한 경로 추가 (상위 폴더인 ai_risk_enterprise를 sys.path에 등록)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st

st.set_page_config(layout="wide", page_title="AI Risk Command Center")

def main():
    st.title("🤖 AI 리스크 통합 커맨드 센터")
    st.markdown("---")
    
    st.info("조사부(Investigation Dept.)의 연구 역량과 AI 기술을 결합한 엔터프라이즈 리스크 관리 플랫폼입니다.")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Sell-Side 솔루션")
        st.write("""
        - 전사 마켓 리스크 실시간 모니터링
        - ELS/채권 포트폴리오 그리스(Greeks) 분석
        - 역위기상황(Reverse Stress Test) 탐색 엔진
        """)
        if st.button("증권사 모드 바로가기"):
            st.switch_page("pages/01_Sell_Side_Risk.py")

    with col2:
        st.subheader("🏦 Buy-Side 솔루션")
        st.write("""
        - AUM 및 영업이익 스트레스 테스트
        - 펀드런(Fund-Run) 유출 시뮬레이션
        - 거시 경제 시나리오 파급 효과 분석
        """)
        if st.button("자산운용사 모드 바로가기"):
            st.switch_page("pages/02_Buy_Side_AM.py")

    st.markdown("---")
    st.caption("© 2026 AI Risk Agent. Powered by Investigation Department Research.")

if __name__ == "__main__":
    main()
