import streamlit as st
from config.settings import config

st.set_page_config(layout="wide", page_title="AI Risk Command Center")

def main():
    st.title("🤖 AI 리스크 통합 커맨드 센터")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🚀 프로젝트 비전")
        st.write("본 시스템은 단순한 계산기를 넘어, 시장의 흐름을 이해하고 사내 규정에 근거하여 처방을 내리는 'AI 리스크 참모'를 지향합니다.")
    
    with col2:
        st.info("💡 좌측 메뉴에서 '증권사(Sell-Side)' 또는 '자산운용사(Buy-Side)' 업무 모드를 선택하세요.")

    st.sidebar.success("접속 상태: 정상")
    if config.NEO4J_URI:
        st.sidebar.write("🔗 Knowledge Graph 연결됨")

if __name__ == "__main__":
    main()