import os
import google.generativeai as genai

class Settings:
    def __init__(self):
        self.GOOGLE_API_KEY = None
        self.NEO4J_URI = None
        self.NEO4J_USER = None
        self.NEO4J_PASSWORD = None
        self._load_config()

    def _load_config(self):
        # 1. 환경 변수 (서버 배포 시 사용)
        self.GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
        self.NEO4J_URI = os.environ.get("NEO4J_URI")
        self.NEO4J_USER = os.environ.get("NEO4J_USER")
        self.NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

        # 2. Streamlit Secrets (로컬/Streamlit Cloud 테스트용)
        if not self.GOOGLE_API_KEY:
            try:
                import streamlit as st
                self.GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
                self.NEO4J_URI = st.secrets.get("NEO4J_URI")
                self.NEO4J_USER = st.secrets.get("NEO4J_USER")
                self.NEO4J_PASSWORD = st.secrets.get("NEO4J_PASSWORD")
            except Exception:
                pass

        # Gemini API 자동 초기화
        if self.GOOGLE_API_KEY:
            genai.configure(api_key=self.GOOGLE_API_KEY)

# 싱글톤 인스턴스 생성 (다른 파일에서는 config 변수만 임포트해서 사용)
config = Settings()