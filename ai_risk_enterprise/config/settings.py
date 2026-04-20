import streamlit as st

class Config:
    def __init__(self):
        self.GOOGLE_API_KEY = None
        self.NEO4J_URI = None
        self.NEO4J_USER = None
        self.NEO4J_PASSWORD = None
        self._load_config()

    def _load_config(self):
        try:
            from google.colab import userdata
            self.GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")
            self.NEO4J_URI = userdata.get("NEO4J_URI")
            self.NEO4J_USER = userdata.get("NEO4J_USER")
            self.NEO4J_PASSWORD = userdata.get("NEO4J_PASSWORD")
        except (ImportError, Exception):
            pass

        if not self.GOOGLE_API_KEY:
            try:
                self.GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
                self.NEO4J_URI = st.secrets["NEO4J_URI"]
                self.NEO4J_USER = st.secrets["NEO4J_USER"]
                self.NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]
            except (KeyError, FileNotFoundError, Exception):
                pass

config = Config()
