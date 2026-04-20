import google.generativeai as genai
import json
import time

class AIOchestrator:
    def __init__(self):
        self.model_flash = genai.GenerativeModel('gemini-3.1-flash-lite-preview') # genai.GenerativeModel('gemini-2.5-flash')
        self.model_lite = genai.GenerativeModel('gemini-3.1-flash-lite-preview')

    def stream_ai_briefing(self, total_pnl, els_pnl, bond_pnl, top_driver, var_pct, kg_context):
        prompt = f"""너는 최고리스크책임자(CRO)를 보좌하는 수석 리스크 AI 참모야. 아래 팩트를 바탕으로 브리핑을 작성해.
        [데이터] 일간 P&L: {total_pnl/100000000:,.1f}억, ELS P&L: {els_pnl/100000000:,.1f}억, 채권 P&L: {bond_pnl/100000000:,.1f}억, 핵심동인: {top_driver}, VaR소진율: {var_pct:.1f}%
        [온톨로지 규정] {kg_context}
        인사말 없이 시작하고, 숫자를 명시하며 마크다운 불릿으로 3문단 이내로 작성해."""
        for chunk in self.model_lite.generate_content(prompt, stream=True):
            if chunk.text:
                yield chunk.text
                time.sleep(0.01)

    def stream_ai_prescription(self, dept_name, usage_pct, metric_name, exposure_amt, limit_amt, kg_context):
        prompt = f"""너는 수석 리스크 AI 참모야. 아래 부서별 한도 초과 상황에 대한 처방을 내려.
        [현황] 부서: {dept_name}, 지표: {metric_name}, 노출도: {exposure_amt:,.1f}억, 한도: {limit_amt:,.1f}억 (소진율 {usage_pct:.1f}%)
        [지침] {kg_context}
        인사말 없이, 규정을 근거로 액션 플랜을 3문단 이내로 강하게 권고해."""
        for chunk in self.model_lite.generate_content(prompt, stream=True):
            if chunk.text: yield chunk.text

    def generate_dynamic_scenario(self, user_input, current_params=None):
        context_str = f"\n[현재 적용된 파라미터]\n{current_params}" if current_params else ""
        
        prompt = f"""너는 금융기관 수석 리스크 AI 참모야. 다음 [사용자 입력]을 분석해 JSON으로 응답해.
        입력: {user_input} {context_str}
        
        [필수 지시사항]
        1. 사용자 입력의 의도(intent)를 다음 4가지 중 하나로 엄격하게 분류해:
           - "irrelevant": 리스크, 금융, 시나리오와 전혀 무관한 일상 대화.
           - "explain": 현재 시나리오를 추천한 근거, 이유, 지식그래프 논리 등을 묻는 단순 '질문'. (파라미터 변경 안 함)
           - "tuning": [현재 적용된 파라미터]의 수치를 변경, 완화, 강화하라는 명시적인 '수정 지시'.
           - "new": 완전히 새로운 위기 상황(예: "유가가 오르네")을 가정하여 새로운 파라미터를 도출해야 하는 상황.
           
        2. intent가 "irrelevant"이면 rag_summary에 "해당 내용은 제 분석 도메인과 관련이 없습니다. 😅"라고 작성.
        3. intent가 "explain"이면 rag_summary에 시나리오 추천 근거와 파급 논리(지식그래프 기반)를 친절하게 설명하고, parameters는 빈 배열 [] 반환.
        4. intent가 "new" 또는 "tuning"이면 인과관계를 rag_summary에 설명하고, 반드시 'KOSPI 200 지수', '삼성전자 주가', '국채/회사채 금리' 3가지 팩터가 모두 포함된 parameters 배열을 생성해.
        
        JSON 구조 예시: 
        {{
            "intent": "explain",
            "rag_summary": "답변 텍스트", 
            "kg_logic": "인과관계 텍스트", 
            "parameters": [
                {{"factor": "KOSPI 200 지수", "current": "100%", "target": "90%", "duration": "14일"}},
                {{"factor": "삼성전자 주가", "current": "100%", "target": "80%", "duration": "14일"}},
                {{"factor": "국채/회사채 금리", "current": "Base Rate", "target": "+50 bp", "duration": "14일"}}
            ]
        }}"""
        response = self.model_flash.generate_content(prompt, generation_config=genai.GenerationConfig(response_mime_type="application/json"))
        return json.loads(response.text)

    def stream_scenario_response(self, total_pnl, scenario_df_json):
        prompt = f"""너는 AI 참모야. 시뮬레이션 결과(파라미터: {scenario_df_json}, 최대손실: {total_pnl/100000000:,.1f}억)를 바탕으로 ELS 및 채권 데스크의 즉각적 액션 플랜을 사내 규정(가상)을 근거로 제시해."""
        for chunk in self.model_lite.generate_content(prompt, stream=True):
            if chunk.text: yield chunk.text

    def stream_rst_response(self, target_loss, k_val, s_val, r_val):
        prompt = f"""너는 AI 참모야. 목표손실 {target_loss:,.0f}억 역산 결과 (KOSPI: {k_val:.1f}%, 삼성전자: {s_val:.1f}%, 금리: {r_val:.0f}bp 충격)에 대한 경영진 시사점을 작성해."""
        for chunk in self.model_lite.generate_content(prompt, stream=True):
            if chunk.text: yield chunk.text

ai_client = AIOchestrator()
