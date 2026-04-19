import google.generativeai as genai
import json
import time

class AIOrchestrator:
    def __init__(self):
        # API 키는 이미 config/settings.py에서 genai.configure()로 세팅되었으므로 바로 모델 호출 가능
        # 최적화된 모델 지정 (시연용으로는 속도가 빠른 flash 모델 권장)
        self.model = genai.GenerativeModel('gemini-3.1-flash-lite-preview') # gemini-2.5-flash
        self.stream_model = genai.GenerativeModel('gemini-3.1-flash-lite-preview')
    
    # ==========================================
    # [증권사 Sell-Side] AI 참모 프롬프트
    # ==========================================
    # 기존 파라미터 맨 앞에 user_input 추가!
    def stream_sell_side_briefing(self, user_input, total_pnl, els_pnl, bond_pnl, top_driver, var_pct, kg_context):
        prompt = f"""너는 최고리스크책임자(CRO)를 보좌하는 수석 리스크 AI 참모야.
        [사용자 입력] {user_input}
        
        [행동 지침]
        1. [사용자 입력]이 "브리핑 해줘", "오늘 시장 어때?", "손익 얼마야?" 등 리스크 시황을 묻는 질문이라면 아래 [데이터]와 [온톨로지 규정]을 활용해 숫자를 명시하며 3문단 이내로 브리핑해.
        2. [사용자 입력]이 "넌 누구냐?", "오늘 날씨 어때?" 등 리스크 도메인과 전혀 무관한 질문이라면, 데이터를 무시하고 "저는 전사 마켓 리스크를 감시하는 AI 참모입니다. 해당 내용은 제 분석 도메인이 아닙니다."라고 정중히 선을 긋는 답변만 해.

        [데이터] 일간 P&L: {total_pnl/100000000:,.1f}억, ELS P&L: {els_pnl/100000000:,.1f}억, 채권 P&L: {bond_pnl/100000000:,.1f}억, 핵심동인: {top_driver}, VaR소진율: {var_pct:.1f}%
        [온톨로지 규정] {kg_context}"""
        
        for chunk in self.stream_model.generate_content(prompt, stream=True):
            if chunk.text:
                yield chunk.text
                time.sleep(0.01)

    def stream_compliance_prescription(self, dept_name, usage_pct, metric_name, exposure_amt, limit_amt, kg_context):
        prompt = f"""너는 수석 리스크 AI 참모야. 아래 부서별 한도 초과 상황에 대한 처방을 내려.
        [현황] 부서: {dept_name}, 지표: {metric_name}, 노출도: {exposure_amt:,.1f}억, 한도: {limit_amt:,.1f}억 (소진율 {usage_pct:.1f}%)
        [지침] {kg_context}
        인사말 없이, 규정을 근거로 액션 플랜을 3문단 이내로 강하게 권고해."""
        
        for chunk in self.stream_model.generate_content(prompt, stream=True):
            if chunk.text: 
                yield chunk.text
                time.sleep(0.01)

    def generate_dynamic_scenario(self, user_input, current_params=None):
        context_str = f"\n[현재 적용된 파라미터]\n{current_params}" if current_params else ""
        
        prompt = f"""너는 금융기관 수석 리스크 AI 참모야. 다음 [사용자 입력]을 분석해 JSON으로 응답해.
        입력: {user_input} {context_str}
        
        [필수 지시사항]
        1. 사용자 입력의 의도(intent)를 "irrelevant", "explain", "tuning", "new" 중 하나로 엄격하게 분류해.
        2. intent가 "explain"이면 시나리오 추천 근거를 설명하고, parameters는 빈 배열 [] 반환.
        3. intent가 "new" 또는 "tuning"이면 인과관계를 설명하고, 'KOSPI 200 지수', '삼성전자 주가', '국채/회사채 금리' 팩터가 포함된 parameters 배열을 생성해.
        
        JSON 구조 예시: 
        {{
            "intent": "explain",
            "rag_summary": "답변 텍스트", 
            "parameters": [
                {{"factor": "KOSPI 200 지수", "current": "100%", "target": "90%", "duration": "14일"}}
            ]
        }}"""
        
        response = self.model.generate_content(
            prompt, 
            generation_config=genai.GenerationConfig(response_mime_type="application/json")
        )
        return json.loads(response.text)

    # ==========================================
    # [자산운용사 Buy-Side] AI 참모 프롬프트
    # ==========================================
    def extract_am_scenario(self, prompt_text):
        """사용자 자연어에서 시나리오 충격 파라미터를 JSON으로 추출"""
        prompt = f"""자산운용사 리스크 AI 참모로서, [입력]을 분석해 JSON 파라미터를 추출해.
        입력: {prompt_text}
        조건: 주가(stock), 환율(fx), 금리(rate) 충격량을 소수로 변환해(예: -25% 하락 -> -0.25, 15% 상승 -> 0.15). 언급이 없으면 0.0으로 둬.
        JSON 형식: {{"is_relevant": true, "summary": "...", "stock": -0.25, "fx": 0.15, "rate": 0.02}}"""
        
        try:
            # JSON 형태로만 응답하도록 강제 (Gemini의 response_mime_type 기능 활용)
            res = self.model.generate_content(
                prompt, 
                generation_config=genai.GenerationConfig(response_mime_type="application/json")
            )
            return json.loads(res.text)
        except Exception:
            # AI 호출 실패 시 시연이 멈추지 않도록 기본값(Fallback) 반환
            return {"is_relevant": True, "stock": -0.25, "fx": 0.15, "rate": 0.02, "summary": "API 오류: 퍼펙트 스톰 기본 적용"}

    def stream_am_briefing(self, tot_drop, curr_op, new_op, outflow_tot, kg_context):
        """계산 엔진의 결과와 지식 그래프를 융합하여 경영진 브리핑을 스트리밍(생성)"""
        prompt = f"""너는 최고투자책임자(CIO)를 보좌하는 운용사 전용 수석 리스크 AI 참모야.
        
        [시뮬레이션 팩트] 
        - 자산가치 손실액: {tot_drop:,.0f}억
        - 뱅크런(Fund Run) 유출액: {outflow_tot:,.0f}억
        - 영업이익 변화: {curr_op:,.0f}억 -> {new_op:,.0f}억 (적자 전환 여부 강조)
        
        [사내 규정 온톨로지 (GraphRAG)]
        {kg_context}

        [지시사항]
        위 숫자를 바탕으로 충격의 심각성을 경고하고, 반드시 [사내 규정 온톨로지]에 명시된 'Rule Name'과 'AI 처방(Action Plan)'을 명시하여 각 펀드 데스크가 당장 취해야 할 행동을 마크다운으로 프로페셔널하게 지시해. 인사말은 생략해."""
        
        # UI로 글자를 한 글자씩 밀어내기 위해 generator(yield) 사용
        for chunk in self.model.generate_content(prompt, stream=True):
            if chunk.text:
                yield chunk.text
                time.sleep(0.01) # 너무 빠르면 읽기 힘드므로 약간의 딜레이

# 다른 파일에서 쉽게 쓰기 위한 싱글톤 인스턴스
ai_client = AIOrchestrator()
