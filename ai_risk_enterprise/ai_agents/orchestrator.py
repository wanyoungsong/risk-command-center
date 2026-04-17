import google.generativeai as genai
import json
import time

class AIOrchestrator:
    def __init__(self):
        # API 키는 이미 config/settings.py에서 genai.configure()로 세팅되었으므로 바로 모델 호출 가능
        # 시연 속도와 퀄리티를 위해 가장 최적화된 모델 지정
        self.model = genai.GenerativeModel('gemini-2.5-flash')
    
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