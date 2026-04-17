from neo4j import GraphDatabase
from config.settings import config

class RiskKnowledgeGraph:
    def __init__(self):
        self.uri = config.NEO4J_URI
        self.user = config.NEO4J_USER
        self.password = config.NEO4J_PASSWORD

    def get_am_kg_context(self, stock_shock, fx_shock):
        """자산운용사(Buy-Side) 충격 조건에 맞는 사내 규정(GraphRAG) 추출"""
        # DB 접속 정보가 없으면 Fallback(가짜) 텍스트 리턴 (시연용 방어 로직)
        if not self.uri or not self.password:
            return "- [DB 연결 없음/시연용] 자산운용사 내부 규정상 환율 5% 초과 급등 시 마진콜 대비 달러 유동성 확보 요망."

        kg_context = ""
        try:
            with GraphDatabase.driver(self.uri, auth=(self.user, self.password)) as driver:
                with driver.session() as session:
                    keywords = []
                    if stock_shock <= -0.10: keywords.append("Stock_Crash")
                    if fx_shock >= 0.05: keywords.append("FX_Spike")
                    
                    if not keywords: 
                        return "✅ 현재 충격 수준은 사내 비상 규정 트리거(Trigger) 기준치 미달입니다."

                    query = """
                    MATCH (rf:RiskFactor)-[:TRIGGERS_POLICY]->(rule:ComplianceRule)
                    MATCH (fund:Fund)-[exp:EXPOSED_TO]->(rf)
                    WHERE rf.name IN $keywords
                    RETURN fund.name AS fund_name, exp.greek AS greek, exp.logic AS logic,
                           rule.name AS rule_name, rule.code AS rule_code, rule.action_plan AS action_plan
                    """
                    result = session.run(query, keywords=keywords)
                    for record in result:
                        kg_context += f"##### 🕸️ 데이터 리니지 (영향도)\n- **대상**: {record['fund_name']}\n- **손실 논리({record['greek']})**: {record['logic']}\n"
                        kg_context += f"##### 🚨 적용 사내 규정\n- **규정**: {record['rule_name']} ({record['rule_code']})\n- **AI 처방**: {record['action_plan']}\n\n"
        except Exception as e:
            # DB 연결 에러 시 프로그램이 죽지 않고 Fallback 반환
            kg_context = f"- [DB 조회 오류] 안전을 위해 리스크 한도 축소 권고. (Error: {str(e)})"
            
        return kg_context

# 다른 파일에서 쉽게 쓰기 위한 싱글톤 인스턴스
kg_client = RiskKnowledgeGraph()