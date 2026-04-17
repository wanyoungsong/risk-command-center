from neo4j import GraphDatabase
from config.settings import config

class RiskKnowledgeGraph:
    def __init__(self):
        self.uri = config.NEO4J_URI
        self.user = config.NEO4J_USER
        self.password = config.NEO4J_PASSWORD

    # ----------------------------------------------------
    # [증권사] 1. 마켓 리스크 드라이버 기반 규정 조회
    # ----------------------------------------------------
    def get_sell_side_kg_context(self, risk_driver):
        if not self.uri or not self.password:
            return self._fallback_sell_side_kg(risk_driver)
        
        kg_context = ""
        try:
            with GraphDatabase.driver(self.uri, auth=(self.user, self.password)) as driver:
                with driver.session() as session:
                    query = """
                    MATCH (rf:RiskFactor)-[:TRIGGERS_POLICY]->(rule:ComplianceRule)
                    MATCH (a:AssetClass)-[exp:EXPOSED_TO]->(rf)
                    WHERE rf.name CONTAINS $keyword OR rf.desc CONTAINS $keyword
                    RETURN a.name AS asset, exp.greek AS greek, exp.logic AS logic,
                           rule.name AS rule_name, rule.code AS rule_code, rule.action_plan AS action_plan
                    """
                    kw = "Volatility" if "Vol" in risk_driver else "Intensity" if "Intensity" in risk_driver else "Interest_Rate"
                    result = session.run(query, keyword=kw)
                    records = list(result)
                    
                    if not records: return self._fallback_sell_side_kg(risk_driver)

                    for record in records:
                        kg_context += f"##### 🕸️ 데이터 리니지 인과관계\n- **대상 자산군**: {record['asset']}\n- **민감도(Greeks)**: {record['greek']}\n- **손실 발생 논리**: {record['logic']}\n"
                        kg_context += f"##### 🚨 적용 규정 및 지침\n- **규정**: {record['rule_name']} ({record['rule_code']})\n- **대응 조치**: {record['action_plan']}\n\n"
        except Exception:
            kg_context = self._fallback_sell_side_kg(risk_driver)
        return kg_context

    def _fallback_sell_side_kg(self, risk_driver):
        if "Vol" in risk_driver:
            return "- **대상 자산군**: Derivatives (ELS 자체 헤지북)\n- **민감도**: Vega\n- **손실 논리**: 변동성 급등 시 헤지 비용 기하급수적 팽창\n- **적용 규정**: Article 14-3\n- **대응 조치**: Vega 중립을 위한 옵션 양매수 헤지 비중 즉각 확대 요망"
        else:
            return "- **대상 자산군**: Fixed_Income (채권 매수북)\n- **민감도**: Rho\n- **손실 논리**: 금리 상승 시 평가손실 발생\n- **적용 규정**: Article 75-1\n- **대응 조치**: 듀레이션 갭 축소를 위한 국채선물 매도 확대 요망"

    # ----------------------------------------------------
    # [증권사] 2. 부서별 한도 소진율 기반 규정 조회
    # ----------------------------------------------------
    def get_compliance_kg_context(self, dept_code, usage_pct):
        if not self.uri or not self.password: 
            return self._fallback_compliance_kg(dept_code, usage_pct)
        
        kg_context = ""
        try:
            with GraphDatabase.driver(self.uri, auth=(self.user, self.password)) as driver:
                with driver.session() as session:
                    th_kw = "100% 이상" if usage_pct >= 100 else "90% 이상" if usage_pct >= 90 else "정상"
                    if th_kw == "정상": return "✅ 현재 한도 소진율은 정상 범위입니다."
                    
                    query = """
                    MATCH (d:Department {code: $dept_code})-[:MONITORS]->(l:RiskLimit)-[tp:TRIGGERS_POLICY]->(rule:ComplianceRule)
                    WHERE tp.threshold CONTAINS $threshold_kw
                    RETURN l.name AS limit_name, l.metric AS metric, tp.action_plan AS action_plan, rule.name AS rule_name
                    """
                    result = session.run(query, dept_code=dept_code, threshold_kw=th_kw)
                    records = list(result)
                    
                    if not records: return self._fallback_compliance_kg(dept_code, usage_pct)
                    for record in records:
                        kg_context += f"- **규정명**: {record['rule_name']}\n- **AI 조치 처방**: {record['action_plan']}\n\n"
        except Exception:
            kg_context = self._fallback_compliance_kg(dept_code, usage_pct)
        return kg_context

    def _fallback_compliance_kg(self, dept_code, usage_pct):
        if usage_pct < 90: return "✅ 정상 범위"
        if dept_code == 'ENTERPRISE': return "- **규정명**: 전사 한도 초과 대응\n- **AI 조치 처방**: 즉각적인 포지션 축소, CRO 대면 보고 소집" if usage_pct >= 100 else "- **규정명**: 경고 대응\n- **AI 조치 처방**: 신규 리스크 테이킹 즉각 중지"
        elif dept_code == 'BOND_DESK': return "- **규정명**: 채권 한도 초과 대응\n- **AI 조치 처방**: IRS 페이 포지션 구축 또는 장기채 즉시 매도"
        elif dept_code == 'ELS_DESK': return "- **규정명**: ELS 비선형 한도 초과\n- **AI 조치 처방**: 롤오버 중지 및 베가 중립 헤지 기계적 확대"
        return "✅ 정상 범위"

    # ----------------------------------------------------
    # [자산운용사] 3. 펀드 충격 기반 규정 조회 (기존 유지)
    # ----------------------------------------------------
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