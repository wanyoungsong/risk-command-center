from neo4j import GraphDatabase
import streamlit as st
from config.settings import config

class KnowledgeGraphClient:
    def __init__(self):
        self.uri = config.NEO4J_URI
        self.user = config.NEO4J_USER
        self.password = config.NEO4J_PASSWORD

    def get_knowledge_graph_context(self, risk_driver):
        if not self.uri or not self.password: return self._fallback_kg_context(risk_driver)
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
                    if not records: return self._fallback_kg_context(risk_driver)

                    for record in records:
                        kg_context += f"##### 🕸️ 데이터 리니지 인과관계\n- **대상 자산군**: {record['asset']}\n- **민감도(Greeks)**: {record['greek']}\n- **손실 발생 논리**: {record['logic']}\n"
                        kg_context += f"##### 🚨 적용 규정 및 지침\n- **규정**: {record['rule_name']} ({record['rule_code']})\n- **대응 조치**: {record['action_plan']}\n\n"
        except Exception:
            kg_context = self._fallback_kg_context(risk_driver)
        return kg_context

    def _fallback_kg_context(self, risk_driver):
        if "Vol" in risk_driver:
            return "- **대상 자산군**: Derivatives (ELS 자체 헤지북)\n- **민감도**: Vega\n- **손실 논리**: 변동성 급등 시 헤지 비용 기하급수적 팽창\n- **적용 규정**: Article 14-3\n- **대응 조치**: Vega 중립을 위한 옵션 양매수 헤지 비중 즉각 확대 요망"
        else:
            return "- **대상 자산군**: Fixed_Income (채권 매수북)\n- **민감도**: Rho\n- **손실 논리**: 금리 상승 시 평가손실 발생\n- **적용 규정**: Article 75-1\n- **대응 조치**: 듀레이션 갭 축소를 위한 국채선물 매도 확대 요망"

    @st.cache_data(ttl=3600)
    def get_dynamic_risk_limits(_self):
        limits = {"전사 VaR 한도": 1500.0, "금리 민감도(Rho) 한도": 40.0, "변동성 민감도(Vega) 한도": 30.0}
        if not _self.uri or not _self.password: return limits
        try:
            with GraphDatabase.driver(_self.uri, auth=(_self.user, _self.password)) as driver:
                with driver.session() as session:
                    result = session.run("MATCH (l:RiskLimit) RETURN l.name AS name, l.limit_value AS val")
                    for record in result: limits[record["name"]] = float(record["val"])
        except Exception: pass
        return limits

    def get_compliance_graph_context(self, dept_code, usage_pct):
        if not self.uri or not self.password: return self._fallback_compliance_context(dept_code, usage_pct)
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
                    if not records: return self._fallback_compliance_context(dept_code, usage_pct)
                    for record in records:
                        kg_context += f"- **규정명**: {record['rule_name']}\n- **AI 조치 처방**: {record['action_plan']}\n\n"
        except Exception:
            kg_context = self._fallback_compliance_context(dept_code, usage_pct)
        return kg_context

    def _fallback_compliance_context(self, dept_code, usage_pct):
        if usage_pct < 90: return "✅ 정상 범위"
        if dept_code == 'ENTERPRISE': return "- **규정명**: 전사 한도 초과 대응\n- **AI 조치 처방**: 즉각적인 포지션 축소, CRO 대면 보고 소집" if usage_pct >= 100 else "- **규정명**: 경고 대응\n- **AI 조치 처방**: 신규 리스크 테이킹 즉각 중지"
        elif dept_code == 'BOND_DESK': return "- **규정명**: 채권 한도 초과 대응\n- **AI 조치 처방**: IRS 페이 포지션 구축 또는 장기채 즉시 매도"
        elif dept_code == 'ELS_DESK': return "- **규정명**: ELS 비선형 한도 초과\n- **AI 조치 처방**: 롤오버 중지 및 베가 중립 헤지 기계적 확대"
        return "✅ 정상 범위"

kg_client = KnowledgeGraphClient()
