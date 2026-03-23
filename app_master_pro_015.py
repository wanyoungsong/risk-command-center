%%writefile app_master_pro_015.py
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import plotly.graph_objects as go
from pyvis.network import Network
import time
from datetime import datetime
import pandas as pd

# --- 1. 세션 상태 관리 (페이지 이동 시 초기화 방지) ---
if 'scenario_step' not in st.session_state:
    st.session_state.scenario_step = 0
if 'rst_step' not in st.session_state:
    st.session_state.rst_step = 0
if 'batch_step' not in st.session_state:
    st.session_state.batch_step = 0
if 'incident_step' not in st.session_state:
    st.session_state.incident_step = 0

# --- 2. 사이드바 네비게이션 ---
st.set_page_config(layout="wide", page_title="Enterprise Risk Command Center")
with st.sidebar:
    st.title("🏦 Risk Command Center")
    st.markdown("---")
    page = st.radio("Navigation", [
        "1. 전사 리스크 브리핑",
        "2. 스트레스 테스트 데스크",
        "3. 시스템 오퍼레이션",
        "4. 장애 대응 가이드 에이전트"
    ])
    st.markdown("---")
    # st.caption("System Status: Normal")
    # st.caption("Last Updated: 08:00 AM")
    st.caption("System Status: Normal (Warning in Batch)")
    st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# --- 3. 공통 데이터 셋업 (Global Variables) ---
bond_portfolio = [
    {"bond_id": "B01", "name": "국고채_3개월", "tenor": 0.25, "face_value": 10000, "base_rate": 0.030, "qty": 1000},
    {"bond_id": "B02", "name": "국고채_6개월", "tenor": 0.50, "face_value": 10000, "base_rate": 0.031, "qty": 1500},
    {"bond_id": "B03", "name": "국고채_1년_A", "tenor": 1.00, "face_value": 10000, "base_rate": 0.032, "qty": 2000},
    {"bond_id": "B04", "name": "회사채_1년_B", "tenor": 1.00, "face_value": 10000, "base_rate": 0.038, "qty": 1200},
    {"bond_id": "B05", "name": "국고채_1.5년", "tenor": 1.50, "face_value": 10000, "base_rate": 0.033, "qty": 800},
    {"bond_id": "B06", "name": "국고채_2년_A", "tenor": 2.00, "face_value": 10000, "base_rate": 0.034, "qty": 2500},
    {"bond_id": "B07", "name": "금융채_2년_B", "tenor": 2.00, "face_value": 10000, "base_rate": 0.039, "qty": 3000},
    {"bond_id": "B08", "name": "국고채_2.5년", "tenor": 2.50, "face_value": 10000, "base_rate": 0.035, "qty": 1000},
    {"bond_id": "B09", "name": "국고채_3년_A", "tenor": 3.00, "face_value": 10000, "base_rate": 0.036, "qty": 5000},
    {"bond_id": "B10", "name": "회사채_3년_B", "tenor": 3.00, "face_value": 10000, "base_rate": 0.042, "qty": 2000},
]
df_bonds = pd.DataFrame(bond_portfolio)

els_portfolio = [
    {"els_id": "E01", "name": "ELS_안정형_KI50_A", "ki_barrier": 50, "qty": 1000},
    {"els_id": "E02", "name": "ELS_안정형_KI50_B", "ki_barrier": 50, "qty": 1200},
    {"els_id": "E03", "name": "ELS_중립형_KI55_A", "ki_barrier": 55, "qty": 800},
    {"els_id": "E04", "name": "ELS_중립형_KI55_B", "ki_barrier": 55, "qty": 1500},
    {"els_id": "E05", "name": "ELS_중립형_KI60_A", "ki_barrier": 60, "qty": 2000},
    {"els_id": "E06", "name": "ELS_중립형_KI60_B", "ki_barrier": 60, "qty": 1000},
    {"els_id": "E07", "name": "ELS_위험형_KI65_A", "ki_barrier": 65, "qty": 900},
    {"els_id": "E08", "name": "ELS_위험형_KI65_B", "ki_barrier": 65, "qty": 1100},
    {"els_id": "E09", "name": "ELS_고위험_KI70_A", "ki_barrier": 70, "qty": 3000},
    {"els_id": "E10", "name": "ELS_고위험_KI70_B", "ki_barrier": 70, "qty": 2500},
]
df_els = pd.DataFrame(els_portfolio)

def revalue_bonds(df, rate_shock_bps):
    """
    현금흐름 할인(DCF) 공식을 직접 적용하여 금리 충격 시나리오 재평가
    rate_shock_bps: 금리 변동폭 (예: 100 -> 100bp(1.0%) 상승)
    """
    results = df.copy()
    shock_rate = rate_shock_bps / 10000.0  # bp를 소수로 변환

    # 기존 가격 산출: P = FV / (1 + r)^t
    results['old_price'] = results['face_value'] / ((1 + results['base_rate']) ** results['tenor'])

    # 충격 반영 가격 산출
    new_rate = results['base_rate'] + shock_rate
    results['new_price'] = results['face_value'] / ((1 + new_rate) ** results['tenor'])

    # 손익 계산
    results['price_change'] = results['new_price'] - results['old_price']
    results['pnl'] = results['price_change'] * results['qty']
    return results

def revalue_els(df, current_k_index, current_h_index):
    """
    기초 자산 지수 변화 시 배리어 근접도에 따른 비선형적 가치 하락을 재평가
    초기 발행 시점의 기준 지수를 100으로 가정
    """
    results = df.copy()
    base_price = 10000

    # 두 기초자산 중 더 낮은 성과를 기록한 지수(Worst-Performer) 기준
    worst_perf = np.minimum(current_k_index, current_h_index)
    distance_to_ki = worst_perf - results['ki_barrier']

    # 1. 델타(Delta) 성향의 선형적 가격 변화
    linear_impact = (worst_perf - 100) * 45

    # 2. 감마(Gamma) 성향의 비선형적 가격 변화 (배리어 근접 시 손실 가속화)
    # distance_to_ki가 작아질수록 비선형적 가치 하락폭(Penalty)이 기하급수적으로 커짐
    nonlinear_penalty = np.where(
        distance_to_ki > 0,
        2000 * np.exp(-0.15 * distance_to_ki),               # KI 터치 이전: 점진적 손실 확대
        3000 + (results['ki_barrier'] - worst_perf) * 100    # KI 터치 이후: 배리어 하회에 따른 직접적 자본 손실
    )

    # 새로운 산출 가격
    results['new_price'] = base_price + linear_impact - nonlinear_penalty

    # 초기 정상 상태(지수 100) 대비 현재 가격의 변화량 계산
    initial_penalty = np.where(100 - results['ki_barrier'] > 0, 2000 * np.exp(-0.15 * (100 - results['ki_barrier'])), 0)
    old_price = base_price - initial_penalty

    results['price_change'] = results['new_price'] - old_price
    results['pnl'] = results['price_change'] * results['qty']

    return results

# ==========================================
# [페이지 1] 전사 리스크 브리핑 & 한도 관리
# ==========================================
if page == "1. 전사 리스크 브리핑":
    st.subheader("전사 마켓 리스크 현황 및 AI 원인 규명 브리핑")
    st.markdown("---")

    # 탭 구성: 브리핑 vs 한도 관리
    tab1, tab2 = st.tabs(["📊 전사 마켓 리스크 브리핑", "🚨 부서별 한도 관리 및 AI 처방"])

    # --- [TAB 1] 전사 마켓 리스크 브리핑 ---
    with tab1:
        # Top Tier: 핵심 지표 (KPIs)
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("전사 통합 P&L (일간)", "-420억 원", "-120억")
        col_m2.metric("전사 통합 VaR (99%, 1D)", "1,430억 원", "+350억 (한도 근접)")
        col_m3.metric("리스크 한도 소진율", "82.4%", "+5.2%p", delta_color="inverse")
        col_m4.metric("주요 리스크 동인", "비선형 리스크 (Vega)", "모니터링 강화")

        st.markdown("<br>", unsafe_allow_html=True)

        # Middle Tier: 전사 포트폴리오 요약 및 드릴다운
        st.markdown("#### 📊 전사 포트폴리오 마켓 리스크 노출도 (Drill-down)")
        st.caption("※ 전사 레벨의 요약 지표이며, 하단의 익스팬더(Expander)를 클릭하여 개별 종목 단위의 세부 내역을 확인할 수 있습니다.")

        df_summary = pd.DataFrame({
            "포트폴리오 (Depth 1)": ["Bond_Portfolio (채권 10종목)", "ELS_Portfolio (ELS 10종목)"],
            "Delta (델타)": ["-15.2억", "-8.4억"],
            "Gamma (감마)": ["-2.1억", "-14.5억"],
            "Vega (베가)": ["-", "-28.7억"],
            "Rho (로)": ["-35.4억", "-1.2억"],
            "포트폴리오 VaR (1D)": ["450억", "980억"],
            "일간 P&L": ["-140억", "-280억"]
        })

        st.dataframe(df_summary, use_container_width=True, hide_index=True)

        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            with st.expander("📂 채권 포트폴리오 세부 10종목 보기 (Depth 2)"):
                st.dataframe(df_bonds, use_container_width=True, hide_index=True)
        with col_ex2:
            with st.expander("📂 ELS 포트폴리오 세부 10종목 보기 (Depth 2)"):
                st.dataframe(df_els, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Bottom Tier: AI 지식 그래프 & 서술형 리포트
        col_b1, col_b2 = st.columns([1.2, 1.8])

        with col_b1:
            st.markdown("#### 📝 AI 원인 규명 리포트 (GraphRAG 기반)")
            st.warning('''
            **[Warning] 전사 리스크 한도 소진율 82.4% 도달**

            금일 KOSPI/HSCEI 지수 동반 하락 및 시장 내재변동성(IV) 급등에 따라, ELS 포트폴리오에서 **비선형 리스크(Vega 및 Gamma)가 크게 확대**되었습니다.

            해당 포트폴리오의 Vega 포지션 평가손실(-280억)이 전사 일간 P&L 하락(-420억)의 66%를 견인하고 있습니다. 동시에 국채 금리 변동성 확대로 인해 채권 포트폴리오의 금리 민감도(Rho 노출도)가 증가하여 전사적 손실 폭이 확대되었습니다.

            **[AI 권고 조치]** ELS 포트폴리오의 Vega 중립(Neutral)을 위한 옵션 헤지 비중 즉각 확대 및 채권 듀레이션 갭 축소 검토 요망.
            ''')

        with col_b2:
            st.markdown("#### 🕸️ 리스크 파급 인과관계 맵 (Knowledge Graph)")
            st.caption("수학적 팩트 기반 데이터 리니지 (위험 요인 ➡ 포트폴리오 ➡ P&L)")

            net_rca = Network(height='320px', width='100%', bgcolor='#ffffff', font_color='black')

            net_rca.add_node("Macro_Rate", label="금리 변동성 확대", color="#ffb3b3", size=20)
            net_rca.add_node("Macro_Vol", label="시장 내재변동성(IV) 폭등", color="#ffb3b3", size=20)

            net_rca.add_node("Port_Bond", label="Bond_Portfolio\n(Rho 리스크 증가)", color="#cce5ff", size=20)
            net_rca.add_node("Port_ELS", label="ELS_Portfolio\n(Vega 한도 초과)", color="#cce5ff", size=25)

            net_rca.add_node("PnL", label="전사 일간 P&L\n(-420억)", color="#ff9999", size=35)

            net_rca.add_edge("Macro_Rate", "Port_Bond", label="금리 상승 / 평가손", arrows="to", color="#cccccc")
            net_rca.add_edge("Macro_Vol", "Port_ELS", label="옵션 가치 하락 (Vega)", arrows="to", color="#cccccc")
            net_rca.add_edge("Port_Bond", "PnL", label="-140억", arrows="to", color="#ff9999")
            net_rca.add_edge("Port_ELS", "PnL", label="-280억 (주요 원인)", arrows="to", color="#ff9999", value=2)

            net_rca.set_options('{"physics": {"solver": "forceAtlas2Based"}, "edges": {"font": {"size": 11, "color": "#555555"}}}')
            net_rca.write_html("kg_rca.html")
            with open("kg_rca.html", 'r', encoding='utf-8') as f:
                components.html(f.read(), height=330)

    # --- [TAB 2] 부서별 한도 관리 및 AI 처방 ---
    with tab2:
        st.markdown("#### 🏢 데스크별 리스크 한도 모니터링 현황")
        st.caption("채권 및 ELS 운용 데스크에 부여된 VaR, 민감도(Greeks), 손실 한도 소진율 실시간 현황입니다.")

        col_t1, col_t2 = st.columns(2)

        with col_t1:
            st.markdown("##### 📈 채권(Bond) 운용 데스크")
            with st.container(border=True):
                st.markdown("**누적 손실 한도 (Stop-Loss)**: 150억 / 200억")
                st.progress(150/200, text="75.0% (정상)")

                st.markdown("**VaR 한도 (99%, 1D)**: 450억 / 500억")
                st.progress(450/500, text="90.0% (주의)")

                st.markdown("**Rho (금리 민감도) 한도**")
                st.progress(0.95, text="95.0% (경고) - 🚨 한도 근접")

        with col_t2:
            st.markdown("##### 📉 ELS 운용 데스크")
            with st.container(border=True):
                st.markdown("**누적 손실 한도 (Stop-Loss)**: 280억 / 300억")
                st.progress(280/300, text="93.3% (경고) - 🚨 한도 근접")

                st.markdown("**VaR 한도 (99%, 1D)**: 980억 / 1,000억")
                st.progress(980/1000, text="98.0% (위험) - 🚨 한도 초과 임박")

                st.markdown("**Vega (변동성 민감도) 한도**")
                # 1.05 한도 초과 상황을 UI 에러 없이 표현하기 위해 값은 1.0으로 고정
                st.progress(1.0, text="105.0% (초과) - ❌ 한도 위반")

        st.markdown("---")
        st.markdown("#### 🤖 AI 에이전트: 한도 초과 인과관계 및 사내 규정 기반 처방")
        st.info('''
        **[AI 상황 인식] ELS 데스크 Vega 한도 105% 초과 및 채권 데스크 Rho 한도 95% 도달**

        지식 그래프 데이터 리니지 추적 결과, 현재의 한도 소진은 개별 데스크의 포지션 오버가 아닌 **'국채 금리 상승'과 '아시아 증시(KOSPI/HSCEI) 급락에 따른 내재변동성 폭등'이라는 매크로 충격**이 동시에 발생하여 유발되었습니다.

        **[사내 리스크 관리 규정(제14조 3항)에 따른 AI 권고 조치]**
        1. **ELS 운용 데스크:** Vega 한도 위반 해소를 위해, 신규 ELS 발행 및 운용을 즉각 중지하고 장외 시장에서 옵션 양매수(Straddle) 포지션을 구축하여 Vega 노출도를 20% 이상 축소하십시오.
        2. **채권 운용 데스크:** Rho 한도 경고 상태이므로, 잔존만기 3년 이상의 장기채(예: 국고채_3년_A, 회사채_3년_B) 포지션의 듀레이션 헤지를 위해 국채선물 매도(Short) 규모를 비례하여 확대하십시오.
        ''')

# ==========================================
# [페이지 2] 스트레스 테스트 데스크
# ==========================================
elif page == "2. 스트레스 테스트 데스크":
    st.subheader("대내외 위기상황분석 (Stress Test) 및 역위기상황 탐색")
    st.markdown("---")

    tab1, tab2 = st.tabs(["▶ WHAT-IF Simulations", "◀ 역방향: 위기 좌표 탐색 (Reverse Stress Test)"])

    # --- [TAB 1] WHAT-IF Simulations ---
    with tab1:
        col_left, col_right = st.columns([1.2, 1.8])

        with col_left:
            st.markdown("#### a. 시나리오 설정 (자연어 지시)")
            prompt = st.text_area(
                "매크로 위기 시나리오 입력:",
                value="전쟁 확전으로 인해 유가가 급등하고 인플레이션 우려로 금리가 오르며 아시아 증시가 장기 침체될 것 같다. 회사 손익과 VaR에 미치는 영향을 분석해 보자."
            )
            if st.button("AI 시나리오 파라미터 기안 생성", key="btn_fw"):
                st.session_state.scenario_step = 1

            if st.session_state.scenario_step >= 1:
                st.markdown("---")
                # --- [수정] b. AI 추론 및 근거 제시 ---
                st.markdown("#### b. AI 시나리오 추론 및 근거 (RAG & Knowledge Graph)")

                with st.container(border=True):
                    st.markdown("**🔍 외부 정보 수집 (RAG)**")
                    st.caption("- 참조 출처: 블룸버그 매크로 리포트 (최근 7일), 글로벌 투자은행(IB) 원자재 전망 보고서")
                    st.write("유가(WTI)가 배럴당 $90를 돌파함에 따라 글로벌 인플레이션 반등 우려가 심화되고 있습니다. 이에 따라 미 연준(Fed)의 금리 인하 지연 전망이 지배적이며, 신흥국(EM) 자본 유출 리스크가 부각되고 있습니다.")

                    st.markdown("**🕸️ 지식 그래프 인과관계 변환**")
                    st.markdown('''
                    1. **유가 급등** ➡ 글로벌 물가 상승 압력 ➡ **국채 금리 동반 상승 (Target: +100bp)**
                    2. **금리 상승** ➡ 외국인 자금 이탈 및 투심 악화 ➡ **KOSPI 200 하락 (Target: 75pt 수준)**
                    3. **지정학적 리스크** ➡ 亞 증시 동반 침체 ➡ **HSCEI 하락 (Target: 55pt 수준)**
                    ''')

                    st.success("💡 위 추론을 바탕으로 시스템 시뮬레이션을 위한 최적의 파라미터를 도출했습니다.")

                st.markdown("---")

                # --- [수정] c. 파라미터 검토 및 승인 ---
                st.markdown("#### c. 시나리오 파라미터 검토 및 승인")
                st.info("AI가 도출한 리스크 팩터별 최대 충격량과 도달 기간을 검토하고, 필요시 표에서 직접 수정(Edit)한 후 승인하십시오.")

                # 표 형태의 파라미터 에디터 (사용자 검토 및 수정용)
                scenario_df = pd.DataFrame({
                    "리스크 팩터": ["국채 금리 (Base Rate)", "KOSPI 200 지수", "HSCEI 지수"],
                    "현재 수준": ["3.50%", "100 pt", "100 pt"],
                    "최대 충격 (Target)": ["+100 bp", "-25 pt (75 도달)", "-45 pt (55 도달)"],
                    "충격 도달 기간": ["14일", "14일", "14일"]
                })
                st.data_editor(scenario_df, use_container_width=True, hide_index=True)

                if st.button("✅ 기안 승인 및 Full Revaluation 실행", key="btn_fw_run"):
                    st.session_state.scenario_step = 2

        with col_right:
            if st.session_state.scenario_step == 2:
                # --- d. 분석 및 결과 ---
                st.markdown("#### d. 시계열 파급 분석 (Time-Step Full Revaluation)")

                chart_placeholder = st.empty()
                metrics_placeholder = st.empty()
                status_text = st.empty()

                x = np.linspace(40, 100, 40)
                y = np.linspace(40, 100, 40)
                X, Y = np.meshgrid(x, y)
                Z = 100 + (X - 100)*0.3 + (Y - 100)*0.3 - np.where(X < 80, (80 - X) * 1.5, 0) - np.where(Y < 60, (60 - Y) * 2.5, 0) - 35 * np.exp(-0.03 * ((X - 75)**2 + (Y - 55)**2))

                time_labels = ["Day 0 (정상)", "Day 1 (초기 충격)", "Day 3 (투심 악화)", "Day 5 (배리어 접근)", "Day 7 (Gamma 상승)", "Day 10 (헤지 꼬임)", "Day 14 (최대 손실)"]
                traj_x = [100, 96, 92, 88, 83, 78, 75]
                traj_y = [100, 93, 85, 76, 68, 60, 55]
                traj_z = [100, 93, 85, 77, 65, 48, 42]
                rate_shocks = [0, 10, 30, 50, 70, 85, 100]

                for i in range(len(traj_x)):
                    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Blues', opacity=0.7)])
                    fig.add_trace(go.Scatter3d(x=traj_x[:i+1], y=traj_y[:i+1], z=traj_z[:i+1], mode='lines', line=dict(color='orange', width=6), name='과거 궤적'))
                    fig.add_trace(go.Scatter3d(x=[traj_x[i]], y=[traj_y[i]], z=[traj_z[i]], mode='markers', marker=dict(size=10, color='red'), name='현재 지표'))
                    fig.update_layout(title=f"⏳ 진행 상태: {time_labels[i]}", scene=dict(xaxis_title='KOSPI', yaxis_title='HSCEI', zaxis_title='가치', camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.2))), margin=dict(l=0, r=0, b=0, t=30), height=350)
                    chart_placeholder.plotly_chart(fig, use_container_width=True)

                    sim_bonds = revalue_bonds(df_bonds, rate_shock_bps=rate_shocks[i])
                    sim_els = revalue_els(df_els, current_k_index=traj_x[i], current_h_index=traj_y[i])

                    bond_pnl = sim_bonds['pnl'].sum()
                    els_pnl = sim_els['pnl'].sum()
                    total_pnl = bond_pnl + els_pnl

                    with metrics_placeholder.container():
                        st.markdown(f"**실시간 통합 평가 손익 (Full Revaluation):** `<span style='color:red;'>{total_pnl/100000000:,.1f}억 원</span>`", unsafe_allow_html=True)
                        m1, m2 = st.columns(2)
                        m1.metric("채권 포트폴리오 P&L", f"{bond_pnl/100000000:,.1f}억")
                        m2.metric("ELS 포트폴리오 P&L", f"{els_pnl/100000000:,.1f}억")

                    status_text.warning(f"엔진 연산 중... 현재 단계: {time_labels[i]}")
                    time.sleep(0.8)

                status_text.error("🚨 시뮬레이션 종료: 포트폴리오가 최대 손실 구간에 진입했습니다.")

                st.markdown("---")
                # --- e. 대응 방안 ---
                st.markdown("#### e. AI 상황 판단 및 사내 규정 기반 대응 방안")
                st.error('''
                **[심각] 전사 누적 손실 한도의 85% 초과 예상**

                시뮬레이션 14일 차 기준, 금리 상승(+100bp)과 증시 폭락이 결합하여 ELS 및 채권 포트폴리오 양측에서 대규모 손실이 발생했습니다.

                **📌 사내 리스크 관리 규정 제75조 3항 (위기상황 발생 시 포지션 한도 축소)에 의거한 AI 지시 조치:**
                1. **ELS 운용 데스크:** 비선형 손실 웅덩이(Local Minimum) 진입을 방지하기 위해 신규 ELS 롤오버 전면 중지 및 옵션 헤지 비중 30% 즉각 확대 요망.
                2. **채권 운용 데스크:** 듀레이션 리스크 상쇄를 위해 금리 스왑(IRS) 페이 포지션 구축 또는 장기채 포지션 축소 요망.
                ''')

# --- [TAB 2] 역위기상황 탐색 (Reverse Stress Test) ---
    with tab2:
        st.markdown("#### 1. 목표 손실(Target Loss) 기반 역위기상황 탐색 (DML Engine)")
        st.caption("사전에 정의된 타겟 손실을 유발하는 최악의 다차원 리스크 팩터 조합(KOSPI, HSCEI, 국채 금리)을 역산합니다.")

        col_input, col_empty = st.columns([1, 2])
        with col_input:
            target_loss_input = st.number_input("목표 손실액 (단위: 억 원, 예: -400):", min_value=-1000, max_value=-10, value=-400, step=10)
            if st.button("▶ 다차원 역탐색 애니메이션 실행", key="btn_rev_run"):
                st.session_state.rst_step = 1

        if st.session_state.rst_step == 1:
            st.success("✅ DML 엔진 경사하강법(Gradient Descent) 탐색 완료. 타겟 손실에 도달하는 최단 위기 경로를 도출했습니다.")

            col_r1, col_r2 = st.columns(2)

            # --- 시뮬레이션 경로 및 그리드 데이터 사전 세팅 ---
            steps = 10
            k_path = np.linspace(100, 75, steps)
            h_path = np.linspace(100, 55, steps)
            r_path = np.linspace(0, 100, steps) # 금리 상승 충격 (0 -> 100bp)

            radar_placeholder = col_r1.empty()
            contour_placeholder = col_r2.empty()
            status_placeholder = st.empty()

            grid_size = 20
            k_grid = np.linspace(50, 100, grid_size)
            h_grid = np.linspace(50, 100, grid_size)
            K_MESH, H_MESH = np.meshgrid(k_grid, h_grid)
            Z_PNL = np.zeros((grid_size, grid_size))

            for i in range(grid_size):
                for j in range(grid_size):
                    temp_k = K_MESH[i, j]
                    temp_h = H_MESH[i, j]
                    temp_r = (100 - temp_k) * 4.0

                    pnl_bond = revalue_bonds(df_bonds, temp_r)['pnl'].sum()
                    pnl_els = revalue_els(df_els, temp_k, temp_h)['pnl'].sum()
                    Z_PNL[i, j] = (pnl_bond + pnl_els) / 100000000

            # [추가] 시계열 데이터를 수집할 빈 리스트 생성
            history_data = []

            # --- 애니메이션 루프 (시간의 흐름에 따른 엔진 재평가 연동) ---
            for step in range(steps):
                current_k = k_path[step]
                current_h = h_path[step]
                current_r = r_path[step]

                sim_bonds = revalue_bonds(df_bonds, current_r)
                sim_els = revalue_els(df_els, current_k, current_h)

                step_bond_pnl = sim_bonds['pnl'].sum()
                step_els_pnl = sim_els['pnl'].sum()
                current_total_pnl = (step_bond_pnl + step_els_pnl) / 100000000

                # [추가] 현재 스텝의 리스크 팩터 및 개별 상품 P&L(가격 변화) 기록
                row_data = {
                    "탐색 단계": f"Step {step+1}",
                    "KOSPI": f"{current_k:.1f} pt",
                    "HSCEI": f"{current_h:.1f} pt",
                    "국채금리 충격": f"+{current_r:.0f} bp"
                }

                # 채권 10종목 가격 변화(P&L) 매핑
                for _, r in sim_bonds.iterrows():
                    row_data[f"B_{r['name']}"] = r['price_change'] * r['qty']

                # ELS 10종목 가격 변화(P&L) 매핑
                for _, r in sim_els.iterrows():
                    row_data[f"E_{r['name']}"] = r['price_change'] * r['qty']

                history_data.append(row_data)

                # [화면 1] 다차원 리스크 팽창 방사형 차트 (Radar)
                risk_k = max(0, (100 - current_k) / 50 * 100)
                risk_h = max(0, (100 - current_h) / 50 * 100)
                risk_r = max(0, current_r / 150 * 100)

                fig_radar = go.Figure(data=go.Scatterpolar(
                    r=[risk_k, risk_h, risk_r, risk_k],
                    theta=['KOSPI 하락 위험', 'HSCEI 하락 위험', '국채금리 상승 위험', 'KOSPI 하락 위험'],
                    fill='toself',
                    fillcolor='rgba(255, 75, 75, 0.4)',
                    line=dict(color='red', width=2)
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=False,
                    title=f"다차원 리스크 팩터 팽창 (Iteration {step+1})",
                    height=420,
                    margin=dict(l=40, r=40, t=50, b=40)
                )
                radar_placeholder.plotly_chart(fig_radar, use_container_width=True)

                # [화면 2] Top-2 팩터 투영 손실 지형도 (Contour)
                fig_contour = go.Figure(data=go.Contour(
                    z=Z_PNL, x=k_grid, y=h_grid, colorscale='RdBu',
                    contours=dict(showlabels=True, labelfont=dict(color='white'))
                ))
                fig_contour.add_trace(go.Scatter(
                    x=k_path[:step+1], y=h_path[:step+1],
                    mode='lines+markers', line=dict(color='#00FF00', width=4, dash='dot'),
                    marker=dict(size=6, color='black'),
                    name='탐색 궤적'
                ))
                fig_contour.add_trace(go.Scatter(
                    x=[current_k], y=[current_h],
                    mode='markers', marker=dict(size=14, color='yellow', symbol='star'),
                    name='현재 탐색 좌표'
                ))
                fig_contour.update_layout(
                    title=f"핵심 팩터(Top-2) 투영 손실 지형도<br>현재 추정 P&L: <span style='color:red;'>{current_total_pnl:,.0f}억 원</span>",
                    xaxis_title="KOSPI 200",
                    yaxis_title="HSCEI",
                    height=420,
                    margin=dict(l=30, r=30, t=60, b=30)
                )
                contour_placeholder.plotly_chart(fig_contour, use_container_width=True)

                time.sleep(0.5)

            # --- 탐색 완료 후 경영진 보고용 시사점 출력 ---
            status_placeholder.error(f'''
            **🚨 역위기상황 탐색 완료: 타겟 손실 도달 최악의 팩터 조합 산출**

            * **KOSPI 200 지수:** {current_k:.1f}pt 하락 시 ELS 포트폴리오 비선형 리스크 확대 구간 진입
            * **HSCEI 지수:** {current_h:.1f}pt 하락 (가장 가파른 손실 기울기 편미분 값을 형성하는 핵심 동인)
            * **국채 금리:** {current_r:.0f}bp 상승 시 채권 포트폴리오 포지션 한도 도달

            **[경영진 보고용 시사점]** 현재 전사 포트폴리오는 단일 팩터의 충격보다 **'아시아 증시 동반 하락'과 '금리 급등'이 결합된 복합 위기 상황**에서 테일 리스크(Tail Risk) 전이 속도가 기하급수적으로 빨라집니다. 스트레스 테스트 시나리오 설정 시 팩터 간의 교차 민감도(Cross-Greeks)를 최우선으로 고려해야 합니다.
            ''')

            # --- [신규 추가] 시계열 상세 데이터 테이블(Audit Trail) 표출 ---
            st.markdown("---")
            st.markdown("#### 3. 역위기상황 탐색 시계열 상세 데이터 (Audit Trail)")
            st.caption("각 탐색 단계별 리스크 팩터의 변화와 구성 상품 20종의 누적 가격 변화(평가손익, 단위: 원) 추적 내역입니다. 가로로 스크롤하여 전체 포트폴리오를 확인할 수 있습니다.")

            df_history = pd.DataFrame(history_data)

            # 숫자 데이터를 보기 좋게 콤마가 포함된 문자열로 포맷팅
            for col in df_history.columns:
                if col not in ["탐색 단계", "KOSPI", "HSCEI", "국채금리 충격"]:
                    df_history[col] = df_history[col].apply(lambda x: f"{x:,.0f}")

            # 가로 스크롤이 자연스럽게 적용되도록 렌더링
            st.dataframe(df_history, use_container_width=True, hide_index=True)

# ==========================================
# [페이지 3] 시스템 오퍼레이션 (자연어 배치 구동)
# ==========================================
elif page == "3. 시스템 오퍼레이션":
    st.subheader("🤖 AI 기반 복합 배치(Batch) 파이프라인 기안 에이전트")
    st.markdown("단순 반복적인 매뉴얼 세팅과 휴먼 에러를 제거하고, **인력을 고부가가치 분석 업무로 재배치**하기 위한 지능형 오퍼레이션 시연입니다.")
    st.markdown("---")

    col1, col2 = st.columns([1.2, 1.8])

    # [좌측] 기존 방식 (AS-IS)
    with col1:
        st.header("📉 AS-IS: 레거시 매뉴얼 설정")
        st.caption("작업자가 모든 종속성과 파라미터를 수동으로 확인하고 세팅해야 함")

        with st.container(border=True):
            st.selectbox("1. 대상 포트폴리오 선택", ["선택하세요", "포트폴리오 A", "포트폴리오 B", "포트폴리오 C", "포트폴리오 D"])
            st.date_input("2. 시작일 설정")
            st.date_input("3. 종료일 설정")
            st.multiselect("4. 실행 엔진 선택", ["Market Data Fetch", "Pricing Engine", "VaR Engine", "Stress Test Engine", "Report Gen"])
            st.checkbox("5. 에러 발생 시 롤백 활성화")
            st.checkbox("6. 선행 작업(Data Fetch) 완료 대기")
            st.button("수동 배치 실행 (위험: 의존성 체크 누락 가능성)")

            st.error("⚠️ **문제점:** 월말 재배치 시 작업자가 10여 개의 화면을 오가며 세팅. 순서 실수 시 8시간짜리 야간 배치 실패 및 익일 보고 누락 발생.")

    # [우측] AI 에이전트 방식 (TO-BE)
    with col2:
        st.header("✨ TO-BE: AI 에이전트 오케스트레이션")
        st.caption("자연어 의도 파악 -> 의존성 분석 -> 파이프라인 자동 기안 -> 사용자 승인")

        user_input = st.text_area(
            "💬 오퍼레이션 지시 (자연어 입력):",
            value="26.01.15부터 26.02.14까지 포트폴리오 D에 대한 재배치를 돌리고, 각 날짜별 보고서 생성 작업도 다시 돌려줘."
        )

        if st.button("AI 파이프라인 기안 (Draft) 생성 🚀"):
            with st.spinner("자연어 분석 및 종속성(Dependency) 파악 중..."):
                time.sleep(1.2)
                st.session_state.batch_step = 1

        if st.session_state.batch_step == 1:
            st.success("✅ 시스템 파라미터 매핑 및 작업 종속성 검증 완료")

            with st.expander("🔍 AI 파라미터 추출 결과 (시스템 매핑)", expanded=True):
                st.code('''
                {
                  "intent": "RE-RUN_BATCH_AND_REPORT",
                  "target_portfolio": "PORT_D",
                  "date_range": {"start": "2026-01-15", "end": "2026-02-14"},
                  "required_engines": ["Pricing", "VaR", "Report_Gen"],
                  "days_count": 31
                }
                ''', language='json')

            st.markdown("### 📋 AI 기안: 자동 생성된 작업 파이프라인(DAG)")
            st.info('''
            💡 **AI 시스템 알림:** 요청하신 '보고서 생성'을 위해서는 필수 선행 작업이 필요합니다.
            AI가 시스템 종속성을 파악하여 **누락된 선행 엔진 구동을 자동으로 파이프라인에 추가**했습니다.
            ''')

            st.markdown("""
            1. 🔄 **[Data Fetch]** 26.01.15 ~ 26.02.14 구간 'PORT_D' 기초자산 및 시장 데이터 검증 (자동 추가됨)
            2. 🧮 **[Pricing Engine]** 31일치 개별 상품 민감도(Greeks) 재산출
            3. 📈 **[VaR Engine]** 31일치 포트폴리오 통합 VaR 및 P&L 재산출
            4. 📊 **[Report Gen]** 31개 일일 리스크 브리핑 보고서 PDF 생성 및 DB 아카이빙
            """)

            st.warning("⏱️ **예상 소요 시간:** 약 45분 / 🖥️ **시스템 부하 예상:** 중간(Medium)")

            st.markdown("---")
            st.markdown("#### 👨‍⚖️ 최종 승인 (Human-in-the-loop)")
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                if st.button("✅ 기안 승인 및 배치 실행"):
                    st.toast("배치 작업이 안전하게 스케줄링 되었습니다.")
            with col2_2:
                if st.button("❌ 기안 반려 (재입력)"):
                    st.session_state.batch_step = 0
                    st.rerun()

# ==========================================
# [페이지 4] 지능형 장애 대응 및 시스템 복구 에이전트")
# ==========================================
elif page == "4. 장애 대응 가이드 에이전트":
    st.title("🛠️ AI 장애 대응 및 유지보수 에이전트")
    st.markdown("지식 그래프에 축적된 과거 장애 조치 이력과 시스템 로그를 분석하여, **1차 해결 가이드라인을 제공**하고 필요시 유지보수팀(L2)에 **컨텍스트 기반 호출**을 수행합니다.")
    st.markdown("---")

    col1, col2 = st.columns([1, 1.5])

    # [좌측] 에러 로그 모니터링 및 입력 창
    with col1:
        st.subheader("🚨 실시간 시스템 알림 (Alert)")

        # 가상의 에러 발생 상황 제시
        st.error("**[CRITICAL] 배치 작업 실패 (Pricing Engine)**\n\n발생 시간: 03:15 AM\n대상: Portfolio_ELS_Book")

        error_log_input = st.text_area(
            "상세 에러 로그 확인 및 분석 요청:",
            value="""[ERROR] MemoryError: Out of memory calculating Pricing Surface for 10,000 ELS instruments.
[TRACE] File "pricing_engine.py", line 428, in revalue_els_portfolio
[TRACE] File "numpy/core/numeric.py", line 330, in full
[INFO] Container memory limit (32GB) exceeded.""",
            height=150
        )

        if st.button("🔍 AI 원인 분석 및 해결 가이드 요청", type="primary"):
            st.session_state.incident_step = 1

    # [우측] AI 분석 및 가이드 제공 영역
    with col2:
        if st.session_state.incident_step >= 1:
            st.subheader("🤖 AI 에이전트 분석 결과")

            with st.status("지식 그래프 탐색 중...", expanded=True) as status:
                st.write("1. 로그 텍스트에서 Entity 추출 중... (Pricing Engine, ELS, Out of memory)")
                time.sleep(0.5)
                st.write("2. 사내 조치 이력(Jira/Wiki) Ontology 매핑 중...")
                time.sleep(0.5)
                st.write("3. 유사 장애 패턴 (Similarity > 92%) 검색 완료.")
                time.sleep(0.5)
                status.update(label="분석 완료: 유사 장애 이력 3건 발견", state="complete", expanded=False)

            # 지식 그래프 추론 결과 매핑
            with st.expander("🔗 지식 그래프 추론 경로 (Traceability)", expanded=False):
                st.code("""
(ErrorLog: OOM) -[OCCURRED_IN]-> (Component: Pricing Engine)
(Component: Pricing Engine) -[HAS_HISTORY]-> (Ticket: INC-2025-08)
(Ticket: INC-2025-08) -[RESOLVED_BY]-> (Action: Increase Container Memory Limit)
(Ticket: INC-2025-08) -[RESOLVED_BY]-> (Action: Enable Chunk Processing)
                """, language="text")

            # 1차 조치 가이드라인
            st.info("💡 **1차 대응 가이드라인 (추천도: 높음)**\n\n과거 `INC-2025-08` (25년 8월 ELS 대규모 재평가 장애) 사례와 동일한 패턴입니다. 다음 조치를 순서대로 수행해 보십시오.")

            st.markdown("""
            **조치 1.** `config/pricing_env.yaml` 파일에서 컨테이너 메모리 제한을 임시 상향 (32GB $\\rightarrow$ 64GB)

            **조치 2.** 배치 파라미터 중 `CHUNK_SIZE`를 10,000에서 2,000으로 분할 설정

            **조치 3.** [시스템 오퍼레이션] 메뉴에서 부분 재배치(Partial Re-run) 실행
            """)

            st.markdown("---")
            st.markdown("#### 🛠️ 조치 결과 입력")

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("✅ 1차 조치로 해결됨 (DB 이력 저장)"):
                    st.success("해결 이력이 지식 그래프에 업데이트되었습니다. 시스템이 정상화되었습니다.")
                    st.session_state.incident_step = 0 # 초기화

            with col_b:
                if st.button("🚨 1차 대응 실패 (유지보수팀 호출)"):
                    st.session_state.incident_step = 2

        # 1차 대응 실패 시: 유지보수팀 에스컬레이션 화면
        if st.session_state.incident_step == 2:
            st.warning("⚠️ 1차 가이드라인으로 해결되지 않았습니다. 유지보수팀(L2/L3)으로 에스컬레이션합니다.")

            st.markdown("**✉️ 자동 생성된 호출 리포트 (Draft)**")
            with st.container(border=True):
                st.markdown("""
                **To:** 리스크 솔루션 개발팀 (dev_risk@company.com)
                **CC:** 시스템 운영 파트장
                **Subject:** [URGENT] Pricing Engine OOM 장애 1차 대응 실패 - 지원 요청

                **1. 장애 개요**
                * **발생 일시:** 2026-03-23 03:15 AM
                * **증상:** ELS Portfolio 평가 중 MemoryError 발생 (Container Limit 32GB Exceeded)

                **2. 1차 대응 내역 (운영팀 조치사항)**
                * 가이드에 따라 메모리 Limit 64GB 상향 및 Chunk Size 2000 분할을 적용 후 재시도 하였으나,
                * **추가 에러:** `CUDA Out of Memory` 로 전이되며 다시 실패함.

                **3. 요청 사항**
                * 로직 내 메모리 누수(Leak) 또는 GPU 메모리 해제 로직 점검 요망.
                * 즉각적인 L3 엔지니어 투입 및 디버깅을 요청합니다.
                """)

            if st.button("🚀 리포트 전송 및 엔지니어 Paging (Slack/Email 발송)"):
                st.toast("유지보수팀 호출이 완료되었습니다. Ticket #INC-2026-03 이 생성되었습니다.")
                time.sleep(1)
                st.session_state.incident_step = 0