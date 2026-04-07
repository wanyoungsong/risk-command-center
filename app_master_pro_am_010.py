import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import google.generativeai as genai
import json
import re
import time

# ==========================================
# 0. API 세팅
# ==========================================
try:
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", None)
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
except Exception:
    pass

# ==========================================
# 1. 자산운용사 포트폴리오 & 엔진 (유저 로직 적용)
# ==========================================
fund_data = {
    'Fund_Name': ['KOSPI 200 인덱스 펀드', 'TIGER 나스닥100 (UH)', 'TIGER 나스닥100 (H)', 'TIGER 미국테크 커버드콜', '글로벌 상업용 부동산 펀드'],
    'Current_AUM': [50000, 80000, 60000, 40000, 70000],
    'Fee_Rate(%)': [0.15, 0.50, 0.50, 0.70, 1.50],
    'Hedge_Type':  ['None', 'UH', 'H', 'UH', 'H'],
    'Delta':       [1.0, 1.0, 1.0, 0.6, 0.0],
    'Gamma':       [0.0, 0.0, 0.0, -1.2, 0.0],
    'Rate_Beta':   [0.0, 0.0, 0.0, 0.0, -6.0]
}
df_base = pd.DataFrame(fund_data).set_index('Fund_Name')
fixed_costs = 800  # 분기 고정비 (억원)

def run_am_stress_test(stock_shock, fx_shock, rate_shock):
    """자산운용사 충격 엔진 (수익률, 펀드런, 영업이익 산출)"""
    df = df_base.copy()
    
    def calc_impact(row):
        base_return = (row['Delta'] * stock_shock) + (0.5 * row['Gamma'] * (stock_shock ** 2)) + (row['Rate_Beta'] * rate_shock)
        final_return = base_return
        
        if row['Hedge_Type'] == 'UH':
            final_return = (1 + base_return) * (1 + fx_shock) - 1
        elif row['Hedge_Type'] == 'H':
            if fx_shock > 0.05:
                # [수정] 단순 수식을 넘어선 실제 '유동성 발작(Liquidity Crunch)' 프리미엄 5배 적용!
                # 환율이 튈 때 환헤지 롤오버 비용과 강제 청산 리스크를 비선형적으로 증폭시킴
                final_return -= ((fx_shock - 0.05) ** 1.5) * 5.0 
        return final_return

    df['Stress_Return'] = df.apply(calc_impact, axis=1)
    df['Run_Rate'] = df['Stress_Return'].apply(lambda r: abs(r + 0.10) * 1.5 if r < -0.10 else 0.0)
    
    df['AUM_MTM'] = df['Current_AUM'] * (1 + df['Stress_Return'])
    df['Outflow'] = df['AUM_MTM'] * df['Run_Rate']
    df['Final_AUM'] = df['AUM_MTM'] - df['Outflow']
    df['New_Rev'] = df['Final_AUM'] * (df['Fee_Rate(%)'] / 100)
    
    tot_curr_aum = df['Current_AUM'].sum()
    tot_final_aum = df['Final_AUM'].sum()
    tot_curr_rev = (df['Current_AUM'] * (df['Fee_Rate(%)'] / 100)).sum()
    
    curr_op = tot_curr_rev - fixed_costs
    new_op = df['New_Rev'].sum() - fixed_costs
    
    return df, tot_curr_aum, tot_final_aum, curr_op, new_op

# ==========================================
# 2. AI 프롬프트 에이전트
# ==========================================
def extract_am_scenario(prompt_text):
    model = genai.GenerativeModel('gemini-2.5-flash')
    prompt = f"""자산운용사 리스크 AI 참모로서, [입력]을 분석해 JSON 파라미터를 추출해.
    입력: {prompt_text}
    조건: 주가(stock), 환율(fx), 금리(rate) 충격량을 소수로 변환해(예: -25% 하락 -> -0.25, 15% 상승 -> 0.15, 2%p 상승 -> 0.02). 언급이 없으면 0.0으로 둬.
    JSON 형식: {{"is_relevant": true, "summary": "...", "stock": -0.25, "fx": 0.15, "rate": 0.02}}"""
    try:
        res = model.generate_content(prompt, generation_config=genai.GenerationConfig(response_mime_type="application/json"))
        return json.loads(res.text)
    except:
        return {"is_relevant": True, "stock": -0.25, "fx": 0.15, "rate": 0.02, "summary": "API 오류: 퍼펙트 스톰(주가 -25%, 환율 +15%, 금리 +2%p) 적용"}

def stream_am_briefing(tot_drop, curr_op, new_op, df):
    model = genai.GenerativeModel('gemini-2.5-flash')
    outflow_tot = df['Outflow'].sum()
    prompt = f"""운용사 경영진 대상 AI 브리핑. AUM 손실액: {tot_drop:,.0f}억, 펀드런(유출): {outflow_tot:,.0f}억. 
    영업이익 변화: {curr_op:,.0f}억 -> {new_op:,.0f}억.
    환헤지 마진콜, 커버드콜 비선형 손실, 유동성(Fund Run) 리스크를 강조하며 3문단 이내로 권고안을 작성해."""
    for chunk in model.generate_content(prompt, stream=True):
        if chunk.text: yield chunk.text

# ==========================================
# 3. Streamlit UI (Agent Layout)
# ==========================================
st.set_page_config(layout="wide", page_title="AM Risk Agent")

modes = ["▶️ 1. 위기 시나리오 분석 (What-If)", "◀️ 2. 역방향 위기 탐색 (RST)"]
for m in modes:
    if f"msgs_{m}" not in st.session_state: st.session_state[f"msgs_{m}"] = []
    if f"step_{m}" not in st.session_state: st.session_state[f"step_{m}"] = 0

with st.sidebar:
    st.title("🏦 Asset Mgmt Agent")
    st.caption("수백조 규모 자산운용사 맞춤형 AI 리스크")
    selected_mode = st.radio("메뉴 선택", modes)

col_chat, col_viz = st.columns([1.3, 1.7], gap="large")
curr_msgs = st.session_state[f"msgs_{selected_mode}"]
curr_step_key = f"step_{selected_mode}"

with col_chat:
    st.subheader(f"{selected_mode.split('. ')[1]}")
    if not curr_msgs:
        intro = "운용사 AUM 방어 AI 참모입니다. 퍼펙트 스톰 시나리오를 지시하세요. (예: '주가 25% 폭락, 환율 15% 상승, 금리 2%p 인상 시나리오 돌려봐')" if "1" in selected_mode else "영업이익 적자 전환점(-800억 이하)을 유발하는 최악의 주가/환율 궤적을 역산합니다. (예: '영업손실 -100억 찍히는 경로 찾아줘')"
        curr_msgs.append({"role": "assistant", "content": intro})

    for msg in curr_msgs:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("AI에게 지시하세요..."):
        curr_msgs.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            if "1" in selected_mode:
                with st.spinner("시나리오 추출 중..."):
                    params = extract_am_scenario(prompt)
                st.session_state.am_params = params
                st.session_state[curr_step_key] = 1
                msg = f"🔍 파라미터 추출: 주가 {params['stock']*100}%, 환율 {params['fx']*100}%, 금리 {params['rate']*100}%p. 우측에서 결과를 확인하세요."
                st.write(msg); curr_msgs.append({"role": "assistant", "content": msg})
                
                # 엔진 연산 및 AI 브리핑 즉시 실행
                df_res, t_curr, t_final, c_op, n_op = run_am_stress_test(params['stock'], params['fx'], params['rate'])
                st.session_state.am_results = (df_res, t_curr, t_final, c_op, n_op)
                
                st.markdown("🚨 **AI 경영진 브리핑**")
                res = st.write_stream(stream_am_briefing(t_curr - t_final, c_op, n_op, df_res))
                curr_msgs.append({"role": "assistant", "content": f"🚨 [AI 브리핑]\n\n{res}"})
                st.rerun()

            elif "2" in selected_mode:
                nums = re.findall(r'-?\d+', prompt)
                target_op = float(nums[0]) if nums else -100.0
                st.session_state.target_op = target_op
                st.session_state[curr_step_key] = 1
                msg = f"🔍 목표 영업이익 **{target_op}억 원**에 도달하는 최단 경로를 엔진으로 탐색합니다."
                st.write(msg); curr_msgs.append({"role": "assistant", "content": msg})
                st.rerun()

with col_viz:
    if "1" in selected_mode and st.session_state[curr_step_key] >= 1:
        df_res, t_curr, t_final, c_op, n_op = st.session_state.am_results
        
        st.subheader("🏢 전사 AUM 및 영업이익 임팩트")
        m1, m2, m3 = st.columns(3)
        m1.metric("총 수탁고 (AUM)", f"{t_final:,.0f}억", f"{t_final - t_curr:,.0f}억 (펀드런 포함)")
        m2.metric("대규모 환매 (Fund Run)", f"-{df_res['Outflow'].sum():,.0f}억", "유동성 위기")
        m3.metric("예상 영업이익 (OP)", f"{n_op:,.0f}억", f"{n_op - c_op:,.0f}억 (고정비 {fixed_costs}억)", delta_color="inverse")
        
        # AUM 변화 폭포수(Waterfall) 차트
        fig = go.Figure(go.Waterfall(
            orientation="v", measure=["absolute", "relative", "relative", "total"],
            x=["현재 AUM", "시장가치 하락", "고객 대규모 환매", "최종 AUM"],
            y=[t_curr, -(t_curr - df_res['AUM_MTM'].sum()), -df_res['Outflow'].sum(), t_final],
            textposition="outside", text=[f"{t_curr:,.0f}", f"-{t_curr - df_res['AUM_MTM'].sum():,.0f}", f"-{df_res['Outflow'].sum():,.0f}", f"{t_final:,.0f}"]
        ))
        fig.update_layout(title="시나리오별 AUM 증감 폭포수 차트", height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### 📊 펀드별 스트레스 테스트 상세 내역")
        df_disp = df_res[['Hedge_Type', 'Stress_Return', 'Run_Rate', 'Outflow', 'Final_AUM']].copy()
        df_disp['Stress_Return'] = (df_disp['Stress_Return'] * 100).map("{:.1f}%".format)
        df_disp['Run_Rate'] = (df_disp['Run_Rate'] * 100).map("{:.1f}%".format)
        df_disp['Outflow'] = df_disp['Outflow'].map("{:,.0f}억".format)
        df_disp['Final_AUM'] = df_disp['Final_AUM'].map("{:,.0f}억".format)
        st.dataframe(df_disp, use_container_width=True)

    elif "2" in selected_mode and st.session_state[curr_step_key] >= 1:
        st.subheader("◀ 역방향: 영업이익 목표 역탐색 (RST)")
        target_op = st.session_state.target_op
        
        # 그리드 서치 엔진 (단순화된 RST)
        with st.spinner("비선형 유동성 리스크를 반영한 최악의 조합 탐색 중..."):
            # [수정] 엔진이 적자를 낼 수 있도록 탐색 한계치를 주가 -60%, 환율 +50%까지 확장
            stock_grid = np.linspace(0, -0.60, 30) 
            fx_grid = np.linspace(0, 0.50, 30)
            S_MESH, F_MESH = np.meshgrid(stock_grid, fx_grid)
            OP_MESH = np.zeros((30, 30))
            
            for i in range(30):
                for j in range(30):
                    _, _, _, _, op = run_am_stress_test(S_MESH[i, j], F_MESH[i, j], 0.0) 
                    OP_MESH[i, j] = op
            
            # 목표치에 가장 가까운 좌표 찾기
            idx = np.unravel_index(np.argmin(np.abs(OP_MESH - target_op)), OP_MESH.shape)
            worst_s, worst_f, worst_op = S_MESH[idx], F_MESH[idx], OP_MESH[idx]
            time.sleep(1) # 애니메이션 효과용
            
        fig_contour = go.Figure(data=go.Contour(z=OP_MESH, x=stock_grid*100, y=fx_grid*100, colorscale='RdBu', colorbar=dict(title='영업이익')))
        fig_contour.add_trace(go.Scatter(x=[worst_s*100], y=[worst_f*100], mode='markers+text', marker=dict(size=14, color='yellow', symbol='star'), text=[f"목표 타격점<br>({worst_op:,.0f}억)"], textposition="top right"))
        fig_contour.update_layout(title="영업이익 손실 지형도 (주가 vs 환율)", xaxis_title="주가 하락률 (%)", yaxis_title="환율 상승률 (%)", height=400)
        
        st.plotly_chart(fig_contour, use_container_width=True)
        st.error(f"🚨 **엔진 탐색 결과:** 주가 **{worst_s*100:.1f}% 폭락** 및 환율 **{worst_f*100:+.1f}% 급등** 시 영업이익 **{worst_op:,.0f}억 원** 도달")
