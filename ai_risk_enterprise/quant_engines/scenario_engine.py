import numpy as np
import pandas as pd
from .pricing import revalue_bonds_multi, revalue_els_multi

class ScenarioEngine:
    def __init__(self, base_mkt_state):
        self.base_mkt = base_mkt_state

    # ----------------------------------------------------
    # 1. 시계열 파급 분석 경로 생성
    # ----------------------------------------------------
    def generate_simulation_path(self, df_bonds, df_els, target_params, steps=7):
        k_target = target_params.get('kospi', 80.0)
        s_target = target_params.get('samsung', 75.0)
        r_target_bp = target_params.get('rate', 50.0) # 금리는 bp단위 타겟
        
        traj_k = np.linspace(100, k_target, steps)
        traj_s = np.linspace(100, s_target, steps)
        traj_r = np.linspace(0, r_target_bp, steps)
        
        history = []
        for i in range(steps):
            curr_mkt = self.base_mkt.copy()
            # [핵심] base_mkt의 원본 주가에 퍼센트를 곱해서 적용
            curr_mkt['KOSPI200_Close'] *= (traj_k[i] / 100.0)
            curr_mkt['Samsung_Close'] *= (traj_s[i] / 100.0)
            
            # 금리는 %p(퍼센트포인트) 단위로 더해줌 (1bp = 0.01%p)
            for tenor in ['KTB_6M', 'KTB_1Y', 'KTB_3Y', 'KTB_5Y', 'Corp_6M', 'Corp_1Y']:
                if tenor in curr_mkt:
                    curr_mkt[tenor] += (traj_r[i] / 100.0) 

            sim_b = revalue_bonds_multi(df_bonds, curr_mkt, self.base_mkt)
            sim_e = revalue_els_multi(df_els, curr_mkt, self.base_mkt)
            total_pnl = sim_b['pnl'].sum() + sim_e['pnl'].sum()
            
            history.append({
                "step": f"Day {i}", "kospi": traj_k[i], "samsung": traj_s[i],
                "rate_shock": traj_r[i], "total_pnl": total_pnl,
                "bond_pnl": sim_b['pnl'].sum(), "els_pnl": sim_e['pnl'].sum()
            })
        return history

    # ----------------------------------------------------
    # 2. 역방향 위기 탐색 (Gradient Vanishing 해결)
    # ----------------------------------------------------
    def find_worst_case_path(self, df_bonds, df_els, target_loss_bn, max_iter=30):
        target_pnl_raw = target_loss_bn * 100000000
        ck, cs, cr = 100.0, 100.0, 0.0 # 주가 100%, 금리 0bp에서 시작
        path = []

        for _ in range(max_iter):
            curr_pnl = self._eval_total_pnl(df_bonds, df_els, ck, cs, cr)
            path.append({"kospi": ck, "samsung": cs, "rate": cr, "pnl": curr_pnl})
            
            if curr_pnl <= target_pnl_raw: break
            
            # 수치적 미분 (Bump and Revalue)
            pnl_k = self._eval_total_pnl(df_bonds, df_els, ck - 1.0, cs, cr) # KOSPI 1% 하락
            pnl_s = self._eval_total_pnl(df_bonds, df_els, ck, cs - 1.0, cr) # 삼성 1% 하락
            pnl_r = self._eval_total_pnl(df_bonds, df_els, ck, cs, cr + 10.0) # 금리 10bp 상승
            
            # 방향성(Gradient) 추출 (손실이 얼마나 더 커졌나)
            grad_k = max(0, curr_pnl - pnl_k)
            grad_s = max(0, curr_pnl - pnl_s)
            grad_r = max(0, curr_pnl - pnl_r)
            
            total_grad = grad_k + grad_s + grad_r + 1e-9 # 0 나누기 방지
            
            # [핵심] 주가와 금리의 스텝(학습률)을 분리! (금리 편식 방지)
            # KOSPI/삼성은 한 번에 최대 3% 하락, 금리는 한 번에 최대 20bp 상승
            step_k = (grad_k / total_grad) * 3.0
            step_s = (grad_s / total_grad) * 3.0
            step_r = (grad_r / total_grad) * 20.0
            
            # 최소 이동폭 보장 (알고리즘 멈춤 방지)
            ck -= max(step_k, 0.5)
            cs -= max(step_s, 0.5)
            cr += max(step_r, 2.0)
            
            ck = max(10.0, ck) # 하한선 방어
            cs = max(10.0, cs)

        return path

    def _eval_total_pnl(self, df_b, df_e, k, s, r):
        mkt = self.base_mkt.copy()
        mkt['KOSPI200_Close'] *= (k / 100.0)
        mkt['Samsung_Close'] *= (s / 100.0)
        for t in ['KTB_6M', 'KTB_1Y', 'KTB_3Y', 'KTB_5Y', 'Corp_6M', 'Corp_1Y']:
            if t in mkt: mkt[t] += (r / 100.0) # r은 bp이므로 100으로 나누어 %p 적용
            
        b_res = revalue_bonds_multi(df_b, mkt, self.base_mkt)
        e_res = revalue_els_multi(df_e, mkt, self.base_mkt)
        return b_res['pnl'].sum() + e_res['pnl'].sum()
