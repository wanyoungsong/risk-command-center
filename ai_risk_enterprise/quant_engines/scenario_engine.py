import numpy as np
import pandas as pd
from .pricing import revalue_bonds_multi, revalue_els_multi

class ScenarioEngine:
    def __init__(self, base_mkt_state):
        self.base_mkt = base_mkt_state

    # ----------------------------------------------------
    # 1. 시계열 파급 분석 경로 생성 (Time-Step Simulation)
    # ----------------------------------------------------
    def generate_simulation_path(self, df_bonds, df_els, target_params, steps=7):
        """AI가 설정한 타겟까지의 충격 경로와 단계별 P&L을 계산"""
        
        # 타겟 파라미터 파싱 (AI가 준 JSON에서 값 추출)
        k_target = target_params.get('kospi', 100.0)
        s_target = target_params.get('samsung', 100.0)
        r_target = target_params.get('rate', 0.0)
        
        # 경로 생성 (100% -> 타겟%)
        traj_k = np.linspace(100, k_target, steps)
        traj_s = np.linspace(100, s_target, steps)
        traj_r = np.linspace(0, r_target, steps)
        
        history = []
        for i in range(steps):
            curr_mkt = self.base_mkt.copy()
            # 시장 변수 업데이트
            curr_mkt['KOSPI200_Close'] *= (traj_k[i] / 100.0)
            curr_mkt['Samsung_Close'] *= (traj_s[i] / 100.0)
            for tenor in ['KTB_6M', 'KTB_1Y', 'KTB_3Y', 'KTB_5Y', 'Corp_6M', 'Corp_1Y']:
                curr_mkt[tenor] += (traj_r[i] / 100.0)
            
            # 변동성 전이 (간단한 프록시 모델)
            for key in curr_mkt.keys():
                if key.startswith('Vol_'):
                    curr_mkt[key] += (100 - traj_k[i]) * 0.005

            # 재평가 실행
            sim_b = revalue_bonds_multi(df_bonds, curr_mkt, self.base_mkt)
            sim_e = revalue_els_multi(df_els, curr_mkt, self.base_mkt)
            total_pnl = sim_b['pnl'].sum() + sim_e['pnl'].sum()
            
            history.append({
                "step": f"Day {i}",
                "kospi": traj_k[i],
                "samsung": traj_s[i],
                "rate_shock": traj_r[i],
                "total_pnl": total_pnl,
                "bond_pnl": sim_b['pnl'].sum(),
                "els_pnl": sim_e['pnl'].sum()
            })
            
        return history

    # ----------------------------------------------------
    # 2. 역방향 위기 탐색 (Reverse Stress Test - Gradient Descent)
    # ----------------------------------------------------
    def find_worst_case_path(self, df_bonds, df_els, target_loss_bn, max_iter=50):
        """목표 손실(단위: 억)에 도달하는 최단 위기 경로를 탐색"""
        target_pnl_raw = target_loss_bn * 100000000
        ck, cs, cr = 100.0, 100.0, 0.0 # 시작점
        path = []

        for _ in range(max_iter):
            # 현재 상태 P&L 계산
            curr_pnl = self._eval_total_pnl(df_bonds, df_els, ck, cs, cr)
            path.append({"kospi": ck, "samsung": cs, "rate": cr, "pnl": curr_pnl})
            
            if curr_pnl <= target_pnl_raw:
                break
            
            # 수치적 미분 (Gradient) 계산
            eps = 1.0
            pnl_k = self._eval_total_pnl(df_bonds, df_els, ck - eps, cs, cr)
            pnl_s = self._eval_total_pnl(df_bonds, df_els, ck, cs - eps, cr)
            pnl_r = self._eval_total_pnl(df_bonds, df_els, ck, cs, cr + eps)
            
            grad_k = max(0, curr_pnl - pnl_k) + 1.0
            grad_s = max(0, curr_pnl - pnl_s) + 1.5
            grad_r = max(0, curr_pnl - pnl_r) + 2.0
            
            total_grad = grad_k + grad_s + grad_r
            ck = max(10.0, ck - (grad_k / total_grad) * 2.0)
            cs = max(10.0, cs - (grad_s / total_grad) * 2.0)
            cr += (grad_r / total_grad) * 5.0

        return path

    def _eval_total_pnl(self, df_b, df_e, k, s, r):
        mkt = self.base_mkt.copy()
        mkt['KOSPI200_Close'] *= (k / 100.0)
        mkt['Samsung_Close'] *= (s / 100.0)
        for t in ['KTB_6M', 'KTB_1Y', 'KTB_3Y', 'KTB_5Y', 'Corp_6M', 'Corp_1Y']:
            mkt[t] += (r / 100.0)
        
        # 간이 변동성 전이
        vol_shock = (100 - min(k, s)) * 0.005
        for key in mkt.keys():
            if key.startswith('Vol_'): mkt[key] += vol_shock
            
        b_res = revalue_bonds_multi(df_b, mkt, self.base_mkt)
        e_res = revalue_els_multi(df_e, mkt, self.base_mkt)
        return b_res['pnl'].sum() + e_res['pnl'].sum()