import pandas as pd
import numpy as np
from scipy.stats import norm

def revalue_bonds_multi(df, current_mkt, base_mkt):
    results = df.copy()
    results['base_rate'] = results['curve'].map(lambda x: base_mkt[x]) / 100.0
    results['current_rate'] = results['curve'].map(lambda x: current_mkt[x]) / 100.0
    results['old_price'] = results['face_value'] / ((1 + results['base_rate']) ** results['tenor'])
    results['new_price'] = results['face_value'] / ((1 + results['current_rate']) ** results['tenor'])
    results['price_change'] = results['new_price'] - results['old_price']
    results['pnl'] = results['price_change'] * results['qty']
    return results

def revalue_els_multi(df, current_mkt, base_mkt):
    results = df.copy()
    r1 = (results['asset1'].map(lambda x: current_mkt[x]) / results['asset1'].map(lambda x: base_mkt[x])) * 100
    r2 = (results['asset2'].map(lambda x: current_mkt[x]) / results['asset2'].map(lambda x: base_mkt[x])) * 100
    worst_perf = np.minimum(r1, r2)
    distance_to_ki = worst_perf - results['ki_barrier']

    gamma_bleed = np.where(distance_to_ki > 0, 2000 * np.exp(-0.15 * distance_to_ki), 3000 + (results['ki_barrier'] - worst_perf) * 100)
    
    vol_asset1 = results['asset1'].apply(lambda x: x.replace('_Close', ''))
    vol_asset2 = results['asset2'].apply(lambda x: x.replace('_Close', ''))
    avg_vol = (vol_asset1.map(lambda x: current_mkt[f'Vol_{x}']) + vol_asset2.map(lambda x: current_mkt[f'Vol_{x}'])) / 2.0
    vega_loss = np.maximum(0, avg_vol - 0.20) * 50000

    avg_intensity = (vol_asset1.map(lambda x: current_mkt[f'{x}_Intensity']) + vol_asset2.map(lambda x: current_mkt[f'{x}_Intensity'])) / 2.0
    liquidity_loss = np.where(avg_intensity < 100, (100 - avg_intensity) * 20, 0)

    book_pnl_per_unit = - gamma_bleed - vega_loss - liquidity_loss

    initial_gamma = np.where(100 - results['ki_barrier'] > 0, 2000 * np.exp(-0.15 * (100 - results['ki_barrier'])), 0)
    initial_vol = (vol_asset1.map(lambda x: base_mkt[f'Vol_{x}']) + vol_asset2.map(lambda x: base_mkt[f'Vol_{x}'])) / 2.0
    initial_vega = np.maximum(0, initial_vol - 0.20) * 50000
    initial_pnl_per_unit = - initial_gamma - initial_vega

    results['old_price'] = initial_pnl_per_unit 
    results['new_price'] = book_pnl_per_unit
    results['price_change'] = results['new_price'] - results['old_price']
    results['pnl'] = results['price_change'] * results['qty']
    return results

def calculate_parametric_var(df_b, df_e, df_mkt, confidence_level=0.99):
    factors_df = df_mkt.drop(columns=['Date'])
    returns_df = factors_df.pct_change().dropna()
    cov_matrix = returns_df.cov()
    base_mkt = factors_df.iloc[-1].to_dict()

    def get_total_pnl(mkt_state):
        b_res = revalue_bonds_multi(df_b, mkt_state, base_mkt)
        e_res = revalue_els_multi(df_e, mkt_state, base_mkt)
        return b_res['pnl'].sum() + e_res['pnl'].sum()

    base_pnl = get_total_pnl(base_mkt)
    shock_size = 0.01
    sensitivities = {}

    for factor in cov_matrix.columns:
        shocked_mkt = base_mkt.copy()
        shocked_mkt[factor] = shocked_mkt[factor] * (1 + shock_size)
        shocked_pnl = get_total_pnl(shocked_mkt)
        sensitivities[factor] = (shocked_pnl - base_pnl) / shock_size

    S = pd.Series(sensitivities)
    portfolio_variance = S.dot(cov_matrix).dot(S)
    portfolio_std_dev = np.sqrt(portfolio_variance)

    z_score = norm.ppf(confidence_level)
    var_amount = z_score * portfolio_std_dev
    return abs(var_amount), S
