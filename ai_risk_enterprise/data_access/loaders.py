import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_market_data(days=30):
    """[미래: Market Data DB에서 시계열 데이터를 가져오는 역할]"""
    np.random.seed(42)
    dates = [datetime.now().date() - timedelta(days=x) for x in range(days)]
    dates.reverse()
    data = {'Date': dates}

    # KTB, Corp, KOSPI 등 가상 데이터 생성 (기존 로직 동일)
    data['KTB_6M'] = np.linspace(3.20, 3.45, days) + np.random.normal(0, 0.03, days)
    data['KTB_1Y'] = np.linspace(3.25, 3.55, days) + np.random.normal(0, 0.04, days)
    data['KTB_3Y'] = np.linspace(3.30, 3.70, days) + np.random.normal(0, 0.05, days)
    data['KTB_5Y'] = np.linspace(3.40, 3.90, days) + np.random.normal(0, 0.06, days)
    data['Corp_6M'] = np.linspace(3.80, 4.10, days) + np.random.normal(0, 0.05, days)
    data['Corp_1Y'] = np.linspace(3.90, 4.30, days) + np.random.normal(0, 0.06, days)

    data['KOSPI200_Close'] = np.linspace(360, 330, days) + np.random.normal(0, 3, days)
    data['Samsung_Close'] = np.linspace(78000, 71000, days) + np.random.normal(0, 800, days)
    data['SKHynix_Close'] = np.linspace(160000, 145000, days) + np.random.normal(0, 1500, days)
    data['Naver_Close'] = np.linspace(200000, 180000, days) + np.random.normal(0, 2000, days)

    data['KOSPI200_Intensity'] = np.random.uniform(85, 115, days)
    data['Samsung_Intensity'] = np.random.uniform(90, 120, days)
    data['SKHynix_Intensity'] = np.random.uniform(80, 110, days)
    data['Naver_Intensity'] = np.random.uniform(70, 105, days)

    data['Vol_KOSPI200'] = np.linspace(0.15, 0.28, days) + np.random.normal(0, 0.01, days)
    data['Vol_Samsung'] = np.linspace(0.18, 0.32, days) + np.random.normal(0, 0.01, days)
    data['Vol_SKHynix'] = np.linspace(0.25, 0.40, days) + np.random.normal(0, 0.02, days)
    data['Vol_Naver'] = np.linspace(0.30, 0.45, days) + np.random.normal(0, 0.02, days)

    return pd.DataFrame(data)

def get_sell_side_portfolio():
    """[미래: Position DB에서 증권사 보유 채권/ELS 원장 데이터를 가져오는 역할]"""
    bond_portfolio = [
        {"bond_id": "B01", "name": "국고채_6개월_A", "tenor": 0.5, "curve": "KTB_6M", "face_value": 10000, "qty": 1000},
        {"bond_id": "B02", "name": "국고채_1년_A", "tenor": 1.0, "curve": "KTB_1Y", "face_value": 10000, "qty": 1500},
        {"bond_id": "B03", "name": "국고채_3년_A", "tenor": 3.0, "curve": "KTB_3Y", "face_value": 10000, "qty": 2000},
        {"bond_id": "B04", "name": "국고채_5년_A", "tenor": 5.0, "curve": "KTB_5Y", "face_value": 10000, "qty": 800},
        {"bond_id": "B05", "name": "회사채_6개월_A", "tenor": 0.5, "curve": "Corp_6M", "face_value": 10000, "qty": 1200},
        {"bond_id": "B06", "name": "회사채_1년_A", "tenor": 1.0, "curve": "Corp_1Y", "face_value": 10000, "qty": 2500},
        {"bond_id": "B07", "name": "국고채_1년_B", "tenor": 1.0, "curve": "KTB_1Y", "face_value": 10000, "qty": 3000},
        {"bond_id": "B08", "name": "국고채_3년_B", "tenor": 3.0, "curve": "KTB_3Y", "face_value": 10000, "qty": 1000},
        {"bond_id": "B09", "name": "회사채_6개월_B", "tenor": 0.5, "curve": "Corp_6M", "face_value": 10000, "qty": 5000},
        {"bond_id": "B10", "name": "회사채_1년_B", "tenor": 1.0, "curve": "Corp_1Y", "face_value": 10000, "qty": 2000},
    ]
    
    els_portfolio = [
        {"els_id": "E01", "name": "ELS_KOSPI_SAMSUNG_KI50", "asset1": "KOSPI200_Close", "asset2": "Samsung_Close", "ki_barrier": 50, "qty": 1000},
        {"els_id": "E02", "name": "ELS_KOSPI_HYNIX_KI55", "asset1": "KOSPI200_Close", "asset2": "SKHynix_Close", "ki_barrier": 55, "qty": 1200},
        {"els_id": "E03", "name": "ELS_SAMSUNG_NAVER_KI55", "asset1": "Samsung_Close", "asset2": "Naver_Close", "ki_barrier": 55, "qty": 800},
        {"els_id": "E04", "name": "ELS_KOSPI_NAVER_KI60", "asset1": "KOSPI200_Close", "asset2": "Naver_Close", "ki_barrier": 60, "qty": 1500},
        {"els_id": "E05", "name": "ELS_HYNIX_NAVER_KI60", "asset1": "SKHynix_Close", "asset2": "Naver_Close", "ki_barrier": 60, "qty": 2000},
        {"els_id": "E06", "name": "ELS_KOSPI_SAMSUNG_KI65", "asset1": "KOSPI200_Close", "asset2": "Samsung_Close", "ki_barrier": 65, "qty": 1000},
        {"els_id": "E07", "name": "ELS_KOSPI_HYNIX_KI65", "asset1": "KOSPI200_Close", "asset2": "SKHynix_Close", "ki_barrier": 65, "qty": 900},
        {"els_id": "E08", "name": "ELS_SAMSUNG_NAVER_KI70", "asset1": "Samsung_Close", "asset2": "Naver_Close", "ki_barrier": 70, "qty": 1100},
        {"els_id": "E09", "name": "ELS_HYNIX_NAVER_KI70", "asset1": "SKHynix_Close", "asset2": "Naver_Close", "ki_barrier": 70, "qty": 3000},
        {"els_id": "E10", "name": "ELS_KOSPI_NAVER_KI70", "asset1": "KOSPI200_Close", "asset2": "Naver_Close", "ki_barrier": 70, "qty": 2500},
    ]
    return pd.DataFrame(bond_portfolio), pd.DataFrame(els_portfolio)

def get_buy_side_funds():
    """[미래: Fund DB에서 자산운용사 펀드 정보를 가져오는 역할]"""
    fund_data = {
        'Fund_Name': ['KOSPI 200 인덱스 펀드', 'TIGER 나스닥100 (UH)', 'TIGER 나스닥100 (H)', 'TIGER 미국테크 커버드콜', '글로벌 상업용 부동산 펀드'],
        'Current_AUM': [50000, 80000, 60000, 40000, 70000],
        'Fee_Rate(%)': [0.15, 0.50, 0.50, 0.70, 1.50],
        'Hedge_Type':  ['None', 'UH', 'H', 'UH', 'H'],
        'Delta':       [1.0, 1.0, 1.0, 0.6, 0.0],
        'Gamma':       [0.0, 0.0, 0.0, -1.2, 0.0],
        'Rate_Beta':   [0.0, 0.0, 0.0, 0.0, -6.0]
    }
    return pd.DataFrame(fund_data).set_index('Fund_Name'), 800 # df_base와 fixed_costs 반환