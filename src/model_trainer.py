import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

class ARIMAModelTrainer:
    def __init__(self, time_series):
        self.ts = time_series
        self.model = None
        self.fitted_model = None
        self.best_params = None
        
    def check_stationarity(self):
        """ADF平稳性检验"""
        result = adfuller(self.ts.dropna())
        print('ADF统计量: %f' % result[0])
        print('p-value: %f' % result[1])
        print('临界值:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))
        
        return result[1] <= 0.05  # 返回是否平稳
    
    def difference_series(self, d=1):
        """对序列进行差分"""
        diff_series = self.ts.copy()
        for _ in range(d):
            diff_series = diff_series.diff().dropna()
        return diff_series
    
    def find_best_arima_params(self, max_p=3, max_d=2, max_q=3):
        """通过网格搜索寻找最佳ARIMA参数"""
        best_aic = np.inf
        best_params = (0, 0, 0)
        
        print("🔍 搜索最佳ARIMA参数...")
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        # 跳过全为0的情况
                        if p == 0 and d == 0 and q == 0:
                            continue
                            
                        # 准备数据
                        if d > 0:
                            temp_ts = self.ts.diff(d).dropna()
                            # 检查差分后数据是否有效
                            if len(temp_ts) < 5:
                                continue
                        else:
                            temp_ts = self.ts
                            
                        model = ARIMA(temp_ts, order=(p, d, q))
                        fitted_model = model.fit()
                        
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_params = (p, d, q)
                            print(f"  新最佳参数: ARIMA{best_params}, AIC: {best_aic:.2f}")
                            
                    except Exception as e:
                        # 忽略参数组合不收敛的情况
                        continue
        
        print(f"🎯 最终最佳参数: ARIMA{best_params}, AIC: {best_aic:.2f}")
        self.best_params = best_params
        return best_params
    
    def train_model(self, order=None):
        """训练ARIMA模型"""
        if order is None:
            if self.best_params is None:
                self.find_best_arima_params()
            order = self.best_params
            
        print(f"🚀 训练ARIMA{order}模型...")
        self.model = ARIMA(self.ts, order=order)
        self.fitted_model = self.model.fit()
        
        print(self.fitted_model.summary())
        return self.fitted_model
    
    def diagnose_model(self):
        """模型诊断"""
        if self.fitted_model is None:
            raise ValueError("请先训练模型")
            
        # 残差分析
        residuals = self.fitted_model.resid.dropna()
        
        # Ljung-Box检验（残差是否为白噪声）
        print("📊 进行模型诊断...")
        lb_test = acorr_ljungbox(residuals, lags=10)
        print("Ljung-Box检验p值:", lb_test['lb_pvalue'].values)
        
        # 检查残差是否近似白噪声（p值应该大于0.05）
        is_white_noise = all(lb_test['lb_pvalue'] > 0.05)
        print(f"残差是否为白噪声: {'是' if is_white_noise else '否'}")
        
        return residuals