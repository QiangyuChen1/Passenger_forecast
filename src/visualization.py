import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config.settings import Settings

class ResultVisualizer:
    def __init__(self):
        plt.style.use(Settings.PLOT_STYLE)
        self.fig_size = Settings.FIG_SIZE
        
        # 设置中文字体
        try:
            # Windows 系统
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
            # Mac 系统
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        except:
            print("⚠️ 中文字体设置失败，使用默认字体")
        
    def plot_forecast(self, historical_data, forecast_df, control_point):
        """绘制预测图表"""
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # 绘制历史数据
        ax.plot(historical_data.index, historical_data.values, 
               label='Historical Data', color='blue', linewidth=2, marker='o', markersize=4)
        
        # 绘制预测数据
        forecast_index = forecast_df.index
        ax.plot(forecast_index, forecast_df['point_forecast'], 
               label='Forecast', color='red', linewidth=2, linestyle='--', marker='s', markersize=4)
        
        # 绘制置信区间
        ax.fill_between(forecast_index, 
                       forecast_df['forecast_lower_80'], 
                       forecast_df['forecast_upper_80'],
                       alpha=0.3, color='red', label='80% Confidence Interval')
        
        ax.fill_between(forecast_index, 
                       forecast_df['forecast_lower_95'], 
                       forecast_df['forecast_upper_95'],
                       alpha=0.2, color='red', label='95% Confidence Interval')
        
        ax.set_title(f'{control_point} - Non-HK Visitor Arrivals Forecast', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Visitors')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 优化x轴标签
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def plot_trend_comparison(self, forecasts_dict):
        """比较不同口岸的预测趋势"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, (control_point, forecast_df) in enumerate(forecasts_dict.items()):
            color = colors[i % len(colors)]
            ax.plot(forecast_df.index, forecast_df['point_forecast'], 
                   label=control_point, linewidth=2, color=color, marker='o')
            
            # 添加置信区间
            ax.fill_between(forecast_df.index, 
                          forecast_df['forecast_lower_80'], 
                          forecast_df['forecast_upper_80'],
                          alpha=0.2, color=color)
        
        ax.set_title('Non-HK Visitor Arrivals Forecast Comparison', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Visitors')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig