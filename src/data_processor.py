import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config.settings import Settings

class DataProcessor:
    def __init__(self):
        self.data_path = os.path.join(Settings.RAW_DATA_DIR, Settings.DATA_FILE)
        self.df = None
        
    def load_data(self):
        """加载原始数据"""
        print(f"📁 加载数据文件: {self.data_path}")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")
            
        self.df = pd.read_csv(self.data_path)
        print(f"✅ CSV文件读取成功，共 {len(self.df)} 行")
        
        # 解析日期
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%d-%m-%Y')
        print(f"📅 日期范围: {self.df['Date'].min()} 到 {self.df['Date'].max()}")
        
        return self.df
    
    def filter_inbound_passengers(self, control_point):
        """筛选指定口岸的入境旅客数据"""
        if self.df is None:
            self.load_data()
            
        print(f"🛃 筛选 {control_point} 入境旅客数据...")
        
        # 筛选指定口岸的入境数据
        point_data = self.df[
            (self.df['Control Point'] == control_point) & 
            (self.df['Arrival / Departure'] == 'Arrival')
        ].copy()
        
        # 计算非香港居民总数
        point_data['Non_HK_Visitors'] = point_data['Mainland Visitors'] + point_data['Other Visitors']
        
        print(f"✅ {control_point} 入境数据: {len(point_data)} 行")
        return point_data
    
    def aggregate_monthly_data(self, control_point):
        """按月份聚合数据 - 修复版本"""
        print(f"📈 为 {control_point} 聚合月度数据...")
        
        point_data = self.filter_inbound_passengers(control_point)
        
        if len(point_data) == 0:
            raise ValueError(f"控制点 '{control_point}' 没有入境数据")
        
        # 按月聚合，使用mean而不是sum，避免异常波动
        monthly_data = point_data.groupby(
            pd.Grouper(key='Date', freq='ME')  # 使用 ME 替代已弃用的 M
        )['Non_HK_Visitors'].mean().reset_index()  # 使用均值而不是总和
        
        monthly_data = monthly_data.set_index('Date')
        print(f"✅ {control_point} 月度数据聚合完成，共 {len(monthly_data)} 个月")
        
        return monthly_data
    
    def prepare_training_data(self, control_point, use_log_transform=True):
        """准备模型训练数据 - 修复版本"""
        print(f"🔧 准备 {control_point} 的训练数据...")
        
        monthly_data = self.aggregate_monthly_data(control_point)
        
        # 确保数据连续性
        full_range = pd.date_range(
            start=monthly_data.index.min(),
            end=monthly_data.index.max(),
            freq='ME'  # 使用 ME 替代已弃用的 M
        )
        monthly_data = monthly_data.reindex(full_range)
        
        # 使用更稳健的缺失值填充
        monthly_data['Non_HK_Visitors'] = monthly_data['Non_HK_Visitors'].ffill()
        monthly_data['Non_HK_Visitors'] = monthly_data['Non_HK_Visitors'].bfill()
        monthly_data['Non_HK_Visitors'] = monthly_data['Non_HK_Visitors'].fillna(0)
        
        ts_data = monthly_data['Non_HK_Visitors']
        
        print(f"原始数据统计:")
        print(f"  最小值: {ts_data.min():.0f}")
        print(f"  最大值: {ts_data.max():.0f}")
        print(f"  平均值: {ts_data.mean():.0f}")
        print(f"  中位数: {ts_data.median():.0f}")
        
        # 应用对数变换来稳定方差（可选）
        if use_log_transform:
            # 确保所有值都大于0
            if (ts_data <= 0).any():
                # 如果有0或负值，先进行平移
                min_val = ts_data.min()
                if min_val <= 0:
                    shift = abs(min_val) + 1
                    ts_data = ts_data + shift
                    print(f"✅ 应用数据平移: +{shift}")
            
            ts_data = np.log(ts_data)  # 使用自然对数
            print("✅ 应用对数变换稳定方差")
        
        # 移除极端异常值（使用更严格的IQR方法）
        Q1 = ts_data.quantile(0.25)
        Q3 = ts_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR  # 使用3倍IQR而不是1.5倍
        upper_bound = Q3 + 3 * IQR
        
        # 将异常值限制在边界内
        ts_data_cleaned = ts_data.clip(lower=lower_bound, upper=upper_bound)
        
        if (ts_data_cleaned != ts_data).any():
            outliers_count = (ts_data < lower_bound).sum() + (ts_data > upper_bound).sum()
            print(f"⚠️  检测并处理了 {outliers_count} 个异常值")
        
        print(f"✅ {control_point} 训练数据准备完成")
        print(f"   最终数据点数: {len(ts_data_cleaned)}")
        print(f"   数据范围: {ts_data_cleaned.min():.2f} 到 {ts_data_cleaned.max():.2f}")
        
        return ts_data_cleaned