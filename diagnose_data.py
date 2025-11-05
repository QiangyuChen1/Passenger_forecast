#!/usr/bin/env python3
"""
检查实际数据值
"""

import pandas as pd
import numpy as np
import os

def check_actual_values():
    """检查各个口岸的实际数据值"""
    print("🔍 检查实际数据值...")
    
    # 读取原始数据
    df = pd.read_csv('data/raw/daily_passenger_traffic.csv')
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    
    target_points = ['Airport', 'Shenzhen Bay', 'Hong Kong-Zhuhai-Macao Bridge']
    
    for control_point in target_points:
        print(f"\n📊 {control_point}:")
        
        # 筛选入境数据
        point_data = df[
            (df['Control Point'] == control_point) & 
            (df['Arrival / Departure'] == 'Arrival')
        ].copy()
        
        # 计算非香港旅客
        point_data['Non_HK'] = point_data['Mainland Visitors'] + point_data['Other Visitors']
        
        # 按月聚合
        monthly_data = point_data.groupby(pd.Grouper(key='Date', freq='ME'))['Non_HK'].mean()
        
        print(f"最近6个月数据:")
        recent_data = monthly_data.tail(6)
        for date, value in recent_data.items():
            print(f"  {date.strftime('%Y-%m')}: {value:,.0f}")
        
        print(f"最新数据 ({monthly_data.index[-1].strftime('%Y-%m')}): {monthly_data.iloc[-1]:,.0f}")

if __name__ == "__main__":
    check_actual_values()