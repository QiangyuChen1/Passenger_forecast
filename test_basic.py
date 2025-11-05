import pandas as pd
import matplotlib.pyplot as plt
import os

# 读取数据
file_path = r"C:\Users\86181\Desktop\CITYU\SEMA\5001\Passenger_forecast\data\raw\daily_passenger_traffic.csv"
df = pd.read_csv(file_path, encoding='utf-8-sig')
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# 使用正确的口岸名称
target_control_points = ['Airport', 'Shenzhen Bay', 'Hong Kong-Zhuhai-Macao Bridge']

print("=== 验证口岸数据分析 ===")

for control_point in target_control_points:
    print(f"\n分析口岸: {control_point}")
    
    # 筛选特定口岸的入境数据
    cp_data = df[(df['Control Point'] == control_point) & 
                (df['Arrival / Departure'] == 'Arrival')]
    
    print(f"- 入境数据行数: {len(cp_data)}")
    
    if len(cp_data) > 0:
        # 按月聚合
        monthly_data = cp_data.groupby(pd.Grouper(key='Date', freq='ME'))['Total'].sum()
        print(f"- 月度数据点: {len(monthly_data)}")
        print(f"- 月度数据示例:")
        print(monthly_data.head())
        
        # 创建图表
        plt.figure(figsize=(12, 4))
        monthly_data.plot(title=f'{control_point} - Monthly Arrival Traffic')
        plt.tight_layout()
        
        # 保存图表
        chart_path = f'results/charts/{control_point.replace(" ", "_").replace("-", "_")}_analysis.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"- 图表保存到: {chart_path}")
        plt.close()
    else:
        print("- 没有找到数据")

print(f"\n=== 验证完成 ===")