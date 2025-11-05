#!/usr/bin/env python3
"""
数据探索脚本 - 替代 notebook 01_data_exploration.ipynb
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print(f"🔍 项目根目录: {project_root}")
print(f"🔍 Python路径: {sys.executable}")

try:
    from config.settings import Settings
    from src.data_processor import DataProcessor
    from src.visualization import ResultVisualizer
    print("✅ 所有模块导入成功")
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    print("请检查项目结构是否正确")
    sys.exit(1)

def explore_data():
    """执行数据探索分析"""
    print("🔍 开始数据探索分析...")
    
    # 初始化
    processor = DataProcessor()
    visualizer = ResultVisualizer()
    
    try:
        # 加载数据
        df = processor.load_data()
        print(f"✅ 数据加载成功:")
        print(f"- 总记录数: {len(df):,}")
        print(f"- 时间范围: {df['Date'].min()} 到 {df['Date'].max()}")
        print(f"- 口岸数量: {df['Control Point'].nunique()}")
        
        # 基本统计信息
        print("\n📊 数据统计信息:")
        print(df.describe())
        
        # 口岸列表
        control_points = df['Control Point'].unique()
        print(f"\n🏢 所有口岸: {list(control_points)}")
        
        # 入境/出境分布
        arrival_departure_dist = df['Arrival / Departure'].value_counts()
        print(f"\n↕️ 入境/出境分布:\n{arrival_departure_dist}")
        
        # 创建探索性图表
        create_exploratory_charts(df, visualizer)
        
        # 目标口岸分析
        analyze_target_points(df, processor)
        
        print("\n✅ 数据探索完成!")
        
    except Exception as e:
        print(f"❌ 数据探索失败: {e}")
        import traceback
        traceback.print_exc()

def create_exploratory_charts(df, visualizer):
    """创建探索性图表"""
    print("\n📈 生成探索性图表...")
    
    try:
        # 总体趋势图
        plt.figure(figsize=(14, 10))
        
        # 1. 按日期聚合的总客流量
        plt.subplot(2, 2, 1)
        daily_total = df.groupby('Date')['Total'].sum()
        plt.plot(daily_total.index, daily_total.values)
        plt.title('每日总客流量趋势')
        plt.xlabel('日期')
        plt.ylabel('客流量')
        plt.xticks(rotation=45)
        
        # 2. 各口岸客流量分布
        plt.subplot(2, 2, 2)
        point_totals = df.groupby('Control Point')['Total'].sum().sort_values(ascending=False)
        point_totals.head(10).plot(kind='bar')
        plt.title('各口岸总客流量排名 (Top 10)')
        plt.xlabel('口岸')
        plt.ylabel('总客流量')
        plt.xticks(rotation=45)
        
        # 3. 入境 vs 出境比例
        plt.subplot(2, 2, 3)
        direction_totals = df.groupby('Arrival / Departure')['Total'].sum()
        plt.pie(direction_totals.values, labels=direction_totals.index, autopct='%1.1f%%')
        plt.title('入境 vs 出境比例')
        
        # 4. 旅客类型分布
        plt.subplot(2, 2, 4)
        visitor_types = ['Hong Kong Residents', 'Mainland Visitors', 'Other Visitors']
        type_totals = df[visitor_types].sum()
        plt.pie(type_totals.values, labels=type_totals.index, autopct='%1.1f%%')
        plt.title('旅客类型分布')
        
        plt.tight_layout()
        plt.savefig(os.path.join(Settings.RESULT_DIR, 'charts', 'data_exploration.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ 探索性图表已保存")
        
    except Exception as e:
        print(f"❌ 图表生成失败: {e}")

def analyze_target_points(df, processor):
    """分析目标口岸数据"""
    print("\n🎯 分析目标口岸...")
    
    for control_point in Settings.TARGET_CONTROL_POINTS:
        print(f"\n分析口岸: {control_point}")
        
        try:
            # 准备训练数据
            ts_data = processor.prepare_training_data(control_point)
            print(f"- 数据点数: {len(ts_data)}")
            print(f"- 时间范围: {ts_data.index.min()} 到 {ts_data.index.max()}")
            print(f"- 平均值: {ts_data.mean():.0f}")
            print(f"- 标准差: {ts_data.std():.0f}")
            
            # 创建单个口岸趋势图
            plt.figure(figsize=(10, 6))
            plt.plot(ts_data.index, ts_data.values, linewidth=2)
            plt.title(f'{control_point} - 非香港居民入境旅客趋势')
            plt.xlabel('日期')
            plt.ylabel('旅客数量')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            # 保存图表
            filename = f"{control_point.replace(' ', '_')}_trend.png"
            plt.savefig(os.path.join(Settings.RESULT_DIR, 'charts', filename), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ {control_point} 分析完成")
            
        except Exception as e:
            print(f"- 分析失败: {str(e)}")

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 启动数据探索脚本")
    print("=" * 60)
    explore_data()
    print("=" * 60)
    print("🎉 脚本执行结束")
    print("=" * 60)