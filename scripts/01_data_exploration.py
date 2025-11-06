#!/usr/bin/env python3

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print(f"🔍 Project root: {project_root}")
print(f"🔍 Python path: {sys.executable}")

try:
    from config.settings import Settings
    from src.data_processor import DataProcessor
    from src.visualization import ResultVisualizer
    print("✅ All modules imported successfully")
except ImportError as e:
    print(f"❌ Module import failed: {e}")
    print("Please check if the project structure is correct")
    sys.exit(1)

def explore_data():
    """Perform data exploration analysis"""
    print("🔍 Starting data exploration analysis...")
    
    # 初始化
    processor = DataProcessor()
    visualizer = ResultVisualizer()
    
    try:
        # 设置 seaborn 样式
        sns.set_style("whitegrid")
        sns.set_palette("husl")

        # 加载数据
        df = processor.load_data()
        
        # 过滤数据，只保留2023-2025年的数据
        df['Date'] = pd.to_datetime(df['Date'])
        start_date = '2023-01-01'
        end_date = '2025-12-31'
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        
        print(f"✅ Data loaded and filtered successfully:")
        print(f"- Total records: {len(df):,}")
        print(f"- Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        print(f"- Number of control points: {df['Control Point'].nunique()}")
        
        # 基本统计信息
        print("\n📊 Data statistics:")
        print(df.describe())
        
        # 口岸列表
        control_points = df['Control Point'].unique()
        print(f"\n🏢 All control points: {list(control_points)}")
        
        # 入境/出境分布
        arrival_departure_dist = df['Arrival / Departure'].value_counts()
        print(f"\n↕️ Arrival/Departure distribution:\n{arrival_departure_dist}")
        
        # 创建探索性图表
        create_exploratory_charts(df, visualizer)
        
        # 目标口岸分析
        analyze_target_points(df, processor)
        
        print("\n✅ Data exploration completed!")
        
    except Exception as e:
        print(f"❌ Data exploration failed: {e}")
        import traceback
        traceback.print_exc()

def create_exploratory_charts(df, visualizer):
    """Create exploratory charts"""
    print("\n📈 Generating exploratory charts...")
    
    try:
        # 总体趋势图
        plt.figure(figsize=(14, 10))
        
        # 1. 按日期聚合的总客流量
        plt.subplot(2, 2, 1)
        daily_total = df.groupby('Date')['Total'].sum()
        sns.lineplot(x=daily_total.index, y=daily_total.values)
        plt.title('Daily Total Passenger Flow Trend (2023-2025)')
        plt.xlabel('Date')
        plt.ylabel('Passenger Flow')
        plt.xticks(rotation=45)
        
        # 2. 各口岸客流量分布
        plt.subplot(2, 2, 2)
        point_totals = df.groupby('Control Point')['Total'].sum().sort_values(ascending=False)
        top_10_points = point_totals.head(10)
        # 创建 DataFrame 用于 seaborn
        top_10_df = pd.DataFrame({
            'Control Point': top_10_points.index,
            'Total': top_10_points.values
        })
        sns.barplot(data=top_10_df, x='Total', y='Control Point')
        plt.title('Top 10 Control Points by Passenger Flow (2023-2025)')
        plt.xlabel('Total Passenger Flow')
        plt.ylabel('Control Point')
        
        # 3. 入境 vs 出境比例
        plt.subplot(2, 2, 3)
        direction_totals = df.groupby('Arrival / Departure')['Total'].sum()
        plt.pie(direction_totals.values, labels=direction_totals.index, autopct='%1.1f%%')
        plt.title('Arrival vs Departure Proportion (2023-2025)')
        
        # 4. 旅客类型分布
        plt.subplot(2, 2, 4)
        visitor_types = ['Hong Kong Residents', 'Mainland Visitors', 'Other Visitors']
        type_totals = df[visitor_types].sum()
        plt.pie(type_totals.values, labels=type_totals.index, autopct='%1.1f%%')
        plt.title('Visitor Type Distribution (2023-2025)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(Settings.RESULT_DIR, 'charts', 'data_exploration_2023_2025.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Exploratory charts saved")
        
    except Exception as e:
        print(f"❌ Chart generation failed: {e}")

def analyze_target_points(df, processor):
    """Analyze target control points data"""
    print("\n🎯 Analyzing target control points...")
    
    for control_point in Settings.TARGET_CONTROL_POINTS:
        print(f"\nAnalyzing control point: {control_point}")
        
        try:
            # 准备训练数据
            ts_data = processor.prepare_training_data(control_point)
            
            # 过滤时间序列数据，只保留2023-2025年
            start_date = '2023-01-01'
            end_date = '2025-12-31'
            ts_data = ts_data[(ts_data.index >= start_date) & (ts_data.index <= end_date)]
            
            print(f"- Data points: {len(ts_data)}")
            print(f"- Date range: {ts_data.index.min().strftime('%Y-%m-%d')} to {ts_data.index.max().strftime('%Y-%m-%d')}")
            print(f"- Mean: {ts_data.mean():.0f}")
            print(f"- Standard deviation: {ts_data.std():.0f}")
            
            # 创建单个口岸趋势图
            plt.figure(figsize=(10, 6))
            plt.plot(ts_data.index, ts_data.values, linewidth=2)
            plt.title(f'{control_point} - Non-Hong Kong Resident Arrivals Trend (2023-2025)')
            plt.xlabel('Date')
            plt.ylabel('Number of Passengers')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            # 保存图表
            filename = f"{control_point.replace(' ', '_')}_trend_2023_2025.png"
            plt.savefig(os.path.join(Settings.RESULT_DIR, 'charts', filename), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ {control_point} analysis completed")
            
        except Exception as e:
            print(f"- Analysis failed: {str(e)}")

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Starting data exploration script (2023-2025)")
    print("=" * 60)
    explore_data()
    print("=" * 60)
    print("🎉 Script execution completed")
    print("=" * 60)