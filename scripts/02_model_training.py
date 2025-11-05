#!/usr/bin/env python3
"""
修复版模型训练脚本 V3 - 使用正确的口岸名称和修复的数据处理
"""

import os
import sys
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print(f"🔧 项目根目录: {project_root}")

try:
    from config.settings import Settings
    from src.data_processor import DataProcessor as DataProcessorFixed
    from src.model_trainer import ARIMAModelTrainer
    print("✅ 所有模块导入成功")
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    # 如果导入失败，尝试动态创建类
    print("⚠️ 尝试动态创建数据处理器...")
    from src.data_processor import DataProcessor as DataProcessorFixed
    from src.model_trainer import ARIMAModelTrainer

def train_models_fixed():
    """使用修复的参数训练模型"""
    print("🤖 开始修复版模型训练...")
    
    # 初始化
    processor = DataProcessorFixed()
    models_info = {}
    
    # 加载数据
    print("📊 加载数据...")
    raw_data = processor.load_data()
    print(f"数据加载完成，共 {len(raw_data)} 条记录")
    
    # 使用正确的口岸名称
    target_points = [
        'Airport',
        'Shenzhen Bay', 
        'Hong Kong-Zhuhai-Macao Bridge'
    ]
    
    for control_point in target_points:
        print(f"\n🔧 训练模型: {control_point}")
        
        try:
            # 准备数据（使用对数变换）
            print("- 准备训练数据...")
            ts_data = processor.prepare_training_data(control_point, use_log_transform=True)
            print(f"- 数据点数: {len(ts_data)}")
            
            # 检查数据质量
            if len(ts_data) < 12:
                print(f"⚠️ 数据点太少 ({len(ts_data)})，跳过 {control_point}")
                continue
                
            if ts_data.std() < 0.1:
                print(f"⚠️ 数据方差太小 ({ts_data.std():.4f})，跳过 {control_point}")
                continue
            
            # 训练模型 - 使用更保守的参数范围
            trainer = ARIMAModelTrainer(ts_data)
            
            # 平稳性检验
            print("- 检查平稳性...")
            is_stationary = trainer.check_stationarity()
            print(f"- 数据平稳性: {'是' if is_stationary else '否'}")
            
            if not is_stationary:
                print("- 进行一阶差分...")
                ts_data = trainer.difference_series(d=1)
                print(f"- 差分后数据点数: {len(ts_data)}")
            
            # 使用更保守的参数搜索
            print("- 寻找最佳ARIMA参数（保守搜索）...")
            best_params = trainer.find_best_arima_params(max_p=2, max_d=1, max_q=2)
            print(f"- 最佳参数: ARIMA{best_params}")
            
            # 训练最终模型
            print("- 训练最终模型...")
            model = trainer.train_model(best_params)
            
            # 模型诊断
            print("- 进行模型诊断...")
            residuals = trainer.diagnose_model()
            
            # 保存模型
            model_filename = f"arima_fixed_v3_{control_point.replace(' ', '_').replace('-', '_')}.pkl"
            model_path = os.path.join(Settings.MODEL_DIR, model_filename)
            
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'params': best_params,
                    'original_series': ts_data,
                    'used_log_transform': True,
                    'control_point': control_point
                }, f)
            
            # 存储模型信息
            models_info[control_point] = {
                'model_path': model_path,
                'params': best_params,
                'aic': model.aic
            }
            
            print(f"✅ {control_point} 模型训练完成并保存")
            
        except Exception as e:
            print(f"❌ {control_point} 模型训练失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 保存模型信息汇总
    if models_info:
        info_path = os.path.join(Settings.MODEL_DIR, 'models_summary_fixed_v3.csv')
        summary_df = pd.DataFrame([
            {
                'control_point': cp,
                'arima_order': info['params'],
                'aic': info['aic'],
                'model_path': info['model_path']
            }
            for cp, info in models_info.items()
        ])
        summary_df.to_csv(info_path, index=False)
        
        print(f"\n📋 修复版模型汇总已保存: {info_path}")
        print(f"🎉 成功训练 {len(models_info)} 个修复版模型!")
    else:
        print("❌ 没有成功训练任何修复版模型")

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 启动修复版模型训练脚本")
    print("=" * 60)
    train_models_fixed()
    print("=" * 60)
    print("🎉 脚本执行结束")
    print("=" * 60)