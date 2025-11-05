import pandas as pd
import numpy as np
from datetime import timedelta

class ForecastAnalyzer:
    def __init__(self, model, original_series):
        self.model = model
        self.original_series = original_series
        
    def generate_forecast(self, steps=3):
        """生成预测结果"""
        print(f"🔮 生成 {steps} 期预测...")
        forecast_result = self.model.get_forecast(steps=steps)
        
        # 创建预测索引（月份）
        last_date = self.original_series.index[-1]
        if pd.infer_freq(self.original_series.index) == 'M':
            forecast_index = pd.date_range(
                start=last_date + pd.offsets.MonthBegin(1), 
                periods=steps, 
                freq='M'
            )
        else:
            # 默认按月频率
            forecast_index = pd.date_range(
                start=last_date + pd.offsets.MonthBegin(1), 
                periods=steps, 
                freq='M'
            )
        
        forecast_df = pd.DataFrame({
            'point_forecast': forecast_result.predicted_mean.values,
            'forecast_lower_80': forecast_result.conf_int().iloc[:, 0],
            'forecast_upper_80': forecast_result.conf_int().iloc[:, 1],
            'forecast_lower_95': forecast_result.conf_int(alpha=0.05).iloc[:, 0],
            'forecast_upper_95': forecast_result.conf_int(alpha=0.05).iloc[:, 1]
        }, index=forecast_index)
        
        return forecast_df
    
    def analyze_trend(self, forecast_df):
        """分析预测趋势"""
        latest_actual = self.original_series.iloc[-1]
        first_forecast = forecast_df['point_forecast'].iloc[0]
        last_forecast = forecast_df['point_forecast'].iloc[-1]
        
        # 计算变化率
        short_term_change = ((first_forecast - latest_actual) / latest_actual) * 100
        overall_change = ((last_forecast - latest_actual) / latest_actual) * 100
        
        # 计算置信区间宽度（作为置信水平指标）
        confidence_width_ratio = (forecast_df['forecast_upper_95'] - forecast_df['forecast_lower_95']).mean() / forecast_df['point_forecast'].mean()
        
        if confidence_width_ratio < 0.2:
            confidence_level = '高'
        elif confidence_width_ratio < 0.4:
            confidence_level = '中'
        else:
            confidence_level = '低'
        
        trend_analysis = {
            'latest_actual': latest_actual,
            'short_term_change_pct': short_term_change,
            'overall_forecast_change_pct': overall_change,
            'trend_direction': '上升' if overall_change > 0 else '下降',
            'confidence_level': confidence_level,
            'confidence_width_ratio': confidence_width_ratio
        }
        
        return trend_analysis
    
    def generate_recommendations(self, control_point, trend_analysis):
        """基于预测结果生成建议"""
        recommendations = []
        
        change_pct = trend_analysis['overall_forecast_change_pct']
        direction = trend_analysis['trend_direction']
        
        if direction == '上升':
            if abs(change_pct) > 10:  # 显著变化
                if 'Airport' in control_point:
                    recommendations.extend([
                        f"预计{control_point}入境旅客将显著增长{change_pct:.1f}%，建议：",
                        "- 提前增加海关查验通道和工作人员",
                        "- 协调机场巴士和出租车调度，增加运力",
                        "- 通知机场零售和餐饮部门增加库存和人员",
                        "- 与航空公司协调增加航班时刻"
                    ])
                elif 'Bridge' in control_point:
                    recommendations.extend([
                        f"预计{control_point}客流将显著增长{change_pct:.1f}%，建议：",
                        "- 增加跨境巴士班次和座位数",
                        "- 检查口岸设施承载能力，必要时增加临时通道",
                        "- 与珠海、澳门方面协调通关安排",
                        "- 提前安排维护检查，确保设施正常运行"
                    ])
                elif 'Bay' in control_point:
                    recommendations.extend([
                        f"预计{control_point}入境客流将显著增长{change_pct:.1f}%，建议：",
                        "- 增加深圳湾口岸的通关通道",
                        "- 协调跨境交通接驳服务",
                        "- 通知周边酒店和旅游景点做好接待准备"
                    ])
            else:  # 温和变化
                recommendations.extend([
                    f"预计{control_point}入境旅客将温和{direction}{abs(change_pct):.1f}%，建议：",
                    "- 维持现有资源配置",
                    "- 密切监控实际客流变化",
                    "- 准备应急预案应对突发客流"
                ])
        else:  # 下降趋势
            recommendations.extend([
                f"预计客流将下降{abs(change_pct):.1f}%，建议：",
                "- 适当调整资源分配，避免浪费",
                "- 分析下降原因，制定营销策略",
                "- 关注市场变化，准备恢复计划"
            ])
            
        # 添加基于置信水平的建议
        if trend_analysis['confidence_level'] == '低':
            recommendations.append("- ⚠️ 预测置信度较低，建议谨慎决策并密切监控实际数据")
        elif trend_analysis['confidence_level'] == '高':
            recommendations.append("- ✅ 预测置信度较高，可基于此制定详细计划")
            
        return recommendations