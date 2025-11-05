import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Settings:
    """项目配置设置"""
    
    # 路径配置
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    RESULT_DIR = os.path.join(BASE_DIR, 'results')
    
    # 数据配置
    DATA_FILE = 'daily_passenger_traffic.csv'
    
    # 模型配置
    TARGET_CONTROL_POINTS = [
        'Airport',  # 香港国际机场
        'Shenzhen Bay',  # 深圳湾口岸
        'Hong Kong-Zhuhai-Macao Bridge'  # 港珠澳大桥口岸
    ]
    
    FORECAST_MONTHS = 3
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # ARIMA 参数搜索范围
    MAX_P = 3
    MAX_D = 2
    MAX_Q = 3
    
    # 可视化配置
    PLOT_STYLE = 'seaborn-v0_8'
    FIG_SIZE = (12, 6)
    DPI = 300
    
    @classmethod
    def create_directories(cls):
        """创建必要的目录结构"""
        directories = [
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR, 
            cls.MODEL_DIR,
            os.path.join(cls.RESULT_DIR, 'forecasts'),
            os.path.join(cls.RESULT_DIR, 'charts'),
            os.path.join(cls.RESULT_DIR, 'reports')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ 创建目录: {directory}")

# 初始化目录
Settings.create_directories()