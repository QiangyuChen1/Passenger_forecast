# check_environment.py
import sys

def check_environment():
    print("🔍 检查项目环境...")
    print(f"Python 版本: {sys.version}")
    print(f"Python 路径: {sys.executable}")
    
    required_packages = [
        'pandas', 'numpy', 'statsmodels', 'matplotlib', 
        'seaborn', 'plotly', 'sklearn', 'jupyter'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
                version = sklearn.__version__
            else:
                module = __import__(package)
                version = getattr(module, '__version__', '未知版本')
            print(f"✅ {package:15} - 版本: {version}")
        except ImportError:
            print(f"❌ {package:15} - 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ 缺少以下包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    else:
        print("\n🎉 所有依赖包已正确安装！")
        return True

if __name__ == "__main__":
    check_environment()