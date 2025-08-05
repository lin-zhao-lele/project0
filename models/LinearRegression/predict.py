import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
import warnings
warnings.filterwarnings('ignore')

# 脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 工程根目录 = 脚本所在目录的上上级
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DATA_DIR = os.path.join(BASE_DIR, "results")

def load_model_and_scaler():
    """
    加载训练好的模型、标准化器和特征选择器
    如果模型文件不存在，则重新训练并保存
    """
    try:
        # 尝试加载已保存的模型
        model = joblib.load(os.path.join(DATA_DIR, 'house_price_model.pkl'))
        scaler = joblib.load(os.path.join(DATA_DIR, 'feature_scaler.pkl'))
        selector = joblib.load(os.path.join(DATA_DIR, 'feature_selector.pkl'))
        return model, scaler, selector
    except FileNotFoundError:
        # 如果模型文件不存在，则重新训练
        # 加载数据
        train_data = pd.read_csv(os.path.join(DATA_DIR, 'train_data.csv'))
        
        # 分离特征和目标变量
        X_train = train_data.drop('price', axis=1)
        y_train = train_data['price']
        
        # 特征选择
        selector = SelectKBest(score_func=f_regression, k=8)
        X_train_selected = selector.fit_transform(X_train, y_train)
        
        # 数据标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        
        # 训练一个线性回归模型
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # 保存模型、标准化器和特征选择器
        joblib.dump(model, os.path.join(DATA_DIR, 'house_price_model.pkl'))
        joblib.dump(scaler, os.path.join(DATA_DIR, 'feature_scaler.pkl'))
        joblib.dump(selector, os.path.join(DATA_DIR, 'feature_selector.pkl'))
        
        return model, scaler, selector


def predict_house_price(area, bedrooms, bathrooms, age, near_subway, floor, renovation_grade, green_space, orientation, has_elevator):
    """
    预测房价
    
    参数:
    area: 面积（平方米）
    bedrooms: 卧室数量
    bathrooms: 浴室数量
    age: 房屋年龄
    near_subway: 是否靠近地铁（1表示是，0表示否）
    floor: 楼层
    renovation_grade: 装修等级（1-5级）
    green_space: 绿化率（百分比）
    orientation: 房屋朝向（0=北, 1=东, 2=南, 3=西）
    has_elevator: 是否有电梯（1表示是，0表示否）
    """
    # 加载模型、标准化器和特征选择器
    model, scaler, selector = load_model_and_scaler()
    
    # 创建输入数据
    input_data = np.array([[area, bedrooms, bathrooms, age, near_subway, floor, renovation_grade, green_space, orientation, has_elevator]])
    
    # 创建特征名称
    feature_names = ['area', 'bedrooms', 'bathrooms', 'age', 'near_subway', 'floor', 'renovation_grade', 'green_space', 'orientation', 'has_elevator']
    input_df = pd.DataFrame(input_data, columns=feature_names)
    
    # 设置特征名称以匹配训练时的特征名称
    input_df.columns = feature_names
    
    # 特征选择
    input_selected = selector.transform(input_df)
    
    # 标准化输入数据
    input_scaled = scaler.transform(input_selected)
    
    # 预测
    predicted_price = model.predict(input_scaled)
    
    return predicted_price[0]


def main():
    print("房价预测系统")
    print("=" * 30)
    
    # 示例预测
    area = 150
    bedrooms = 3
    bathrooms = 2
    age = 5
    near_subway = 1
    floor = 10
    renovation_grade = 4
    green_space = 30
    orientation = 2  # 南向
    has_elevator = 1
    
    predicted_price = predict_house_price(area, bedrooms, bathrooms, age, near_subway, floor, renovation_grade, green_space, orientation, has_elevator)
    
    print(f"房屋信息:")
    print(f"  面积: {area} 平方米")
    print(f"  卧室: {bedrooms} 间")
    print(f"  浴室: {bathrooms} 间")
    print(f"  年龄: {age} 年")
    print(f"  靠近地铁: {'是' if near_subway else '否'}")
    print(f"  楼层: {floor} 层")
    print(f"  装修等级: {renovation_grade} 级")
    print(f"  绿化率: {green_space}%")
    print(f"  朝向: {'北' if orientation == 0 else '东' if orientation == 1 else '南' if orientation == 2 else '西'}")
    print(f"  有电梯: {'是' if has_elevator else '否'}")
    print(f"\n预测价格: {predicted_price:.2f} 元")

if __name__ == "__main__":
    main()