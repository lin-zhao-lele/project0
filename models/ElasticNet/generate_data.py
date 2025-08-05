import os
import numpy as np
import pandas as pd
import random

# 设置随机种子以确保结果可重现
np.random.seed(42)
random.seed(42)

# 脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def generate_house_data(n_samples=1000):
    """
    生成房价数据集
    特征包括：面积、卧室数量、浴室数量、年龄、是否靠近地铁、楼层、装修等级、绿化率等
    """
    # 生成基础特征
    area = np.random.normal(150, 50, n_samples)  # 面积（平方米）
    area = np.clip(area, 50, 300)  # 限制在合理范围内

    bedrooms = np.random.randint(1, 6, n_samples)  # 卧室数量
    bathrooms = np.random.randint(1, 4, n_samples)  # 浴室数量

    age = np.random.randint(0, 50, n_samples)  # 房屋年龄

    # 是否靠近地铁（0或1）
    near_subway = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])

    # 新增特征
    floor = np.random.randint(1, 31, n_samples)  # 楼层（1-30层）

    # 装修等级（1-5级，5级最好）
    renovation_grade = np.random.randint(1, 6, n_samples)

    # 小区绿化率（百分比）
    green_space = np.random.uniform(10, 60, n_samples)

    # 房屋朝向（0=北, 1=东, 2=南, 3=西）
    orientation = np.random.randint(0, 4, n_samples)

    # 是否有电梯（0或1）
    has_elevator = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])

    # 添加一些噪声
    noise = np.random.normal(0, 20000, n_samples)

    # 计算价格（基于特征的线性组合加上噪声）
    price = (area * 3000 +
             bedrooms * 10000 +
             bathrooms * 15000 +
             (50 - age) * 1000 +
             near_subway * 50000 +
             floor * 2000 +
             renovation_grade * 15000 +
             green_space * 1000 +
             (orientation == 2) * 20000 +  # 南向房屋更贵
             has_elevator * 30000 +
             noise)

    # 确保价格为正数
    price = np.clip(price, 100000, 2000000)

    # 创建DataFrame
    data = pd.DataFrame({
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age': age,
        'near_subway': near_subway,
        'floor': floor,
        'renovation_grade': renovation_grade,
        'green_space': green_space,
        'orientation': orientation,
        'has_elevator': has_elevator,
        'price': price
    })

    return data


def main():
    # 生成训练数据
    train_data = generate_house_data(1800)
    train_data.to_csv(os.path.join(RESULTS_DIR, 'train_data.csv'), index=False)
    print("训练数据已保存到 train_data.csv")

    # 生成推理数据（测试数据）
    test_data = generate_house_data(260)
    test_data.to_csv(os.path.join(RESULTS_DIR, 'test_data.csv'), index=False)
    print("测试数据已保存到 test_data.csv")


if __name__ == "__main__":
    main()