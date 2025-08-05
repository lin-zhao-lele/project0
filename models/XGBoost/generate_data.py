import pandas as pd
import numpy as np
import random
import os
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# 脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 工程根目录 = 脚本所在目录的上上级
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DATA_DIR = os.path.join(BASE_DIR, "results")

# 创建数据目录
os.makedirs(DATA_DIR, exist_ok=True)

def generate_credit_data(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, n_clusters_per_class=1):
    """
    生成信用卡用户风险评估数据集
    
    Parameters:
    n_samples (int): 样本数量
    n_features (int): 特征数量
    n_informative (int): 有用特征数量
    n_redundant (int): 冗余特征数量
    n_clusters_per_class (int): 每个类别的簇数
    
    Returns:
    X (DataFrame): 特征数据
    y (Series): 标签数据
    """
    # 生成分类数据
    X, y = make_classification(n_samples=n_samples, 
                               n_features=n_features, 
                               n_informative=n_informative, 
                               n_redundant=n_redundant, 
                               n_clusters_per_class=n_clusters_per_class, 
                               random_state=42)
    
    # 转换为DataFrame，只使用少量基础特征
    X = pd.DataFrame(X[:, :5], columns=['payment_history_score', 'credit_utilization_rate', 'debt_to_income_ratio', 'employment_stability', 'savings_ratio'])
    y = pd.Series(y, name='risk_flag')
    
    # 添加现实的特征
    # 性别 (0=女性, 1=男性)
    X['gender'] = np.random.randint(0, 2, n_samples)
    
    # 年龄 (18-80)
    X['age'] = np.random.randint(18, 81, n_samples)
    
    # 婚姻状况 (0=未婚, 1=已婚, 2=离异)
    X['marital_status'] = np.random.randint(0, 3, n_samples)
    
    # 教育水平 (1=高中以下, 2=高中, 3=专科, 4=本科, 5=研究生及以上)
    X['education_level'] = np.random.randint(1, 6, n_samples)
    
    # 工作年限 (0-40)
    X['employment_years'] = np.random.randint(0, 41, n_samples)
    
    # 收入 (20000-200000)
    X['income'] = np.random.randint(20000, 200001, n_samples)
    
    # 住房类型 (0=租房, 1=自有, 2=父母所有)
    X['housing_type'] = np.random.randint(0, 3, n_samples)
    
    # 车辆拥有情况 (0=无, 1=有)
    X['car_ownership'] = np.random.randint(0, 2, n_samples)
    
    # 家庭人口数 (1-8)
    X['family_size'] = np.random.randint(1, 9, n_samples)
    
    # 信用评分 (300-850)
    X['credit_score'] = np.random.randint(300, 851, n_samples)
    
    # 储蓄账户 (0-100000)
    X['savings_account'] = np.random.randint(0, 100001, n_samples)
    
    # 投资组合 (0-500000)
    X['investment_portfolio'] = np.random.randint(0, 500001, n_samples)
    
    # 账户余额 (0-50000)
    X['account_balance'] = np.random.randint(0, 50001, n_samples)
    
    # 债务收入比 (0.0-2.0)
    X['debt_to_income_ratio'] = np.round(np.random.uniform(0.0, 2.0, n_samples), 2)
    
    # 付款历史 (0-100)
    X['payment_history'] = np.random.randint(0, 101, n_samples)
    
    # 信用利用率 (0.0-1.0)
    X['credit_utilization'] = np.round(np.random.uniform(0.0, 1.0, n_samples), 2)
    
    # 破产历史 (0=无, 1=有)
    X['bankruptcy_history'] = np.random.randint(0, 2, n_samples)
    
    # 查询次数 (0-20)
    X['inquiry_count'] = np.random.randint(0, 21, n_samples)
    
    # 就业状况 (0=全职, 1=兼职, 2=自雇, 3=失业)
    X['employment_status'] = np.random.randint(0, 4, n_samples)
    
    # 历史违约次数 (0-10)
    X['default_history'] = np.random.randint(0, 11, n_samples)
    
    # 居住地址年数 (0-50)
    X['years_at_address'] = np.random.randint(0, 51, n_samples)
    
    # 贷款数量 (0-15)
    X['loan_count'] = np.random.randint(0, 16, n_samples)
    
    # 信用卡数量 (1-10)
    X['credit_card_count'] = np.random.randint(1, 11, n_samples)
    
    # 手机类型 (0=预付费, 1=合约, 2=高端)
    X['phone_type'] = np.random.randint(0, 3, n_samples)
    
    # 月支出 (500-15000)
    X['monthly_spending'] = np.random.randint(500, 15001, n_samples)
    
    # 网络使用情况 (0=低, 1=中, 2=高)
    X['internet_usage'] = np.random.randint(0, 3, n_samples)
    
    # 社交媒体活动 (0=低, 1=中, 2=高)
    X['social_media_activity'] = np.random.randint(0, 3, n_samples)
    
    # 公用事业账单支付 (0=准时, 1=偶尔延迟, 2=经常延迟)
    X['utility_bills_payment'] = np.random.randint(0, 3, n_samples)
    
    # 月收入比 (0.1-1.0)
    X['spending_income_ratio'] = np.round(np.random.uniform(0.1, 1.0, n_samples), 2)
    
    return X, y


def save_datasets():
    """
    生成并保存训练集和测试集
    """
    # 生成数据
    X, y = generate_credit_data()
    
    # 合并特征和标签
    data = pd.concat([X, y], axis=1)
    
    # 分割训练集和测试集
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # 保存训练集
    train_data.to_csv(os.path.join(DATA_DIR, 'train_data.csv'), index=False)
    print(f"训练集已保存，包含 {len(train_data)} 条记录")
    
    # 保存测试集
    test_data.to_csv(os.path.join(DATA_DIR, 'test_data.csv'), index=False)
    print(f"测试集已保存，包含 {len(test_data)} 条记录")
    
    # 生成推理数据（包含标签）
    inference_sample = data.sample(n=1000, random_state=42)
    inference_data = inference_sample.drop('risk_flag', axis=1)
    inference_data_with_label = inference_sample
    inference_data_with_label.to_csv(os.path.join(DATA_DIR, 'inference_data.csv'), index=False)
    print(f"推理数据集已保存，包含 {len(inference_data)} 条记录")


if __name__ == "__main__":
    save_datasets()