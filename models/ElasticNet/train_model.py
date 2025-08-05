import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use('Agg')  # 非交互模式
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 解决中文和负号显示问题
if sys.platform == 'darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS'] # 首选苹方，备选Arial Unicode MS（更通用，但可能需要系统安装）
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
elif sys.platform == 'win32':  # Windows
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # Windows 上使用微软雅黑
    plt.rcParams['axes.unicode_minus'] = False


# 脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 工程根目录 = 脚本所在目录的上上级
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DATA_DIR = os.path.join(BASE_DIR, "results")


def load_and_prepare_data(train_file=os.path.join(DATA_DIR, 'train_data.csv'), test_file=os.path.join(DATA_DIR, 'test_data.csv')):
    """
    加载并准备数据
    """
    # 加载数据
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    
    # 分离特征和目标变量
    X_train = train_data.drop('price', axis=1)
    y_train = train_data['price']
    X_test = test_data.drop('price', axis=1)
    y_test = test_data['price']
    
    return X_train, X_test, y_train, y_test


def train_and_tune_model(X_train, X_test, y_train, y_test):
    """
    训练ElasticNet模型，包含特征选择
    """
    # 特征选择 - 选择最重要的8个特征
    selector = SelectKBest(score_func=f_regression, k=8)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # 获取选中的特征名称
    selected_features = X_train.columns[selector.get_support()].tolist()
    print(f"选中的特征: {selected_features}")
    
    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # 创建ElasticNet模型并进行超参数调优
    print("\n训练 ElasticNet 模型并进行超参数调优...")
    elastic_net = ElasticNet()
    
    # 定义超参数搜索空间
    param_grid = {
        'alpha': [0.1, 1.0, 10.0, 100.0],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    }
    
    # 使用GridSearchCV进行超参数调优
    grid_search = GridSearchCV(elastic_net, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    # 获取最佳模型
    model = grid_search.best_estimator_
    
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证得分: {grid_search.best_score_:.4f}")
    
    # 预测
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # 计算指标
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # 存储结果
    results = {
        'model': model,
        'scaler': scaler,
        'selector': selector,
        'selected_features': selected_features,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_mae': train_mae,
        'test_mae': test_mae
    }
    
    print(f"训练集 R2: {train_r2:.4f}")
    print(f"测试集 R2: {test_r2:.4f}")
    print(f"测试集 MSE: {test_mse:.2f}")
    print(f"测试集 MAE: {test_mae:.2f}")
    
    return results


def plot_results(results):
    """
    可视化结果
    """
    # 创建指标对比图（仅显示ElasticNet模型的结果）
    train_r2_score = results['train_r2']
    test_r2_score = results['test_r2']
    
    x = np.arange(1)
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, [train_r2_score], width, label='训练集 R2')
    rects2 = ax.bar(x + width/2, [test_r2_score], width, label='测试集 R2')
    
    ax.set_xlabel('ElasticNet模型')
    ax.set_ylabel('R2 分数')
    ax.set_title('ElasticNet模型性能')
    ax.set_xticks(x)
    ax.set_xticklabels(['ElasticNet'])
    ax.legend()
    
    # 在柱状图上添加数值标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, 'model_performance.png'))
    plt.show()
    
    # 直接返回模型名称
    print("\n使用模型: ElasticNet")
    
    return 'ElasticNet'


def predict_with_model(results, X_test, y_test):
    """
    使用ElasticNet模型进行预测并可视化结果
    """
    model = results['model']
    scaler = results['scaler']
    selector = results['selector']
    
    # 特征选择
    X_test_selected = selector.transform(X_test)
    
    # 标准化测试数据
    X_test_scaled = scaler.transform(X_test_selected)
    
    # 预测
    y_pred = model.predict(X_test_scaled)
    
    # 绘制预测值vs真实值
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('真实价格')
    plt.ylabel('预测价格')
    plt.title('ElasticNet模型预测结果')
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, 'prediction_results.png'))
    plt.show()
    
    return y_pred


def main():
    # 加载数据
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    print("数据加载完成")
    
    # 训练ElasticNet模型
    results = train_and_tune_model(X_train, X_test, y_train, y_test)
    
    # 保存模型、标准化器和特征选择器
    import joblib
    joblib.dump(results['model'], os.path.join(DATA_DIR, 'house_price_model.pkl'))
    joblib.dump(results['scaler'], os.path.join(DATA_DIR, 'feature_scaler.pkl'))
    joblib.dump(results['selector'], os.path.join(DATA_DIR, 'feature_selector.pkl'))
    
    # 可视化结果
    model_name = plot_results(results)
    
    # 使用模型进行预测
    y_pred = predict_with_model(results, X_test, y_test)
    
    # 保存预测结果
    results_df = pd.DataFrame({
        '真实价格': y_test.values,
        '预测价格': y_pred
    })
    results_df.to_csv(os.path.join(DATA_DIR, 'prediction_results.csv'), index=False)
    print("\n预测结果已保存到 results\\prediction_results.csv")

if __name__ == "__main__":
    main()