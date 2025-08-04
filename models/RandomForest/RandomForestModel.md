```
"auto_tune": true → 自动调参   false → 用配置里的参数直接训练

"model":     "true 表示只加载已有模型 FinalModel 进行预测，false 表示重新训练并保存模型"


```


# 模型评价

``` 
滑动窗口的RandomForest模型，用20150101_20241231的数据作为training data 预测20250101_20250730的收盘价
 MSE 是541. R2是0.88，
 
加大数据量后用20010827_20241231的数据作为training data 预测20250101_20250730的收盘价
 MSE 是3439. R2是0.24 
 

这个现象挺典型的，数据量变大后模型表现反而变差（MSE变大，R²显著下降）
 
1. 训练数据跨度变大，数据分布变化（非平稳）
-- 用较短时间段（20150101-20241231）训练，数据相对“集中”，特征和目标的统计特征更稳定。

-- 加入更早的历史数据（从20010827开始），可能引入了很多和最近数据不同分布的样本。金融时间序列经常表现为非平稳，
即早期市场结构、价格行为、波动性可能跟近几年大不一样。

--模型难以同时拟合这么长时间跨度内的复杂且变化的关系，导致泛化能力下降。

2. 数据质量和特征相关问题
-- 早期数据可能质量较差（缺失、噪声），或者有较多异常值。

-- 滑动窗口构造的特征是固定长度的时间段内历史值，窗口内较早的数据可能包含与当前预测目标无关或者误导性的信息。

-- 没有做针对不同时间段的特征归一化或标准化，也可能导致训练过程中模型难以收敛到好的效果。


3. 模型复杂度和容量不足
-- 随机森林模型本身是基于树的集成方法，虽然强大，但对于长时间跨度的非平稳时间序列，其“记忆”能力有限。

-- 可能需要更复杂的特征工程（比如分段训练、分时段特征处理）或者更强大的模型（深度学习模型、带时间上下文的模型）来捕捉长期变化。

4. 过拟合 vs 欠拟合的表现
-- 用较短数据时表现较好，说明模型对这段数据学习的比较充分。

-- 扩展数据后MSE大幅升高且R²下降，说明模型无法有效拟合所有数据，可能是欠拟合（模型没能学到更复杂的模式），也可能是训练目标变复杂，导致泛化性能下降。

5. 训练集和测试集时间序列分布差异
-- 如果2001-2014年数据和2025年预测数据的市场环境差异非常大，模型学到的规则在未来测试集上失效。

-- 训练数据和测试数据的分布差异（数据漂移）是时间序列预测常见难题。

总结：你目前用随机森林滑动窗口做时间序列预测，短期数据表现好，长期数据和未来表现差，属于典型的时间序列非平稳和模型泛化能力不足问题。适当加强数据处理和模型设计，可以提升效果。

 
随机森林模型本身是基于树的集成方法，虽然强大，但对于长时间跨度的非平稳时间序列，其“记忆”能力有限。

可能需要更复杂的特征工程（比如分段训练、分时段特征处理）或者更强大的模型（深度学习模型、带时间上下文的模型）
来捕捉长期变化。
 
```

# 对该模型的评价和建议

``` 
评价
-- 随机森林滑动窗口模型能在相对稳定的数据区间表现不错（R²约0.88，MSE较小），说明它能捕捉一定的短期依赖关系和非线性模式。

-- 但在扩展历史数据后表现大幅下降，表明该模型对于长跨度且潜在非平稳的金融数据，泛化能力和适应能力有限。

-- 随机森林不具备像RNN/LSTM那样的序列记忆结构，难以捕捉长期依赖和复杂动态变化。

建议
-- 分段训练：考虑将数据按时间分段建模，比如近几年单独训练，再对不同时间段模型加权集成。

-- 特征工程：

   对不同时间段数据分别归一化或标准化。

   增加宏观经济指标、行业指标等外部特征，帮助模型捕捉宏观变化。

-- 模型升级：

   尝试时序深度模型（LSTM、GRU、Transformer）。

   结合滑动窗口和深度模型，或多模型集成。

-- 滑动窗口设计：

   适当调整窗口大小。

   考虑多尺度窗口组合。

-- 数据质量检查：

   清理异常值，处理缺失。

   确保数据一致性和准确性。

-- 模型评估：

   观察残差，定位哪些时间段预测差异大。

   做更多分层分析（按年份、行情涨跌区间）评估模型表现。 
```


```angular2html

编写python代码，用随机森林模型来预测股票价格
数据文件 表头为 ['open', 'high', 'low', 'vol', 'amount', 'turnover_rate', 'turnover_rate_f', 'volume_ratio','ma5', 'ma10', 'return_1d', 'vol_ma5, "close"] 其中"close"为目标，
存放位置为
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 脚本所在目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))  # 工程根目录
DATA_DIR = os.path.join(os.path.join(PROJECT_ROOT, "data"), "processed")
，
训练好的模型保存在当前目录模型 名为RandomForestSliding；
定义函数
def resolve_path(path_str, base="project"):
    if os.path.isabs(path_str):
        return os.path.normpath(path_str)
    if base == "project":
        return os.path.normpath(os.path.join(PROJECT_ROOT, path_str))
    elif base == "script":
        return os.path.normpath(os.path.join(BASE_DIR, path_str))
    elif base == "data":
        return os.path.normpath(os.path.join(DATA_DIR, path_str))

使用配置文件存放输入数据的位置
config_path = resolve_path("RandomForestSliding_args.json", base="script")；

使用滑动窗口处理训练数据，使得模型可以适用于连续的时间序列，"window_size"默认为5，这个参数也放入配置json，
的例子
{
    "training": "002594.SZ_20150101_20241231_daily_adjusted.csv",
    "predict": "002594.SZ_20250101_20250730_daily_adjusted.csv",
    "model": false,
    "auto_tune": true
}
；
"training"是训练数据集 "predict"是预测数据集；
训练模型支持optuna自动调参，控制参数为"auto_tune": true；
字段"model": "true 表示只加载保存在当前目录模型 RandomForestSliding 进行预测，false 表示重新训练并保存模型"；
给出测试和预测阶段的MSE  和 R2；
对训练数据进行必要的处理；

将最终预测结果保存RandomForestSliding_inference_output.csv 并可视化存为 RandomForestSliding_inference_plot.png


```