# 基于复合型注意力机制的药物-靶标相互作用预测算法
为了全面、有效地解析药物－蛋白相互作用潜在的复杂本质，设计了一种名为DPAG（Drug-Target Interaction Prediction by Fusion of Attention and Graph Neural Network）的药物-靶标相互作用（DTI）预测模型。DPAG的独特之处在于其两种注意力机制结合在一起，并且加入了图神经网络，这有别于现存仅单独应用其中一种技术的模型。
# 依赖包
+ Python 3.8
+ torch 2.1
+ numpy >=1.24
+ scikit_learn
+ rdkit
+ transformers
# 结构
+ README.md：此文件。
+ data：里存放的是论文中使用的三个数据集
+ data_pre.py：数据处理。
+ gat.py：图神经网络。
+ hyperparameter.py：超参数
+ model.py：DPAG 模型结构。
+ transformer.py：transformer网络
# 运行
~~~
Python main.py
~~~
