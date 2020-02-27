# 逻辑回归需要掌握的知识点
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200227104544726.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM1NDU2MDQ1,size_16,color_FFFFFF,t_70)
- 知道逻辑回归的损失函数
- 知道逻辑回归的优化方法
- 知道sigmoid函数
- 知道逻辑回归的应用场景
- 应用LogisticRegression实现逻辑回归预测
- 知道精确率、召回率指标的区别
- 知道如何解决样本不均衡情况下的评估
- 了解ROC曲线的意义说明AUC指标大小
- 应用classification_report实现精确率、召回率计算
- 应用roc_auc_score实现指标计算

@[toc]
# 逻辑回归api介绍

- **sklearn.linear_model.LogisticRegression(solver='liblinear', penalty=‘l2’, C = 1.0)**
  - solver可选参数:{'liblinear', 'sag', 'saga','newton-cg', 'lbfgs'}，
    - 默认: 'liblinear'；用于优化问题的算法。
    - 对于小数据集来说，“liblinear”是个不错的选择，而“sag”和'saga'对于大型数据集会更快。
    - 对于多类问题，只有'newton-cg'， 'sag'， 'saga'和'lbfgs'可以处理多项损失;“liblinear”仅限于“one-versus-rest”分类。
  - penalty：正则化的种类 **<font color=DarkTurquoise face="楷体">正则化项又叫做惩罚项，因为防止模型太精细了，要惩罚它</font>**
  - C：正则化力度 **<font color=DarkTurquoise face="楷体">和线性回归中的alpha是一个道理</font>**

> **默认将类别数量少的当做正例**

LogisticRegression方法相当于 SGDClassifier(loss="log", penalty=" "),SGDClassifier实现了一个普通的随机梯度下降学习。而使用LogisticRegression实现了SAG、

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200227114628512.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM1NDU2MDQ1,size_16,color_FFFFFF,t_70)
