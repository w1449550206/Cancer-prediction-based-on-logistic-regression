# 逻辑回归需要掌握的知识点
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200227205601263.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM1NDU2MDQ1,size_16,color_FFFFFF,t_70)
- 知道逻辑回归的损失函数
- 知道逻辑回归的优化方法
- 知道sigmoid函数
- 知道逻辑回归的应用场景
- **应用LogisticRegression实现逻辑回归预测**
- 知道精确率、召回率指标的区别
- 知道如何解决样本不均衡情况下的评估
- 了解ROC曲线的意义说明AUC指标大小
- 应用classification_report实现精确率、召回率计算
- 应用roc_auc_score实现指标计算

@[toc]
# 一、分类评估方法

> 复习:分类评估指标

## 1. 分类评估方法

### 1.1 精确率与召回率

#### 1.1.1 混淆矩阵

在分类任务下，预测结果(Predicted Condition)与正确标记(True Condition)之间存在四种不同的组合，构成混淆矩阵(适用于多分类)
**<font color=DarkTurquoise face="楷体">（交叉表）</font>**
**<font color=DarkTurquoise face="楷体">例子：</font>**
**<font color=DarkTurquoise face="楷体">真实值------预测结果
A --------------A
A---------------B
B---------------A
B--------------- B
A ---------------A
那么
真正例AA 
伪反例AB
伪正例BA
真反例BB

</font>**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200227170946124.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM1NDU2MDQ1,size_16,color_FFFFFF,t_70)

#### 1.1.2 精确率(Precision)与召回率(Recall)

- 精确率：预测结果为正例样本中**真实为正例**的比例（了解）**<font color=DarkTurquoise face="楷体">TP/(TP+FP)</font>**

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020022717095242.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM1NDU2MDQ1,size_16,color_FFFFFF,t_70)

- 召回率：真实为正例的样本中预测结果为正例的比例（查得全，对正样本的区分能力）**<font color=DarkTurquoise face="楷体">TP/(TP+FN)</font>**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200227170956618.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM1NDU2MDQ1,size_16,color_FFFFFF,t_70)
### 1.2 F1-score

还有其他的评估标准，F1-score，反映了模型的稳健型
**<font color=DarkTurquoise face="楷体">计算出了精确率和召回率并且知道了TP,FN,FP，就可以计算出F--score，体现了精确率和召回率，这个值越大越好</font>**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200227171002148.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM1NDU2MDQ1,size_16,color_FFFFFF,t_70)

------

### 1.3 分类评估报告api

**- sklearn.metrics.classification_report(y_true, y_pred, labels=[], target_names=None )**
- - y_true：真实目标值
  - y_pred：估计器预测目标值
  - labels:指定类别对应的数字
  - target_names：目标类别名称
  - return：每个类别精确率与召回率

```python
ret = classification_report(y_test, y_predict, labels=(2,4), target_names=("良性", "恶性"))
print(ret)
```
**<font color=DarkTurquoise face="楷体">labels=(2,4)本文中的数据2是良性，4是恶性，通过target_names=("良性", "恶性"))给了他们名字。**
**假设这样一个情况，如果99个样本癌症，1个样本非癌症，不管怎样我全都预测正例(默认癌症为正例),准确率就为99%但是这样效果并不好，这就是样本不均衡下的评估问题**



问题：**如何衡量样本不均衡下的评估**？


### 1.4 分类评估报告api的癌症演示

**<font color=DarkTurquoise face="楷体">至此，我们去癌症的案例中去尝试**
案例地址：[https://github.com/w1449550206/Cancer-prediction-based-on-logistic-regression.git](https://github.com/w1449550206/Cancer-prediction-based-on-logistic-regression.git)

```bash
#获取预测值
y_predict = estimate.predict(x_test)

res = classification_report(y_true=y_test, y_pred=y_predict,labels= [2,4],target_names=['良性','恶性'])

res

type(res)

print(res)
# support指的是样本
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200227174736627.png)

## 2. ROC曲线与AUC指标

### 2.1 TPR与FPR
**<font color=DarkTurquoise face="楷体">一个混淆矩阵只能计算出一个TPR一个FPR</font>**

- TPR = TP / (TP + FN)**<font color=DarkTurquoise face="楷体">正例的召回率</font>**
  - 所有真实类别为1的样本中，预测类别为1的比例
- FPR = FP / (FP + TN))**<font color=DarkTurquoise face="楷体">假例的召回率</font>**
  - 所有真实类别为0的样本中，预测类别为1的比例
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200227170946124.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM1NDU2MDQ1,size_16,color_FFFFFF,t_70)
<font color=DarkTurquoise face="楷体">
**例子：
五个样本
计算出概率后
设置阈值为0.5
可以获得预测值
拿预测值和真实值得到一个混淆矩阵**</font>


概率     | 预测值 | 真实值
-------- | -----| -----
0.3  |B |B
0.4  | B |B
0.2  | B |A
0.8  | A |A
0.7  | A |B

<font color=DarkTurquoise face="楷体">**改变阈值的话，虽然概率不变，但预测值也不一样，阈值从0,0.1,0.2。。。一直变到1
概率是固定的
阈值改变一次
就改变一次混淆矩阵**</font>
### 2.2 ROC曲线

- ROC曲线的横轴就是FPRate，纵轴就是TPRate，当二者相等时，表示的意义则是：对于不论真实类别是1还是0的样本，分类器预测为1的概率是相等的，此时AUC为0.5
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200227194441642.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM1NDU2MDQ1,size_16,color_FFFFFF,t_70)

<font color=DarkTurquoise face="楷体">**假值的召回率FPR作为横坐标**</font>

<font color=DarkTurquoise face="楷体">**真值的召回率TPR作为纵坐标**</font>

<font color=DarkTurquoise face="楷体">**Threshold是阈值的意思**</font>

<font color=DarkTurquoise face="楷体">**阈值从0变化到1，才会得到不同的TPR金额FPR，才可以有很多的点，去绘制ROC曲线**</font>

<font color=DarkTurquoise face="楷体">**模型越好，蓝线越弯向于左上角**</font>

<font color=DarkTurquoise face="楷体">**这样可以看出模型的好坏，对比两个模型的ROC**</font>

<font color=DarkTurquoise face="楷体">**但是从ROC图形来看模型好不好是我们人工看的，计算机怎么识别呢？这就需要AUC指标了**</font>

<font color=DarkTurquoise face="楷体">**红色的这根线就是随便猜测的一个模型，最差的得到的就是红色的这个模型**</font>


### 2.3 AUC指标
<font color=DarkTurquoise face="楷体">**AUC指标是由ROC曲线中得到来的，就是下面这部分的面积**</font>



![在这里插入图片描述](https://img-blog.csdnimg.cn/20200227202802501.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM1NDU2MDQ1,size_16,color_FFFFFF,t_70)
- AUC的概率意义是随机取一对正负样本，正样本得分大于负样本的概率
- AUC的最小值为0.5，最大值为1，取值越高越好 **<font color=DarkTurquoise face="楷体">AUC是面积，最大值是1，最小值是0.5，因为最差的就设置到0.5了，就是那条红色的虚线**</font>
- **AUC=1，完美分类器，采用这个预测模型时，不管设定什么阈值都能得出完美预测。绝大多数预测的场合，不存在完美分类器。
- **0.5<AUC<1，优于随机猜测。这个分类器（模型）妥善设定阈值的话，能有预测价值。**

> **最终AUC的范围在[0.5, 1]之间，并且越接近1越好**

### 2.4 AUC计算API

**- from sklearn.metrics import roc_auc_score**
  **- sklearn.metrics.roc_auc_score(y_true, y_score)**
    - 计算ROC曲线面积，即AUC值
    - y_true：每个样本的真实类别，必须为0(反例),1(正例)标记
    - y_score：**预测得分**，可以是正类的估计概率、置信值或者分类器方法的返回值

```python
# 0.5~1之间，越接近于1约好
y_test = np.where(y_test > 2.5, 1, 0)

print("AUC指标：", roc_auc_score(y_test, y_predict)
```

**<font color=DarkTurquoise face="楷体">不会错的计算方法：**

```bash
import numpy as np
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_c, y_d, pos_label=2)
metrics.auc(fpr, tpr)
```

### 2.5 ROC的AUC计算的癌症演示

**<font color=DarkTurquoise face="楷体">至此，我们去癌症的案例中去尝试**
案例地址：[https://github.com/w1449550206/Cancer-prediction-based-on-logistic-regression.git](https://github.com/w1449550206/Cancer-prediction-based-on-logistic-regression.git)







**<font color=DarkTurquoise face="楷体">需要注意的是，roc_auc_score 中如果没有pos_label是不能使用的，具体见[另一篇文章](https://vicky.blog.csdn.net/article/details/104545096)：[https://vicky.blog.csdn.net/article/details/104545096](https://vicky.blog.csdn.net/article/details/104545096)**

代码：

```bash
y_test

y_predict

y_a = y_test

y_b = y_predict

# # 尝试1 不行
# roc_auc_score(y_true=y_a,y_score=y_b)

# #尝试2 不行
# y_c = np.array(y_a,dtype='float64')

# y_d = np.array(y_b,dtype='float64')

# roc_auc_score(y_true=y_c,y_score=y_d)

# 尝试3 可以
import numpy as np
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_a, y_b, pos_label=2)
metrics.auc(fpr, tpr)

# 尝试4 可以
import numpy as np
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_c, y_d, pos_label=2)
metrics.auc(fpr, tpr)
```

**<font color=DarkTurquoise face="楷体">小技巧：在jupyter notebook中 shift+m 考科一合并cell**


# 二、ROC曲线的绘制

关于ROC曲线的绘制过程，通过以下举例进行说明

假设有6次展示记录，有两次被点击了，得到一个展示序列（1:1,2:0,3:1,4:0,5:0,6:0），前面的表示序号，后面的表示点击（1）或没有点击（0）。

然后在这6次展示的时候都通过model算出了点击的概率序列。

下面看三种情况。

## 1.曲线绘制

### 1.1 如果概率的序列是（1:0.9,2:0.7,3:0.8,4:0.6,5:0.5,6:0.4）。

与原来的序列一起，得到序列（从概率从高到低排）

| 1    | 1    | 0    | 0    | 0    | 0    |
| ---- | ---- | ---- | ---- | ---- | ---- |
| 0.9  | 0.8  | 0.7  | 0.6  | 0.5  | 0.4  |

绘制的步骤是：

1）把概率序列从高到低排序，得到顺序（1:0.9,3:0.8,2:0.7,4:0.6,5:0.5,6:0.4）；

2）从概率最大开始取一个点作为正类，取到点1，计算得到TPR=0.5，FPR=0.0；

3）从概率最大开始，再取一个点作为正类，取到点3，计算得到TPR=1.0，FPR=0.0；

4）再从最大开始取一个点作为正类，取到点2，计算得到TPR=1.0，FPR=0.25;

5）以此类推，得到6对TPR和FPR。

然后把这6对数据组成6个点(0,0.5),(0,1.0),(0.25,1),(0.5,1),(0.75,1),(1.0,1.0)。

这6个点在二维坐标系中能绘出来。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200227171118839.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM1NDU2MDQ1,size_16,color_FFFFFF,t_70)

看看图中，那个就是ROC曲线。

### 1.2 如果概率的序列是（1:0.9,2:0.8,3:0.7,4:0.6,5:0.5,6:0.4）

与原来的序列一起，得到序列（从概率从高到低排）

| 1    | 0    | 1    | 0    | 0    | 0    |
| ---- | ---- | ---- | ---- | ---- | ---- |
| 0.9  | 0.8  | 0.7  | 0.6  | 0.5  | 0.4  |

绘制的步骤是：

6）把概率序列从高到低排序，得到顺序（1:0.9,2:0.8,3:0.7,4:0.6,5:0.5,6:0.4）；

7）从概率最大开始取一个点作为正类，取到点1，计算得到TPR=0.5，FPR=0.0；

8）从概率最大开始，再取一个点作为正类，取到点2，计算得到TPR=0.5，FPR=0.25；

9）再从最大开始取一个点作为正类，取到点3，计算得到TPR=1.0，FPR=0.25;

10）以此类推，得到6对TPR和FPR。

然后把这6对数据组成6个点(0,0.5),(0.25,0.5),(0.25,1),(0.5,1),(0.75,1),(1.0,1.0)。

这6个点在二维坐标系中能绘出来。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200227171126455.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM1NDU2MDQ1,size_16,color_FFFFFF,t_70)

看看图中，那个就是ROC曲线。

### 1.3 如果概率的序列是（1:0.4,2:0.6,3:0.5,4:0.7,5:0.8,6:0.9）

与原来的序列一起，得到序列（从概率从高到低排）

| 0    | 0    | 0    | 0    | 1    | 1    |
| ---- | ---- | ---- | ---- | ---- | ---- |
| 0.9  | 0.8  | 0.7  | 0.6  | 0.5  | 0.4  |

绘制的步骤是：

11）把概率序列从高到低排序，得到顺序（6:0.9,5:0.8,4:0.7,2:0.6,3:0.5,1:0.4）；

12）从概率最大开始取一个点作为正类，取到点6，计算得到TPR=0.0，FPR=0.25；

13）从概率最大开始，再取一个点作为正类，取到点5，计算得到TPR=0.0，FPR=0.5；

14）再从最大开始取一个点作为正类，取到点4，计算得到TPR=0.0，FPR=0.75;

15）以此类推，得到6对TPR和FPR。

然后把这6对数据组成6个点(0.25,0.0),(0.5,0.0),(0.75,0.0),(1.0,0.0),(1.0,0.5),(1.0,1.0)。

这6个点在二维坐标系中能绘出来。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200227171134481.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM1NDU2MDQ1,size_16,color_FFFFFF,t_70)

看看图中，那个就是ROC曲线。

## 2.意义解释

如上图的例子，总共6个点，2个正样本，4个负样本，取一个正样本和一个负样本的情况总共有8种。

上面的第一种情况，从上往下取，无论怎么取，正样本的概率总在负样本之上，所以分对的概率为1，AUC=1。再看那个ROC曲线，它的积分是什么？也是1，ROC曲线的积分与AUC相等。

上面第二种情况，如果取到了样本2和3，那就分错了，其他情况都分对了；所以分对的概率是0.875，AUC=0.875。再看那个ROC曲线，它的积分也是0.875，ROC曲线的积分与AUC相等。

上面的第三种情况，无论怎么取，都是分错的，所以分对的概率是0，AUC=0.0。再看ROC曲线，它的积分也是0.0，ROC曲线的积分与AUC相等。

很牛吧，其实AUC的意思是——Area Under roc Curve，就是ROC曲线的积分，也是ROC曲线下面的面积。

绘制ROC曲线的意义很明显，不断地把可能分错的情况扣除掉，从概率最高往下取的点，每有一个是负样本，就会导致分错排在它下面的所有正样本，所以要把它下面的正样本数扣除掉（1-TPR，剩下的正样本的比例）。总的ROC曲线绘制出来了，AUC就定了，分对的概率也能求出来了。






