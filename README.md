# 心衰抵抗/非抵抗预测模型
基于临床指标构建的 LightGBM 二分类预测模型

## 模型信息
- 最优模型：LightGBM
- 测试集 AUC：0.8459
- 准确率：0.8087
- 输入特征：7项临床指标

## 输入特征
1. 淋巴细胞绝对值(×10⁹/L)
2. 乳酸脱氢酶(U/L)
3. 血尿素氮(mg/dL)
4. 估算肾小球滤过率(mL/min/1.73m²)
5. 血钾(mmol/L)
6. 白蛋白(g/dL)
7. NLR

## 在线使用
https://share.streamlit.io/你的GitHub用户名/heart-failure-prediction/app.py