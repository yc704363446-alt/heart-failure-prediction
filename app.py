import streamlit as st
import lightgbm as lgb
import numpy as np
import pickle

# 页面配置
st.set_page_config(page_title="心衰抵抗预测模型", page_icon="❤️")
st.title("❤️ 心衰抵抗/非抵抗预测模型")
st.subheader("基于 LightGBM 最优模型")
st.write("输入7项临床指标，一键预测心衰抵抗风险")

# 加载训练好的模型
@st.cache_resource
def load_model():
    try:
        with open("best_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except:
        st.error("模型加载失败，请检查模型文件")

model = load_model()

# ---------------------- 7个真实特征输入框 ----------------------
st.markdown("### 请输入临床指标")
lymphocyte = st.number_input("淋巴细胞绝对值(×10⁹/L)", format="%.2f", value=1.50)
ldh = st.number_input("乳酸脱氢酶(U/L)", format="%.1f", value=200.0)
bun = st.number_input("血尿素氮(mg/dL)", format="%.1f", value=15.0)
egfr = st.number_input("估算肾小球滤过率(mL/min/1.73m²)", format="%.1f", value=60.0)
k = st.number_input("血钾(mmol/L)", format="%.2f", value=4.00)
alb = st.number_input("白蛋白(g/dL)", format="%.2f", value=35.00)
nlr = st.number_input("NLR", format="%.2f", value=2.00)

# 预测按钮
if st.button("🔍 开始预测"):
    # 组装特征顺序（必须和训练时一致）
    X = np.array([[lymphocyte, ldh, bun, egfr, k, alb, nlr]])
    
    # 模型预测
    prob = model.predict_proba(X)[0][1]
    pred_class = model.predict(X)[0]
    
    # 输出结果
    st.markdown("---")
    st.subheader("📊 预测结果")
    if pred_class == 1:
        st.error(f"⚠️  预测结果：心衰抵抗 | 风险概率：{prob:.2%}")
    else:
        st.success(f"✅ 预测结果：心衰非抵抗 | 风险概率：{prob:.2%}")
    
    # 模型性能
    st.write("📌 模型性能指标")
    st.write("AUC = 0.8459 | 准确率 = 0.8087 | F1 = 0.7843")