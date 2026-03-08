import gradio as gr
import numpy as np
import joblib
import os

# 加载本地模型（Render会同步GitHub的模型文件）
def load_model():
    try:
        model = joblib.load("best_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler, "✅ 模型加载成功"
    except Exception as e:
        return None, None, f"❌ 模型加载失败：{str(e)}"

# 初始化模型
model, scaler, load_status = load_model()

# 预测核心函数
def predict_heart_failure(
    淋巴细胞绝对值, 乳酸脱氢酶, 血尿素氮,
    估算肾小球滤过率, 血钾, 白蛋白, NLR
):
    # 模型加载失败时返回提示
    if model is None or scaler is None:
        return load_status
    
    # 数值校验
    try:
        features = np.array([
            [float(淋巴细胞绝对值), float(乳酸脱氢酶), float(血尿素氮),
             float(估算肾小球滤过率), float(血钾), float(白蛋白), float(NLR)]
        ])
    except:
        return "❌ 输入错误：请确保所有值都是数字"
    
    # 标准化+预测
    features_scaled = scaler.transform(features)
    pred_prob = model.predict_proba(features_scaled)[0]
    result = "抵抗" if pred_prob[1] > 0.5 else "非抵抗"
    
    # 返回易读结果
    return f"""
### 心衰抵抗/非抵抗预测结果
- 最终判断：**{result}**
- 抵抗概率：{pred_prob[1]:.1%}
- 非抵抗概率：{pred_prob[0]:.1%}

⚠️ 本结果仅用于学术研究，不构成临床诊疗建议
"""

# 搭建Gradio界面（适配Render）
with gr.Blocks(theme=gr.themes.Medical()) as demo:
    gr.Markdown(f"# 心衰预测工具\n**模型状态：{load_status}**")
    
    # 输入区（分组显示，更清晰）
    with gr.Row():
        with gr.Column():
            lym = gr.Number(label="淋巴细胞绝对值 (×10⁹/L)", value=1.2)
            ldh = gr.Number(label="乳酸脱氢酶 (U/L)", value=250)
            bun = gr.Number(label="血尿素氮 (mg/dL)", value=15)
        with gr.Column():
            egfr = gr.Number(label="估算肾小球滤过率 (mL/min/1.73m²)", value=60)
            k = gr.Number(label="血钾 (mmol/L)", value=4.5)
            alb = gr.Number(label="白蛋白 (g/dL)", value=35)
            nlr = gr.Number(label="NLR", value=5.2)
    
    # 预测按钮+结果输出
    predict_btn = gr.Button("开始预测", variant="primary")
    output = gr.Markdown()
    
    # 绑定按钮与预测函数
    predict_btn.click(
        fn=predict_heart_failure,
        inputs=[lym, ldh, bun, egfr, k, alb, nlr],
        outputs=output
    )

# 启动应用（适配Render的端口/网络配置）
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # 必须配置，Render需要公网访问
        server_port=int(os.getenv("PORT", 7860)),  # Render自动分配端口
        auth=None,
        show_error=True,
        share=False  # 禁用临时分享，避免冲突
    )
