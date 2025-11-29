import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image
import pandas as pd
import google.generativeai as genai

# ページ設定
st.set_page_config(page_title="顔の表情分析アプリ Integrated", layout="wide")

st.title("😊 表情分析AIシステム")
st.markdown("数値による定量分析と、生成AIによる定性評価を統合したシステムです。")

# 日本語変換用辞書（定量分析用）
EMOTION_TRANSLATION = {
    "angry": "怒り",
    "disgust": "嫌悪",
    "fear": "恐れ",
    "happy": "喜び",
    "sad": "悲しみ",
    "surprise": "驚き",
    "neutral": "無表情"
}

# --- サイドバー設定 ---
st.sidebar.header("設定")

# モード選択
app_mode = st.sidebar.selectbox(
    "使用する機能を選択",
    ("📊 感情の定量分析 (DeepFace)", "📝 感情変化の定性分析 (Gemini)")
)

# 定性分析用APIキー入力（定性分析モードのときだけ表示、あるいは常時表示）
gemini_api_key = st.sidebar.text_input("Google AI Studio API Key", type="password", help="定性分析機能にはAPIキーが必要です")


# ==========================================
# 機能1: 既存の定量分析 (DeepFace)
# ==========================================
def run_quantitative_analysis():
    st.header("📊 感情の定量分析")
    st.write("DeepFaceを使用して、画像から感情数値を測定します。")

    min_confidence = st.sidebar.slider("検出感度", 0.0, 1.0, 0.5)
    
    input_option = st.radio("入力モード", ("画像をアップロード", "カメラで撮影"), horizontal=True)
    input_image = None

    if input_option == "画像をアップロード":
        uploaded_file = st.file_uploader("画像をアップロードしてください", type=['jpg', 'png', 'jpeg'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            input_image = np.array(image.convert('RGB'))

    elif input_option == "カメラで撮影":
        camera_image = st.camera_input("カメラで撮影してください")
        if camera_image is not None:
            image = Image.open(camera_image)
            input_image = np.array(image.convert('RGB'))

    if input_image is not None:
        st.divider()
        col_input, col_btn = st.columns([1, 2])
        with col_input:
            st.image(input_image, caption="入力画像", use_container_width=True)
        
        with col_btn:
            if st.button("🔍 分析を開始する", type="primary"):
                with st.spinner('DeepFace分析中...'):
                    try:
                        img_cv = input_image.copy()
                        results = DeepFace.analyze(img_cv, actions=['emotion'], enforce_detection=False)
                        if not isinstance(results, list): results = [results]

                        display_data = []
                        for res in results:
                            region = res['region']
                            emotion_eng = res['dominant_emotion']
                            scores = res['emotion']

                            if region['w'] < 20 or region['h'] < 20: continue

                            x, y, w, h = region['x'], region['y'], region['w'], region['h']
                            cv2.rectangle(img_cv, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            text = f"{emotion_eng} ({scores[emotion_eng]:.2f}%)"
                            cv2.putText(img_cv, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)

                            formatted_scores = {EMOTION_TRANSLATION.get(k, k): round(v, 2) for k, v in scores.items()}
                            display_data.append({"dominant_jp": EMOTION_TRANSLATION.get(emotion_eng, emotion_eng), "scores": formatted_scores})

                        if not display_data:
                            st.warning("顔が検出されませんでした。")
                        else:
                            st.image(img_cv, caption="検出結果", use_container_width=True)
                            for i, person in enumerate(display_data):
                                st.subheader(f"👤 顔 #{i+1} : {person['dominant_jp']}")
                                df = pd.DataFrame(list(person["scores"].items()), columns=["感情", "スコア (%)"])
                                df = df.sort_values(by="スコア (%)", ascending=False)
                                st.dataframe(df, hide_index=True, use_container_width=True, column_config={"スコア (%)": st.column_config.ProgressColumn("確信度", format="%.2f%%", min_value=0, max_value=100)})

                    except Exception as e:
                        st.error(f"エラー: {e}")


# ==========================================
# 機能2: 新規の定性分析 (Gemini Multimodal)
# ==========================================
def run_qualitative_analysis():
    st.header("📝 感情変化の定性分析")
    st.write("2枚の画像をAI（Gemini）が直接視認し、表情の変化や雰囲気を言葉で定性評価します。")

    col1, col2 = st.columns(2)
    img1_pil = None
    img2_pil = None

    with col1:
        st.subheader("1. 変化前 (Before)")
        file1 = st.file_uploader("1枚目を選択", type=['jpg', 'png', 'jpeg'], key="q_img1")
        if file1:
            img1_pil = Image.open(file1)
            st.image(img1_pil, use_container_width=True)

    with col2:
        st.subheader("2. 変化後 (After)")
        file2 = st.file_uploader("2枚目を選択", type=['jpg', 'png', 'jpeg'], key="q_img2")
        if file2:
            img2_pil = Image.open(file2)
            st.image(img2_pil, use_container_width=True)

    st.divider()

    if st.button("🤖 定性評価を実行する", type="primary"):
        if not gemini_api_key:
            st.error("⚠️ エラー: サイドバーでGoogle API Keyを設定してください。")
            return
        
        if img1_pil is None or img2_pil is None:
            st.warning("⚠️ 画像を2枚ともアップロードしてください。")
            return

        with st.spinner('Geminiが画像を観察し、レポートを作成中...'):
            try:
                # Geminiの設定
                genai.configure(api_key=gemini_api_key)
                
                # 画像処理に特化した軽量モデルを使用
                model = genai.GenerativeModel('gemini-2.0-flash')

                # プロンプトの作成
                prompt = """
                あなたは熟練した心理カウンセラーであり、表情分析の専門家です。
                以下の2枚の画像（1枚目が変化前、2枚目が変化後）を見て、人物の表情や雰囲気がどのように変化したか、定性的な評価を行ってください。

                以下の観点でレポートを作成してください：
                1. **全体的な印象**: パッと見た時の雰囲気の違い。
                2. **表情の詳細な変化**: 目つき、口角、眉間のシワなど、顔のパーツの具体的な変化。
                3. **推定される心理状態**: どのような感情の推移（例：緊張から緩和へ、あるいは悲しみから希望へなど）が見て取れるか。
                
                文章は丁寧な日本語で、観察結果に基づいた洞察を含めてください。
                """

                # マルチモーダル入力 (テキスト + 画像 + 画像)
                response = model.generate_content([prompt, img1_pil, img2_pil])
                
                st.subheader("📄 分析レポート")
                st.markdown(response.text)
                
            except Exception as e:
                st.error(f"APIエラーが発生しました: {e}")


# ==========================================
# メイン分岐
# ==========================================
if app_mode == "📊 感情の定量分析 (DeepFace)":
    run_quantitative_analysis()
elif app_mode == "📝 感情変化の定性分析 (Gemini)":
    run_qualitative_analysis()