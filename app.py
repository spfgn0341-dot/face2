import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image
import pandas as pd # 表作成用に追加

# ページ設定
st.set_page_config(page_title="顔の表情分析アプリ", layout="wide")

st.title("😊 顔の表情分析システム v2")
st.markdown("画像をアップロードするか、カメラで撮影して感情を分析します。")

# 日本語変換用辞書
EMOTION_TRANSLATION = {
    "angry": "怒り",
    "disgust": "嫌悪",
    "fear": "恐れ",
    "happy": "喜び",
    "sad": "悲しみ",
    "surprise": "驚き",
    "neutral": "無表情"
}

# サイドバー設定
st.sidebar.header("設定")
option = st.sidebar.selectbox(
    "入力モードを選択してください",
    ("画像をアップロード", "カメラで撮影")
)
min_confidence = st.sidebar.slider("検出感度（枠の調整用）", 0.0, 1.0, 0.5)

def analyze_emotion(image_np):
    try:
        img_cv = image_np.copy()
        
        # DeepFace分析
        results = DeepFace.analyze(img_cv, actions=['emotion'], enforce_detection=False)
        
        if not isinstance(results, list):
            results = [results]

        # 結果を格納するリスト（UI表示用）
        display_data = []

        for res in results:
            region = res['region']
            emotion_eng = res['dominant_emotion']
            scores = res['emotion']

            # 顔領域のフィルタリング（小さすぎる誤検知などを防ぐ簡易処理）
            if region['w'] < 20 or region['h'] < 20:
                continue

            # 画像への描画 (OpenCVは日本語NGなので英語のまま、小数点2桁に)
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            cv2.rectangle(img_cv, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            score_val = scores[emotion_eng]
            text = f"{emotion_eng} ({score_val:.2f}%)" # 小数点第2位まで
            cv2.putText(img_cv, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, (36, 255, 12), 2)

            # UI表示用のデータを整形（日本語化）
            formatted_scores = {}
            for k, v in scores.items():
                jp_key = EMOTION_TRANSLATION.get(k, k)
                formatted_scores[jp_key] = round(v, 2) # 値を丸める

            display_data.append({
                "dominant_jp": EMOTION_TRANSLATION.get(emotion_eng, emotion_eng),
                "scores": formatted_scores
            })
        
        return img_cv, display_data

    except Exception as e:
        st.error(f"エラー: {e}")
        return image_np, []

# --- メイン処理 ---

input_image = None

if option == "画像をアップロード":
    uploaded_file = st.file_uploader("画像をアップロードしてください", type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        input_image = np.array(image.convert('RGB'))

elif option == "カメラで撮影":
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
        st.write("準備ができました。下のボタンを押して分析を開始してください。")
        analyze_btn = st.button("🔍 表情を分析する", type="primary")

    if analyze_btn:
        with st.spinner('AIが分析中...'):
            result_img, analysis_data = analyze_emotion(input_image)
            
            st.divider()
            
            if not analysis_data:
                st.warning("顔が検出されませんでした。別の画像を試してください。")
            else:
                # 複数人の顔が検出された場合に対応
                for i, person_data in enumerate(analysis_data):
                    st.subheader(f"👤 検出された顔 #{i+1}")
                    
                    # カラム分け: 左に画像、右にデータ
                    res_col1, res_col2 = st.columns(2)
                    
                    with res_col1:
                        # 画像表示（結果描画済み）
                        st.image(result_img, caption="分析結果", use_container_width=True)

                    with res_col2:
                        # 最も強い感情を目立たせる
                        dom_emotion = person_data["dominant_jp"]
                        st.metric(label="最も強い感情", value=dom_emotion)

                        # データフレームの作成
                        df = pd.DataFrame(
                            list(person_data["scores"].items()),
                            columns=["感情", "スコア (%)"]
                        )
                        # スコアが高い順に並び替え
                        df = df.sort_values(by="スコア (%)", ascending=False).reset_index(drop=True)

                        # 表を表示
                        st.dataframe(
                            df, 
                            hide_index=True,
                            use_container_width=True,
                            column_config={
                                "スコア (%)": st.column_config.ProgressColumn(
                                    "確信度",
                                    format="%.2f%%",
                                    min_value=0,
                                    max_value=100,
                                )
                            }
                        )
                        
                        # シンプルな棒グラフも表示したい場合（お好みでコメントアウト解除）
                        # st.bar_chart(df.set_index("感情"))