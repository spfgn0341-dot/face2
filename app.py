import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image
import tempfile
import os

# ページ設定
st.set_page_config(page_title="顔の表情分析アプリ", layout="wide")

st.title("😊 顔の表情分析システム")
st.write("画像をアップロードするか、カメラで撮影して感情を分析します。")

# サイドバーでモード選択
option = st.sidebar.selectbox(
    "入力モードを選択してください",
    ("画像をアップロード", "カメラで撮影")
)

def analyze_emotion(image_np):
    """
    画像を読み込み、DeepFaceで感情分析を行い、
    顔の枠と感情ラベルを描画した画像を返す関数
    """
    try:
        # OpenCV形式の画像コピーを作成
        img_cv = image_np.copy()
        
        # DeepFaceで分析 (backendsはopencvやretinafaceなど選べますが、opencvが最速)
        # enforce_detection=Falseにすると、顔が見つからなくてもエラーにならず処理が進む
        results = DeepFace.analyze(img_cv, actions=['emotion'], enforce_detection=False)
        
        # 結果はリスト形式で返ってくる場合があるため対応
        if not isinstance(results, list):
            results = [results]

        for res in results:
            # 信頼度が低い、または顔領域が極端に小さい場合はスキップする処理を入れても良い
            region = res['region']
            emotion = res['dominant_emotion']
            score = res['emotion'][emotion]

            # 顔の座標
            x, y, w, h = region['x'], region['y'], region['w'], region['h']

            # 矩形を描画
            cv2.rectangle(img_cv, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # テキスト（感情とスコア）を描画
            text = f"{emotion} ({score:.1f}%)"
            cv2.putText(img_cv, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, (36, 255, 12), 2)
        
        return img_cv, results

    except Exception as e:
        st.error(f"分析中にエラーが発生しました: {e}")
        return image_np, []

# 画像入力の処理
input_image = None

if option == "画像をアップロード":
    uploaded_file = st.file_uploader("JPGまたはPNG画像を選択", type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        input_image = np.array(image.convert('RGB'))

elif option == "カメラで撮影":
    camera_image = st.camera_input("カメラで撮影してください")
    if camera_image is not None:
        image = Image.open(camera_image)
        input_image = np.array(image.convert('RGB'))

# 分析実行ボタン
if input_image is not None:
    st.subheader("入力画像")
    st.image(input_image, caption="元画像", use_container_width=True)

    if st.button("表情を分析する"):
        with st.spinner('分析中...（初回はモデルのダウンロードに時間がかかります）'):
            result_img, analysis_data = analyze_emotion(input_image)
            
            # 結果表示カラム
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("分析結果画像")
                st.image(result_img, caption="検出結果", use_container_width=True)
            
            with col2:
                st.subheader("詳細データ")
                if analysis_data:
                    # 1人目のデータのみ詳細表示（複数人対応も可能）
                    data = analysis_data[0]
                    st.write(f"**支配的な感情:** {data['dominant_emotion']}")
                    st.write("**感情スコア:**")
                    st.json(data['emotion'])
                else:
                    st.warning("顔が検出されませんでした。")