import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import pandas as pd
import mediapipe as mp
#from google.colab.patches import cv2_imshow
#from IPython.display import display, Image
from PIL import Image
import matplotlib.pyplot as plt
import io
import os
from pathlib import Path

# ヘッダーセクション
st.title('Welcome to    "AI Strike Vision" !!')

# 注入するカスタム CSS
css = """
<style>
    .stApp {
        background-color: #F0FFF0;
    }
</style>
"""

# スタイルを注入
st.markdown(css, unsafe_allow_html=True)


# 背景
st.image('背景2.png')

st.write("""
このページでは、AIを使用して、ピッチャーが投げたボールがストライクかボールかを判定することができます。
動画をアップロードすると、解析が自動で行われ、結果が出力されます。
サイドバーのサンプル動画をダウンロードして試してみてください。
""")

# サイドバーにタイトルを追加
st.sidebar.title("サンプル動画のダウンロード")

# ダウンロードする動画ファイルのリスト
video_files = ["ストライク1.mp4", "ストライク2.mp4", "ストライク3.mp4","ボール1.mp4","ボール2.mp4"]

# 各動画ファイルに対してダウンロードボタンを追加
for video_file_name in video_files:
    video_file_path = Path(video_file_name)

    # ファイルが存在するかどうかを確認
    if video_file_path.is_file():
        # ファイルを読み込む
        with open(video_file_path, "rb") as file:
            # サイドバーにダウンロードボタンを追加
            st.sidebar.download_button(
                label=f"{video_file_name}をダウンロード",
                data=file,
                file_name=video_file_name,
                mime='video/mp4'
            )
    else:
        st.sidebar.write(f"{video_file_name}が見つかりません。")



# サンプル動画のリンクを追加
# sample_video_url = "ストライク.mp4"  # サンプル動画のURLを指定
# st.sidebar.markdown(f"**[サンプル動画をダウンロードする]({sample_video_url})**")

# 手法の説明
#st.subheader('●手法の説明')
with st.expander("ストライク/ボール判定のアルゴリズムを見る"):
  st.image('説明図2.JPG')


# ファイルアップロードセクション
st.subheader('●判定したい動画をアップロードしてください↓')


# モデルの読み込み
model = YOLO('yolov8n.pt') 

# Input
upload_file = st.file_uploader("動画アップロード", type='mp4')

# 処理済みの動画をStreamlitで表示
st.subheader('↓アップロードされた動画はこちら↓')
st.video(upload_file)





# Process
if upload_file is not None:

  # 解析中のメッセージを表示
    with st.spinner('解析中です...少々お待ちください'):

      # プログレスバーを追加
      progress_bar = st.progress(0)

      temp_file = tempfile.NamedTemporaryFile(delete=False) 
      temp_file.write(upload_file.read())

      cap = cv2.VideoCapture(temp_file.name)
      width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
      height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
      count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
      fps = cap.get(cv2.CAP_PROP_FPS)

      writer = cv2.VideoWriter('./output/ball_object_detection_app_results.mp4',
                                cv2.VideoWriter_fourcc(*'MP4V',),fps,
                                frameSize=(int(width),int(height)))
    
      sports_ball_frames = []  # スポーツボールが検出されたフレームのインデックスを保存するリスト
      sports_ball_detection = []  # スポーツボールが検出されたバウンディングボックスのxyxy

      frame_number = 0  # 現在のフレーム番号

      while True:
          ret, frame = cap.read()
          if not ret:
              break

          # フレームに対して物体検出を行う
          results = model(frame,conf=0.2,classes=[32])
          frame = results[0].plot(labels=True,conf=True)
          categories = results[0].boxes.cls
          # スポーツボール(クラスID 32)が検出されたかチェック
          if 32 in categories:
              sports_ball_frames.append(frame)  # スポーツボールが検出されたフレームをリストに追加
              sports_ball_detection.append(results[0].boxes.xyxy)
              writer.write(frame)  # スポーツボールが検出されたフレームのみを書き込む

          frame_number += 1  # フレーム番号をインクリメント
          # プログレスバーを更新
          progress_bar.progress(frame_number / count)

      cap.release()
      writer.release()

      # プログレスバーを完了状態にする
      progress_bar.progress(1.0)
      # プログレスバーのプレースホルダーを削除
      progress_bar.empty()

      #ボールの左側x
      ball_x_left = sports_ball_detection[-1][0][0]

      #ボールの右側x
      ball_x_right = sports_ball_detection[-1][0][2]

      #ボールの下側y
      ball_y_down = sports_ball_detection[-1][0][3]

      #ボールの上側y
      ball_y_up = sports_ball_detection[-1][0][1]


      # sports_ball_frames[-1] から画像データを取得（ここで配列を仮定）
      image = sports_ball_frames[-1]

      # 画像の右半分を対象領域とするための準備
      h, w, _ = image.shape
      roi = image[:, w//2:w]  # 画像の右半分を抽出

      # MediaPipe Poseのセットアップ
      mp_pose = mp.solutions.pose
      mp_drawing = mp.solutions.drawing_utils
      pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)


      # 対象領域(ROI)でポーズ検出を実行
      results_pipe = pose.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

      # 結果を元の画像に描画
      estimation_image = image.copy()
      if results_pipe.pose_landmarks:
          # 注: 描画座標を調整するために、ROIのオフセットを加味
          mp_drawing.draw_landmarks(
              estimation_image[:, w//2:w],
              results_pipe.pose_landmarks,
              mp_pose.POSE_CONNECTIONS,
              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
          )

      #左ひざの位置
      left_knee = results_pipe.pose_landmarks.landmark[25]
      left_knee_y = left_knee.y * height

      #左腰と左肩の位置
      left_hip_y = results_pipe.pose_landmarks.landmark[23].y
      left_shoulder_y = results_pipe.pose_landmarks.landmark[11].y

      #ストライクゾーンの上側
      strike_zone_upper = (left_hip_y + left_shoulder_y) * 0.5
      strike_zone_upper = strike_zone_upper * height



      image_array = cv2.cvtColor(estimation_image, cv2.COLOR_BGR2RGB) 
      

      # ホームベースが白色であることを利用して探す関数を定義
      def is_white(pixel, threshold=200):
          # RGB値が全て閾値以上であれば白色とみなす
          return all(channel >= threshold for channel in pixel)

      # ホームベースの予想されるX座標範囲を指定
      expected_x_min = int(0.39 * width)  #250
      expected_x_max = int(0.62 * width)   #400

      # ホームベースが存在しそうな縦の範囲を指定
      expected_y_min = int(image_array.shape[0] * 0.80)
      expected_y_max = int(image_array.shape[0] * 0.95)

      # 白いピクセルを検出してリストに保存
      white_pixels = []

      # 指定範囲内で白いピクセルを探す
      for y in range(expected_y_min, expected_y_max):
          for x in range(expected_x_min, expected_x_max):
              if is_white(image_array[y, x]):
                  white_pixels.append((x, y))

      # 白いピクセルが見つかればそのX座標をホームベースの左右のX座標として推定
      if white_pixels:
          white_x_coords = [x for x, y in white_pixels]
          home_plate_left_x = min(white_x_coords)
          home_plate_right_x = max(white_x_coords)
      else:
          home_plate_left_x = None
          home_plate_right_x = None

      # 結果を画像上に可視化する
      plt.figure(figsize=(10, 8))
      plt.imshow(image_array)

      # X座標が見つかった場合に線を描画
      if home_plate_left_x is not None and home_plate_right_x is not None:
          plt.axvline(x=home_plate_left_x, color='red', linestyle='--')
          plt.axvline(x=home_plate_right_x, color='red', linestyle='--')

      # 探索範囲を示す水平線を描画
      plt.axhline(y=strike_zone_upper, color='red', linestyle='--')
      plt.axhline(y=left_knee_y, color='red', linestyle='--')

      plt.axis('off')

      # 結果の画像をファイルに保存する
      output_path = './output/judgement_stream.png'  # 保存するパスを指定
      plt.savefig(output_path, bbox_inches='tight')  # 余白を削除して保存

      plt.show()

    st.success('解析完了！')
    st.subheader('・・・解析結果はこちら↓')
    st.image(output_path, caption='判定結果')

# Output
    #ボールとストライクゾーンの位置の比較
    #if home_plate_left_x < ball_x < home_plate_right_x and strike_zone_upper< ball_y < left_knee_y: ボールの中心で判定する場合
    if home_plate_left_x < ball_x_right and ball_x_left < home_plate_right_x and strike_zone_upper < ball_y_down and ball_y_up < left_knee_y:
        st.subheader('判定は・・・ストライク！！')
        st.image('ストライク.png')
    else:
        st.subheader('判定は・・・ボール！！')
        st.image('ボール.png')

 # フッター
st.write('© 2024 AI Strike Vision')   





    