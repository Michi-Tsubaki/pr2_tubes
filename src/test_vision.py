import cv2
import numpy as np
import pyrealsense2 as rs

# グローバルパラメータの初期値（トラックバーで調整可能）
# カラーフィルタリングパラメータ
H_MIN = 1
H_MAX = 90
S_MIN = 0
S_MAX = 255
V_MIN = 0
V_MAX = 255

# エッジ検出パラメータ
CANNY_THRESHOLD1 = 50
CANNY_THRESHOLD2 = 150

# ハフ変換パラメータ
MIN_RADIUS = 30
MAX_RADIUS = 100
HOUGH_PARAM1 = 50
HOUGH_PARAM2 = 30

# 針の幅パラメータ
MIN_NEEDLE_WIDTH = 3
MAX_NEEDLE_WIDTH = 15

# 半円の角度パラメータ
START_ANGLE = 0
END_ANGLE = 180
ANGLE_OFFSET = 0  # 半円の回転角度

# ウィンドウ名の定義
WINDOW_CONTROLS = 'Parameter Controls'
WINDOW_RESULT = 'Detected Needle'

def create_trackbars():
    """パラメータ調整用のトラックバーを作成する関数"""
    # トラックバー用のウィンドウを作成
    cv2.namedWindow(WINDOW_CONTROLS, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_CONTROLS, 600, 700)
    
    # HSVフィルタ用のトラックバー
    cv2.createTrackbar('H Min', WINDOW_CONTROLS, H_MIN, 179, lambda x: None)
    cv2.createTrackbar('H Max', WINDOW_CONTROLS, H_MAX, 179, lambda x: None)
    cv2.createTrackbar('S Min', WINDOW_CONTROLS, S_MIN, 255, lambda x: None)
    cv2.createTrackbar('S Max', WINDOW_CONTROLS, S_MAX, 255, lambda x: None)
    cv2.createTrackbar('V Min', WINDOW_CONTROLS, V_MIN, 255, lambda x: None)
    cv2.createTrackbar('V Max', WINDOW_CONTROLS, V_MAX, 255, lambda x: None)
    
    # Cannyエッジ検出用のトラックバー
    cv2.createTrackbar('Canny Threshold1', WINDOW_CONTROLS, CANNY_THRESHOLD1, 255, lambda x: None)
    cv2.createTrackbar('Canny Threshold2', WINDOW_CONTROLS, CANNY_THRESHOLD2, 255, lambda x: None)
    
    # ハフ変換用のトラックバー
    cv2.createTrackbar('Min Radius', WINDOW_CONTROLS, MIN_RADIUS, 200, lambda x: None)
    cv2.createTrackbar('Max Radius', WINDOW_CONTROLS, MAX_RADIUS, 200, lambda x: None)
    cv2.createTrackbar('Hough Param1', WINDOW_CONTROLS, HOUGH_PARAM1, 100, lambda x: None)
    cv2.createTrackbar('Hough Param2', WINDOW_CONTROLS, HOUGH_PARAM2, 100, lambda x: None)
    
    # 針の幅用のトラックバー
    cv2.createTrackbar('Min Needle Width', WINDOW_CONTROLS, MIN_NEEDLE_WIDTH, 50, lambda x: None)
    cv2.createTrackbar('Max Needle Width', WINDOW_CONTROLS, MAX_NEEDLE_WIDTH, 50, lambda x: None)
    
    # 半円の角度用のトラックバー
    cv2.createTrackbar('Start Angle', WINDOW_CONTROLS, START_ANGLE, 360, lambda x: None)
    cv2.createTrackbar('End Angle', WINDOW_CONTROLS, END_ANGLE, 360, lambda x: None)
    cv2.createTrackbar('Angle Offset', WINDOW_CONTROLS, ANGLE_OFFSET, 360, lambda x: None)

def get_trackbar_values():
    """トラックバーから現在の値を取得する関数"""
    values = {}
    
    # HSVフィルタの値を取得
    values['h_min'] = cv2.getTrackbarPos('H Min', WINDOW_CONTROLS)
    values['h_max'] = cv2.getTrackbarPos('H Max', WINDOW_CONTROLS)
    values['s_min'] = cv2.getTrackbarPos('S Min', WINDOW_CONTROLS)
    values['s_max'] = cv2.getTrackbarPos('S Max', WINDOW_CONTROLS)
    values['v_min'] = cv2.getTrackbarPos('V Min', WINDOW_CONTROLS)
    values['v_max'] = cv2.getTrackbarPos('V Max', WINDOW_CONTROLS)
    
    # Cannyエッジ検出の値を取得
    values['canny_threshold1'] = cv2.getTrackbarPos('Canny Threshold1', WINDOW_CONTROLS)
    values['canny_threshold2'] = cv2.getTrackbarPos('Canny Threshold2', WINDOW_CONTROLS)
    
    # ハフ変換の値を取得
    values['min_radius'] = cv2.getTrackbarPos('Min Radius', WINDOW_CONTROLS)
    values['max_radius'] = cv2.getTrackbarPos('Max Radius', WINDOW_CONTROLS)
    values['hough_param1'] = cv2.getTrackbarPos('Hough Param1', WINDOW_CONTROLS)
    values['hough_param2'] = cv2.getTrackbarPos('Hough Param2', WINDOW_CONTROLS)
    
    # 針の幅の値を取得
    values['min_needle_width'] = cv2.getTrackbarPos('Min Needle Width', WINDOW_CONTROLS)
    values['max_needle_width'] = cv2.getTrackbarPos('Max Needle Width', WINDOW_CONTROLS)
    
    # 半円の角度の値を取得
    values['start_angle'] = cv2.getTrackbarPos('Start Angle', WINDOW_CONTROLS)
    values['end_angle'] = cv2.getTrackbarPos('End Angle', WINDOW_CONTROLS)
    values['angle_offset'] = cv2.getTrackbarPos('Angle Offset', WINDOW_CONTROLS)
    
    return values

def detect_and_fill_curved_needle(image, params):
    """
    カラーフィルタリング、エッジ検出、ハフ変換を用いて
    銀色の半円弧針を検出して塗りつぶす関数
    
    Args:
        image: RealSense D405のカラー画像
        params: パラメータの辞書
        
    Returns:
        result_image: 針部分を塗りつぶした画像
        mask: 検出された針の領域（マスク画像）
        debug_images: デバッグ用の中間処理画像の辞書
    """
    # 画像をコピー
    result_image = image.copy()
    height, width = image.shape[:2]
    
    # デバッグ用画像を保存する辞書
    debug_images = {}
    
    # ステップ1: カラーフィルタリングで彩度の低い部分（銀色/グレー）を抽出
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # トラックバーから取得した値でHSVフィルタリング
    lower_gray = np.array([params['h_min'], params['s_min'], params['v_min']])
    upper_gray = np.array([params['h_max'], params['s_max'], params['v_max']])
    
    # マスク作成
    gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
    debug_images['hsv_mask'] = gray_mask
    
    # マスクを適用して銀色部分だけを残す
    gray_filtered = cv2.bitwise_and(image, image, mask=gray_mask)
    debug_images['color_filtered'] = gray_filtered
    
    # グレースケール変換
    gray = cv2.cvtColor(gray_filtered, cv2.COLOR_BGR2GRAY)
    
    # ノイズ除去
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # ステップ2: エッジ検出
    edges = cv2.Canny(blurred, params['canny_threshold1'], params['canny_threshold2'])
    debug_images['edges'] = edges
    
    # モルフォロジー演算で線を強調
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    debug_images['dilated_edges'] = dilated_edges
    
    # ステップ3: ハフ変換で半円を検出
    # 針の検出マスク
    needle_mask = np.zeros((height, width), dtype=np.uint8)
    
    # 最小半径が最大半径より小さいことを確認
    min_radius = min(params['min_radius'], params['max_radius'])
    max_radius = max(params['min_radius'], params['max_radius'])
    
    # パラメータが0の場合は1に設定（0だとエラーになる場合がある）
    hough_param1 = max(1, params['hough_param1'])
    hough_param2 = max(1, params['hough_param2'])
    
    # ハフ変換で円を検出
    circles = cv2.HoughCircles(
        dilated_edges,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=hough_param1,
        param2=hough_param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    # 検出された円がある場合
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        for circle in circles[0, :]:
            x, y, r = circle
            
            # 円の指定された部分をマスクとして使用
            # 開始角度と終了角度をトラックバーから取得
            start_angle = params['start_angle']
            end_angle = params['end_angle']
            angle_offset = params['angle_offset']
            
            # 開始角度と終了角度が異なることを確認
            if start_angle == end_angle:
                end_angle = (start_angle + 180) % 360
            
            cv2.ellipse(
                needle_mask,
                center=(x, y),
                axes=(r, r),
                angle=angle_offset,
                startAngle=start_angle,
                endAngle=end_angle,
                color=255,
                thickness=-1  # 塗りつぶし
            )
            
            # デバッグ用に元の半円マスクを保存
            semicircle_mask = needle_mask.copy()
            debug_images['semicircle_mask'] = semicircle_mask
            
            # 元の画像でエッジが強い部分だけを残す
            # この行をコメントアウトすると半円全体を塗りつぶします
            # needle_mask = cv2.bitwise_and(needle_mask, dilated_edges)
            
            # 針の幅のみを残すための処理（この部分もコメントアウトして半円全体を塗る）
            # contours, _ = cv2.findContours(needle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # for contour in contours:
            #     # 輪郭を囲む最小の矩形を取得
            #     rect = cv2.minAreaRect(contour)
            #     (cx, cy), (width, height), angle = rect
            #     
            #     # 針の幅に近い輪郭を選択
            #     min_dimension = min(width, height)
            #     if params['min_needle_width'] <= min_dimension <= params['max_needle_width']:
            #         cv2.drawContours(needle_mask, [contour], -1, 255, -1)
    
    # モルフォロジー演算で検出領域を調整
    kernel = np.ones((3, 3), np.uint8)
    needle_mask = cv2.morphologyEx(needle_mask, cv2.MORPH_CLOSE, kernel)
    debug_images['needle_mask'] = needle_mask
    
    # 針の塗りつぶし
    result_image[needle_mask == 255] = [0, 0, 255]  # 赤色で塗りつぶし (BGR)
    
    return result_image, needle_mask, debug_images

def main():
    # トラックバーを作成
    create_trackbars()
    
    # RealSense パイプラインの設定
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # パイプラインの開始
    pipeline.start(config)
    
    try:
        while True:
            # フレームを待機
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue
                
            # カラーフレームを画像に変換
            color_image = np.asanyarray(color_frame.get_data())
            
            # トラックバーから現在のパラメータ値を取得
            params = get_trackbar_values()
            
            # 針を検出して塗りつぶす
            result_image, mask, debug_images = detect_and_fill_curved_needle(color_image, params)
            
            # 結果を表示
            cv2.imshow('Original', color_image)
            cv2.imshow('HSV Mask', debug_images['hsv_mask'])
            cv2.imshow('Color Filtered', debug_images['color_filtered'])
            cv2.imshow('Edge Detection', debug_images['edges'])
            cv2.imshow('Dilated Edges', debug_images['dilated_edges'])
            if 'semicircle_mask' in debug_images:
                cv2.imshow('Semicircle Mask', debug_images['semicircle_mask'])
            cv2.imshow(WINDOW_RESULT, result_image)
            cv2.imshow('Needle Mask', mask)
            
            # 現在のパラメータを画面に表示
            param_display = np.zeros((350, 600, 3), dtype=np.uint8)
            cv2.putText(param_display, f"HSV: [{params['h_min']},{params['s_min']},{params['v_min']}] - [{params['h_max']},{params['s_max']},{params['v_max']}]", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(param_display, f"Canny: {params['canny_threshold1']} - {params['canny_threshold2']}", 
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(param_display, f"Hough: Param1={params['hough_param1']}, Param2={params['hough_param2']}", 
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(param_display, f"Radius: {params['min_radius']} - {params['max_radius']}", 
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(param_display, f"Needle Width: {params['min_needle_width']} - {params['max_needle_width']}", 
                        (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(param_display, f"Arc Angle: {params['start_angle']} - {params['end_angle']} (Offset: {params['angle_offset']}°)", 
                        (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(param_display, "Press 's' to save current parameters", 
                        (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(param_display, "Press 'q' to quit", 
                        (10, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Current Parameters', param_display)
            
            # キー入力をチェック
            key = cv2.waitKey(1) & 0xFF
            
            # 's'キーでパラメータを保存
            if key == ord('s'):
                # パラメータをファイルに保存
                with open('needle_detection_params.txt', 'w') as f:
                    for param_name, value in params.items():
                        f.write(f"{param_name} = {value}\n")
                print("Parameters saved to needle_detection_params.txt")
            
            # 'q'キーで終了
            if key == ord('q'):
                break
                
    finally:
        # パイプラインを停止
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()