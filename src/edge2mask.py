#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class EdgeToMaskAndFilter:
    def __init__(self):
        self.node_name = "edge_to_mask"
        rospy.init_node(self.node_name)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Parameters
        # 小さいオブジェクトもマスクできるよう小さな値に設定
        self.min_area_ratio = rospy.get_param('~min_area_ratio', 0.0005)  # 0.05%
        # 大きすぎるオブジェクトを除外するための新しいパラメータ
        self.max_area_ratio = rospy.get_param('~max_area_ratio', 0.5)  # 50%
        self.fill_color = 255  # 白で塗りつぶす
        self.debug = rospy.get_param('~debug', True)
        
        # 深度マスクを保存（深度フィルタリングを行う場合）
        self.depth_mask = None
        self.depth_mask_timestamp = None
        
        # パブリッシャー
        self.mask_pub = rospy.Publisher('~output/mask', Image, queue_size=1)
        self.filtered_depth_pub = rospy.Publisher('~output/filtered_depth_mask', Image, queue_size=1)
        
        if self.debug:
            self.debug_pub = rospy.Publisher('~debug/filled_contours', Image, queue_size=1)
            self.debug_depth_pub = rospy.Publisher('~debug/filtered_depth', Image, queue_size=1)
        
        # サブスクライバー
        self.edge_sub = rospy.Subscriber('~input', Image, self.edge_callback, queue_size=1)
        
        # パラメータが設定されている場合のみ深度マスクをサブスクライブ
        apply_to_depth = rospy.get_param('~apply_to_depth', False)
        if apply_to_depth:
            self.depth_mask_sub = rospy.Subscriber('~depth_mask', Image, self.depth_mask_callback, queue_size=1)
        
        rospy.loginfo(f"[{self.node_name}] Initialized with min_area_ratio={self.min_area_ratio}, max_area_ratio={self.max_area_ratio}")
    
    def depth_mask_callback(self, depth_mask_msg):
        """受信した深度マスクを保存"""
        try:
            self.depth_mask = self.bridge.imgmsg_to_cv2(depth_mask_msg, "mono8")
            self.depth_mask_timestamp = depth_mask_msg.header.stamp
        except CvBridgeError as e:
            rospy.logerr(f"[{self.node_name}] Error converting depth mask: {e}")
    
    def edge_callback(self, edge_msg):
        """エッジ画像を処理してマスクを作成し、オプションで深度をフィルタリング"""
        try:
            # ROSの画像をOpenCV形式に変換
            edge_img = self.bridge.imgmsg_to_cv2(edge_msg, "mono8")
            
            # 出力マスク（入力と同じサイズ）を作成
            mask = np.zeros_like(edge_img)
            
            # エッジ画像から輪郭を検出
            contours, hierarchy = cv2.findContours(
                edge_img.copy(), 
                cv2.RETR_EXTERNAL,  # 外部の輪郭のみを取得
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # 面積の閾値を計算
            image_area = edge_img.shape[0] * edge_img.shape[1]
            min_area = self.min_area_ratio * image_area
            max_area = self.max_area_ratio * image_area
            
            # デバッグ用の可視化
            debug_img = None
            if self.debug:
                # 白黒の画像を作成
                debug_img = np.zeros_like(edge_img)
            
            # 各輪郭を処理
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                
                # 最小面積より大きく、最大面積より小さい輪郭のみを処理
                if min_area < area < max_area:
                    # マスクの輪郭内部を塗りつぶす
                    cv2.fillPoly(mask, [contour], self.fill_color)
                    
                    if self.debug:
                        # 白黒表示用に白（255）で塗りつぶす
                        cv2.fillPoly(debug_img, [contour], 255)
                        
                        # 輪郭の境界も白で描画
                        cv2.drawContours(debug_img, [contour], 0, 255, 1)
            
            # マスク画像を公開
            mask_msg = self.bridge.cv2_to_imgmsg(mask, "mono8")
            mask_msg.header = edge_msg.header
            self.mask_pub.publish(mask_msg)
            
            # デバッグ画像を公開（有効な場合）
            if self.debug and debug_img is not None:
                debug_msg = self.bridge.cv2_to_imgmsg(debug_img, "mono8")
                debug_msg.header = edge_msg.header
                self.debug_pub.publish(debug_msg)
                
            # 深度マスクが利用可能な場合はマスクを適用
            if self.depth_mask is not None:
                # 深度マスク画像が同じサイズかチェック
                if mask.shape[:2] == self.depth_mask.shape[:2]:
                    # マスクを組み合わせる（論理積）
                    filtered_depth = cv2.bitwise_and(self.depth_mask, mask)
                    
                    # フィルタリングされた深度マスクを公開
                    filtered_msg = self.bridge.cv2_to_imgmsg(filtered_depth, "mono8")
                    filtered_msg.header.stamp = rospy.Time.now()
                    filtered_msg.header.frame_id = edge_msg.header.frame_id
                    self.filtered_depth_pub.publish(filtered_msg)
                    
                    # デバッグ可視化を公開（有効な場合）
                    if self.debug:
                        # 白黒の可視化画像を作成
                        vis_img = filtered_depth.copy()
                        
                        debug_depth_msg = self.bridge.cv2_to_imgmsg(vis_img, "mono8")
                        debug_depth_msg.header = filtered_msg.header
                        self.debug_depth_pub.publish(debug_depth_msg)
                else:
                    rospy.logwarn(f"[{self.node_name}] 深度マスクのサイズ ({self.depth_mask.shape[:2]}) " +
                                 f"がエッジマスクのサイズと一致しません ({mask.shape[:2]})")
            
            valid_contours = sum(1 for contour in contours if min_area < cv2.contourArea(contour) < max_area)
            rospy.logdebug(f"[{self.node_name}] {len(contours)}個の輪郭を検出、{valid_contours}個が有効、{np.sum(mask > 0)/255}ピクセルを塗りつぶしました")
            
        except CvBridgeError as e:
            rospy.logerr(f"[{self.node_name}] {e}")
        except Exception as e:
            rospy.logerr(f"[{self.node_name}] エラーが発生しました: {e}")

if __name__ == '__main__':
    try:
        node = EdgeToMaskAndFilter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
