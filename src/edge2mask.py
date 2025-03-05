#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

class EdgeToMask:
    def __init__(self):
        self.bridge = CvBridge()
        
        # パブリッシャー設定
        self.mask_pub = rospy.Publisher('~output/mask', Image, queue_size=1)
        
        # サブスクライバー設定
        self.edge_sub = rospy.Subscriber('~input', Image, self.edge_callback, queue_size=1)
        
    def edge_callback(self, edge_msg):
        try:
            # エッジ画像をOpenCVフォーマットに変換
            edge_img = self.bridge.imgmsg_to_cv2(edge_msg, "mono8")
            
            # 輪郭を検出
            contours, hierarchy = cv2.findContours(edge_img, 
                                                  cv2.RETR_EXTERNAL,  # 外部輪郭のみ検出
                                                  cv2.CHAIN_APPROX_SIMPLE)
            
            # マスク画像の作成（黒で初期化）
            mask = np.zeros_like(edge_img)
            
            # 見つかった輪郭内部を白で塗りつぶす
            for contour in contours:
                # 小さすぎる輪郭はノイズとして無視（必要に応じて調整）
                if cv2.contourArea(contour) > 100:  
                    cv2.drawContours(mask, [contour], 0, 255, -1)  # -1は塗りつぶしを意味する
            
            # マスク画像をROSメッセージに変換して発行
            mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
            mask_msg.header = edge_msg.header
            self.mask_pub.publish(mask_msg)
            
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

if __name__ == '__main__':
    rospy.init_node('edge_to_mask')
    node = EdgeToMask()
    rospy.spin()
