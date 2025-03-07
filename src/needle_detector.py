#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class NeedleDetector:
    def __init__(self):
        self.node_name = "needle_detector"
        rospy.init_node(self.node_name)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Parameters
        self.debug = rospy.get_param('~debug', True)
        
        # Publishers
        self.mask_pub = rospy.Publisher('~output/mask', Image, queue_size=1)
        if self.debug:
            self.debug_pub = rospy.Publisher('~debug/detection', Image, queue_size=1)
        
        # Subscribers
        self.image_sub = rospy.Subscriber('~input', Image, self.image_callback, queue_size=1)
        
        rospy.loginfo(f"[{self.node_name}] Initialized")
    
    def image_callback(self, image_msg):
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "mono8")
            
            # Create a copy for debugging visualization
            debug_img = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR) if self.debug else None
            
            # Create output mask (same size as input)
            mask = np.zeros_like(cv_image)
            
            # Method 1: Try to close gaps with morphological operations
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(cv_image, kernel, iterations=2)
            closed = cv2.erode(dilated, kernel, iterations=1)
            
            # Method 2: Detect arcs with Hough circles
            circles = cv2.HoughCircles(
                cv_image, 
                cv2.HOUGH_GRADIENT, 
                dp=1, 
                minDist=20, 
                param1=50, 
                param2=15,  # Lower threshold to detect partial circles
                minRadius=10, 
                maxRadius=100
            )
            
            # Create the mask based on morphological operations
            # Find contours in the closed image
            contours, _ = cv2.findContours(
                closed.copy(), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Process each contour
            for contour in contours:
                # Get contour properties
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                # Calculate shape properties
                # For a needle, we expect a high perimeter-to-area ratio
                if area > 0:
                    ratio = perimeter * perimeter / (4 * np.pi * area)
                    
                    # Needles typically have high ratio values (not circular)
                    if ratio > 5:  # Adjust this threshold as needed
                        cv2.drawContours(mask, [contour], 0, 255, -1)
                        if debug_img is not None:
                            cv2.drawContours(debug_img, [contour], 0, (0, 255, 0), 2)
            
            # Add detected circles to the mask
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :]:
                    center = (circle[0], circle[1])
                    radius = circle[2]
                    
                    # Draw circle on mask
                    cv2.circle(mask, center, radius, 255, -1)
                    
                    if debug_img is not None:
                        # Draw the center of the circle
                        cv2.circle(debug_img, center, 2, (0, 0, 255), 3)
                        # Draw the outline of the circle
                        cv2.circle(debug_img, center, radius, (255, 0, 0), 3)
            
            # Publish mask image
            mask_msg = self.bridge.cv2_to_imgmsg(mask, "mono8")
            mask_msg.header = image_msg.header
            self.mask_pub.publish(mask_msg)
            
            # Publish debug image if enabled
            if self.debug and debug_img is not None:
                debug_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
                debug_msg.header = image_msg.header
                self.debug_pub.publish(debug_msg)
            
        except CvBridgeError as e:
            rospy.logerr(f"[{self.node_name}] {e}")
        except Exception as e:
            rospy.logerr(f"[{self.node_name}] Error processing image: {e}")

if __name__ == '__main__':
    try:
        node = NeedleDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
