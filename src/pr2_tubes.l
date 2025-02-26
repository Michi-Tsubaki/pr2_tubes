#!/usr/bin/env roseus

#| DESCRIPTION
PR2で血管モデルを縫合する．
20250219 血管の縫い方
20250221 曲率を持った軌跡の作成 - 片方の針を血管内部に刺すことに成功
20250224 持ち替え
20250225 完成しているコードには英語でコメントを個人的TODOやissueには日本語でコメントを書く
|#

#| Review tips for building model in euslisp (@Michi-Tsubaki)
- euslispのscriptで作れる基本的な幾何構造
  https://euslisp.github.io/jskeus/jmanual-node118.html
- coords
  右手座標系 red-> x, green-> y, blue-> z
|#

;; for debug view
;; (send (send *left-tube* :copy-worldcoords) :draw-on :flush t :size 100)
;; (send *left-tube-rc* :draw-on :flush t :size 100)
;; (send (send *right-tube* :copy-worldcoords) :draw-on :flush t :size 100)
;; (send *right-tube-lc* :draw-on :flush t :size 100)


;; Import PR2 interface
(require "package://pr2eus/pr2.l")
(require "package://pr2eus/pr2-utils.l")
(require "package://pr2eus/pr2-interface.l") ;; *ri*
(require "package://pr2eus/speak.l") ;; pkg for speaking

;; Import curved trajectory
;;(load "trajectory.l")

;; Load jsk msg type
(ros::load-ros-package "jsk_recognition_msgs")
(ros::roseus-add-msgs "speech_recognition_msgs")
(ros::roseus-add-msgs "geometry_msgs")

;; Initialize PR2
(if (not (boundp '*pr2*)) (pr2-init)) ;; if not *pr2*, initialize and setq *pr2*

;; Import curved trajectory class
(load "trajectory.l")

;; "Center position of experiment desk"
(setq *centerx* 700)
(setq *centery* 0)
(setq *centerz* 723)
(setq *center* (make-cube 10 10 10))
(send *center* :translate (float-vector *centerx* *centery* *centerz*))
(send *center* :set-color :black)
(setq *o* (send *center* :copy-worldcoords))

;; Desk configuration
(setq *deskw* 500)
(setq *deskd* 800)
(setq *deskh* *centerz*)
(setq *desk* (make-cube *deskw* *deskd* *deskh*))
(send *desk* :translate (float-vector *centerx* *centery* (* *centerz* 0.5)))
(send *desk* :set-color :white)

;; Table (on which blood vessels are set) configuration
(setq *tablew* 50)
(setq *tabled* 80)
(setq *tableh* 10)
(setq *table* (make-cube *tablew* *tabled* *tableh*))
(send *table* :translate (float-vector (- *centerx* 200) *centery* (+ *centerz* (* *tableh* 0.5)) ))
(send *table* :set-color :red)

;; Set the bend needle model
(setq *needle-radius* 10)
(setq *needle-thickness* 1)
(setq *needle-angle* pi)
(setq segments 40)
;;(load "needle.l")
(setq theta-step (/ *needle-angle* segments))
(setq points nil)
(dotimes (i (+ segments 1))
  (let ((theta (* i theta-step)))
    (push (float-vector 
           (* *needle-radius* (- 1 (cos theta)))  ; x
           0                                      ; y
           (* *needle-radius* (sin theta)))       ; z
          points)))
(setq points (reverse points))

;; Function to calculate rotation matrix from two points
(defun make-rotation-matrix (p1 p2)
  (let* ((diff (v- p2 p1))
         (z-axis (normalize-vector diff))
         (y-axis (float-vector 0 1 0))
         (x-axis (v* y-axis z-axis)))
    (setq y-axis (v* z-axis x-axis))  ; ensure orthogonality
    (transpose (matrix (normalize-vector x-axis) 
                      (normalize-vector y-axis) 
                      (normalize-vector z-axis)))))

;; Create cylinders along the arc
(setq needle-parts nil)
(let ((prev-point (car points)))
  (dolist (point (cdr points))
    (let* ((diff (v- point prev-point))
           (height (norm diff))
           (center (midpoint 0.5 prev-point point))
           (cylinder (make-cylinder *needle-thickness* height :segments 12))
           (rot-mat (make-rotation-matrix prev-point point)))
      ;; Create coordinates at center with proper orientation
      (setq coords (make-coords :pos center :rot rot-mat))
      ;; Orient and position cylinder
      (send cylinder :transform coords)
      (push cylinder needle-parts)
      (setq prev-point point))))

;; Create the bodyset
(setq needle-bodies (reverse needle-parts))

;; Associate all parts with the first one
(dolist (body (cdr needle-bodies))
  (send (car needle-bodies) :assoc body))

;; Set color
(dolist (body needle-bodies)
  (send body :set-color :yellow))

;; Define needle as bodyset-link
(setq *needle* 
      (instance bodyset-link :init (make-cascoords)
                :bodies needle-bodies))

;; Make a bend needle instance 
(setq *needle* (instance bodyset-link :init (make-cascoords) :bodies needle-bodies))
(send *needle* :translate (float-vector (- *centerx* 200) *centery* (+ *centerz* 40)))
(send *needle* :rotate #d180 :y)
(send *needle* :rotate #d90 :z)

;; Left blood vessel
(setq *left-tube* (make-cylinder 2.5 20))
(send *left-tube* :translate (float-vector (- *centerx* 200) *centery* (+ *centerz* 10)))
(send *left-tube* :rotate #d90 :x)
(send *left-tube* :translate (float-vector 0 0 -21))
(send *left-tube* :set-color :pink)
(setq *left-tube-rc* (send (send *left-tube* :copy-worldcoords) :translate (float-vector 0 0 20)))

;; Right blood vessel
(setq *right-tube* (make-cylinder 2.5 20))
(send *right-tube* :translate (float-vector (- *centerx* 200) *centery* (+ *centerz* 10)))
(send *right-tube* :rotate #d90 :x)
(send *right-tube* :translate (float-vector 0 0 1))
(send *right-tube* :set-color :pink)
(setq *right-tube-lc* (send (send *right-tube* :copy-worldcoords) :translate (float-vector 0 0 0)))

;; Robot Configuration
;; Set finger configuration
(setq *finger-offset* 45)

;; Set finger extention
(setq *finger-thickness* 8)
(setq *finger-width* 10)
(setq *finger-length* *finger-offset*)

;; Make left finger extension
(setq *left-finger-extension* (make-cube *finger-length* *finger-width* *finger-thickness*))
(send *left-finger-extension* :set-color :green) ;; Set color
(setq *left-finger-coords* (send (send *pr2* :larm :end-coords) :copy-worldcoords)) ;; *left-finger-coords* is the TMP coords to set finger extension
(send *left-finger-coords* :translate (float-vector (/ *finger-length* 2) 0 0))  ;; Offset
(send *left-finger-coords* :rotate (deg2rad -90) :x)  ;; Offset(angle)
(send *left-finger-extension* :move-to *left-finger-coords* :world)
(send (send *pr2* :larm :end-coords) :assoc *left-finger-extension*)
;; ** This is the coords which is used  representative coords! ** 
(setq *left-finger-tip* (make-cascoords))
(send *left-finger-tip* :transform (send *left-finger-extension* :copy-worldcoords))
(send *left-finger-tip* :translate (float-vector (/ *finger-length* 2) 0 0))
(send *left-finger-tip* :rotate #d90 :x)
(send (send *pr2* :larm :end-coords) :assoc *left-finger-tip*)

;; Make right finger extension
(setq *right-finger-extension* (make-cube *finger-length* *finger-width* *finger-thickness*))
(send *right-finger-extension* :set-color :green) ;; Set color
(setq *right-finger-coords* (send (send *pr2* :rarm :end-coords) :copy-worldcoords))
(send *right-finger-coords* :translate (float-vector (/ *finger-length* 2) 0 0))  ;; Offset
(send *right-finger-coords* :rotate (deg2rad -90) :x)  ;; Offset (angle)
(send *right-finger-extension* :move-to *right-finger-coords* :world)
(send (send *pr2* :rarm :end-coords) :assoc *right-finger-extension*)
;; ** This is the coords which is used as right arm's representative coords!
(setq *right-finger-tip* (make-cascoords))
(send *right-finger-tip* :transform (send *right-finger-extension* :copy-worldcoords))
(send *right-finger-tip* :translate (float-vector (/ *finger-length* 2) 0 0))
(send (send *pr2* :rarm :end-coords) :assoc *right-finger-tip*)

;; Update objects list for draw
(objects (list *pr2* *center* *desk* *table* *needle* *left-tube* *right-tube* *left-finger-extension* *right-finger-extension*))
(send *irtviewer* :draw-objects)
(ros::duration-sleep 1)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Trajectory Planning
(defclass traj
  :super cascaded-coords
  :slots (points))

(defmethod traj
  (:init (&rest args)
	 (send-super* :init args)
	 (dotimes (i *traj_num*)
	   (push
	    (make-cube 10 10 10 :pos (float-vector (* (- i 5) *interval*) 0 0))
	    points))
	 (dotimes (i (- (length points) 1))
	   (send (elt points i) :assoc (elt points (+ 1 i))))
	 (send self :assoc (car points))
	 self)
  (:points () points)
  )


;; Define bend trajectories.
(defclass curved-traj
  :super cascaded-coords
  :slots (points))
(defmethod curved-traj
  (:init (&key (radius 30.0) (start-angle -10) (end-angle 10)) ;; 初期値は適当..
         (send-super :init)
         (let* ((angle-step (/ (- end-angle start-angle) (1- *traj_num*)))
                (current-angle start-angle))
           ;; generate points
           (setq points nil)  ;; initialize points
           (dotimes (i *traj_num*)
             (let* ((theta (deg2rad current-angle))
		    (x (* radius (sin theta)))
		    (z (* radius (cos theta)))
		    (point (make-cube 2 2 2 :pos (float-vector x 0 z)))
		    (tangent-x (* radius (cos theta)))
		    (tangent-z (* -1 radius (sin theta)))
		    (tangent-angle (rad2deg (atan tangent-x tangent-z))))
	       (send point :set-color :black)
	       (send point :orient (deg2rad tangent-angle) :y)
               (push point points)
               (setq current-angle (+ current-angle angle-step))))
           (setq points (reverse points)))
         ;; assoc
         (dotimes (i (1- (length points)))
           (send (elt points i) :assoc (elt points (1+ i))))
         (send self :assoc (car points))
         self)
  (:points () points)
  )

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Main Function
;; Define needle coords system
(defun set-needle-tip ()
  (setq *needle-tip* (make-cascoords 
		      :pos (send (car (last needle-bodies)) :worldpos)))
  (send *needle-tip* :rotate #d90 :y)
  (send *needle-tip* :rotate #d180 :x)
  (send *needle-tip* :rotate #d270 :z)
  (send *needle* :assoc *needle-tip*) ;; Use needle-tip as the main coords of the needle. Do not edit it (Feb 25, Michi-Tsubaki)
  )

;; Define suture function
(defun suture (trajectory time-interval)
  (set-needle-tip)
  (let (
        (tm time-interval)
        (path (list))
        (time-vec (list 10000))
        )
    ;; ** assoc **
    (send *left-finger-tip* :assoc *needle-tip*)    
    ;; Solve inverse problem using needle-tip coords
    (dolist (e (send trajectory :points))
      (setq ee (send e :copy-worldcoords))
      (send ee :translate (float-vector 0 0 0) :local)
      (send ee :rotate #d270 :y :local)
      (send ee :rotate #d90 :x :local)
      (send ee :translate (float-vector 0 0 0))
      (send ee :draw-on :flush t :size 100)
      (push (print (send *pr2* :larm :inverse-kinematics ee 
			 :move-target *needle-tip*
			 :link-list (send *pr2* :link-list 
					  (send *pr2* :larm :end-coords :parent))
			 :use-torso t 
			 :rotation-axis t)) 
            path)
      (push tm time-vec)
      )
    (pop time-vec)
    (nreverse time-vec)
    ;; (nreverse path) ;; しなくていいみたいだけど，なんで？ゼミのときはした．
    (dolist (i path)
      (send *pr2* :angle-vector i)
      (send *irtviewer* :draw-objects)
      (unix:usleep 100000))
    (send *ri* :angle-vector-sequence path time-vec)
          ;;  :default-controller 0.001
          ;; :min-time 0.0001
          ;; :minjerk-interpolation t)

    (send *ri* :wait-interpolation)    
    ;; Dissoc needle-tip coords ;; Do not forget!
    (send *left-finger-tip* :dissoc *needle-tip*)
    (send *pr2* :angle-vector (car (last path)))
    (send *irtviewer* :draw-objects)
    ) ;; let
  ) ;; defun

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Main
;; Set init pose
(setq *voice-mode* nil)
(if *voice-mode* (send *ri* :speak-jp "初期姿勢に戻ります" :wait t))
(send *ri* :stop-grasp)
(send *pr2* :reset-pose)
(send *pr2* :torso :waist-z :joint-angle 300) ;; Torso up to avoid ik failure.
(send *ri* :angle-vector (send *pr2* :angle-vector) 10000)
(send *ri* :wait-interpolation)
(send *irtviewer* :draw-objects)
(ros::duration-sleep 1)

;; Grab needle in left hand in the viewer
(setq *larm-coords* (send *pr2* :larm :end-coords :copy-worldcoords))
(send *larm-coords* :rotate #d180 :y)
(send *larm-coords* :rotate #d180 :z)
(send *larm-coords* :translate (float-vector (- *finger-offset* 12) 0 -8))
(send *needle* :move-to *larm-coords* :world)
(send (send *pr2* :larm :end-coords) :assoc *needle*)
(send *irtviewer* :draw-objects)

;; Robot actually grab needle by human support
(if *voice-mode* (send *ri* :speak-jp "針を持たせてください" :wait t))
(send *ri* :stop-grasp :rarm)
(send *ri* :stop-grasp :larm)
(send *ri* :wait-interpolation)
(ros::duration-sleep 2)
(if *voice-mode* (send *ri* :speak-jp "閉じます．注意して下さい．" :wait t))
(send *ri* :start-grasp :larm)
(send *ri* :wait-interpolation)
(ros::duration-sleep 1)
;; TODO: FVが実装されたらここで正しく持てているかを確認して必要に応じて繰り返す.
;; lerobotのはじめの例みたいな感じで逆側のアームでつついて正しい姿勢にするみたいな？
;; パターンマッチングとかしてみたい
(send *irtviewer* :draw-objects)

;; Suturing
;; Go to wait position
(setq *target-coords* (send *left-tube-rc* :copy-worldcoords))
(send *target-coords* :rotate #d90 :x) 
(send *target-coords* :rotate #d90 :z)
(send *target-coords* :rotate #d270 :y)
(send *target-coords* :rotate #d180 :x)
(send *target-coords* :translate #f(-80 0 0))
(send *pr2* :larm :inverse-kinematics *target-coords* :rotation-axis t)
(send *ri* :angle-vector (send *pr2* :angle-vector) 5000)
(send *ri* :wait-interpolation)
(send *irtviewer* :draw-objects)

;; Make trajectories
(setq *interval* 0.3)
(setq *traj_num* 8)
(setq *traj_len* (* *interval* *traj_num*))
(setq *curved-path* (instance curved-traj :init))
(send *curved-path* :translate (send *left-tube-rc* :worldpos))
;; TODO: translateする先の位置(*left-tube-rc*)をDepth cameraで取得したい (Visionタスク)
;;        エイプリルタグ + d405とか
(send *curved-path* :translate #f(0 0 32))
(send *curved-path* :rotate #d90 :z)
(send *curved-path* :rotate #d180 :x)
(send *irtviewer* :draw-objects)

;; Append *curved-path* to object list
(objects (append (list *pr2* *center* *desk* *table* *needle* *left-tube* *right-tube* *left-finger-extension* *right-finger-extension*) (send *curved-path* :points)))
(send *irtviewer* :draw-objects)

;; Suture!
(ros::duration-sleep 1)
(suture *curved-path* 3000)
(ros::duration-sleep 1)

(send *pr2* :larm :move-end-pos #f(0 0 10) :world)
(send *ri* :angle-vector (send *pr2* :angle-vector) 1000)
(send *ri* :wait-interpolation)
(send *irtviewer* :draw-objects)
(ros::duration-sleep 1)


;; TODO: 右手で掴みに行く (now in construction)
;; ikで :use-torso nil にしてもtorsoが動いてしまうので，torsoを保存することにする
;; Rarm goes to wait position
(setq *larm-angles* (send *pr2* :larm :angle-vector))
(setq *torso-angle* (send *pr2* :torso :angle-vector))
(setq *pre-grasp-coords* (send *needle-tip* :copy-worldcoords))
(send *pre-grasp-coords* :translate #f(-50 10 0))
(send *pr2* :rarm :inverse-kinematics *pre-grasp-coords* 
      :move-target *right-finger-tip*
      :link-list (send *pr2* :link-list (send *pr2* :rarm :end-coords :parent))
      :rotation-axis t)
(send *pr2* :larm :angle-vector *larm-angles*) ;; Recover torso angle
(send *pr2* :torso :angle-vector *torso-angle*)
(send *ri* :angle-vector (send *pr2* :angle-vector) 6000)
(send *ri* :wait-interpolation)
(send *irtviewer* :draw-objects)
(ros::duration-sleep 1)

;; Approaching
(setq *grasp-coords* (send *needle-tip* :copy-worldcoords))
(send *grasp-coords* :translate #f(0 22 0))
(send *pr2* :rarm :inverse-kinematics *grasp-coords*
      :move-target *right-finger-tip*
      :link-list (send *pr2* :link-list (send *pr2* :rarm :end-coords :parent))
      :rotation-axis t)
(send *ri* :angle-vector (send *pr2* :angle-vector) 5000)
(send *ri* :wait-interpolation)
(send *irtviewer* :draw-objects)
(ros::duration-sleep 1)

;; Exchange needle L2R
(send *ri* :start-grasp :rarm :wait t)
(send *ri* :wait-interpolation)
(ros::duration-sleep 2)
(send *ri* :stop-grasp :larm :wait t)
(send *ri* :wait-interpolation)
(send (send *pr2* :larm :end-coords) :dissoc *needle*)
(send (send *pr2* :rarm :end-coords) :assoc *needle*)
(send *ri* :wait-interpolation)
(ros::duration-sleep 1)


;; Larm moves a little distance
(send *pr2* :larm :move-end-pos #f(0 0 30) :world)
(send *ri* :angle-vector (send *pr2* :angle-vector) 1000)
(send *ri* :wait-interpolation)
(send *irtviewer* :draw-objects)
(ros::duration-sleep 1)

;; Draw a needle from tube
(setq *post-grasp-coords* (send *grasp-coords* :copy-worldcoords))
(send *post-grasp-coords* :translate #f(-10 0 0))
(send *pr2* :rarm :inverse-kinematics *post-grasp-coords*
      :move-target *right-finger-tip*
      :link-list (send *pr2* :link-list (send *pr2* :rarm :end-coords :parent))
      :rotation-axis t)
(send *ri* :angle-vector (send *pr2* :angle-vector) 10000)
(send *ri* :wait-interpolation)
(send *irtviewer* :draw-objects)
