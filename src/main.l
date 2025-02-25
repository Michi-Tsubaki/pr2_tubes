#!/usr/bin/env roseus

#| DESCRIPTION
PR2で血管モデルを縫合する．
20250219 血管の縫い方
20250221 曲率を持った軌跡の作成 - 片方の針を血管内部に刺すことに成功
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
(load "trajectory.l")

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
(setq *centerz* 780)
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
(load "needle.l")

;; Make a bend needle instance 
(setq *needle* 
      (instance bodyset-link :init (make-cascoords)
                :bodies needle-bodies))

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
(setq *left-finger-coords* (send (send *pr2* :larm :end-coords) :copy-worldcoords))
(send *left-finger-coords* :translate (float-vector (/ *finger-length* 2) 0 0))  ;; Offset
(send *left-finger-coords* :rotate (deg2rad -90) :x)  ;; Offset(angle)
(send *left-finger-extension* :move-to *left-finger-coords* :world)
(send (send *pr2* :larm :end-coords) :assoc *left-finger-extension*)

;; Make right finger extension
(setq *right-finger-extension* (make-cube *finger-length* *finger-width* *finger-thickness*))
(send *right-finger-extension* :set-color :green) ;; Set color
(setq *right-finger-coords* (send (send *pr2* :rarm :end-coords) :copy-worldcoords))
(send *right-finger-coords* :translate (float-vector (/ *finger-length* 2) 0 0))  ;; Offset
(send *right-finger-coords* :rotate (deg2rad -90) :x)  ;; Offset (angle)
(send *right-finger-extension* :move-to *right-finger-coords* :world)
(send (send *pr2* :rarm :end-coords) :assoc *right-finger-extension*)

;; Update objects list for draw
(objects (list *pr2* *center* *desk* *table* *needle* *left-tube* *right-tube* *left-finger-extension* *right-finger-extension*))
(send *irtviewer* :draw-objects)
(ros::duration-sleep 1)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; main

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

;; Grab needle in left hand
;; irtviewer上で針をロボットのところに移動している
(setq *larm-coords* (send *pr2* :larm :end-coords :copy-worldcoords))
(send *larm-coords* :rotate #d180 :y)
(send *larm-coords* :rotate #d180 :z)
(send *larm-coords* :translate (float-vector (- *finger-offset* 12) 0 -8))
(send *needle* :move-to *larm-coords* :world)
(send (send *pr2* :larm :end-coords) :assoc *needle*)
(send *irtviewer* :draw-objects)

;; Robot actually grab needle by human support
(if *voice-mode* (send *ri* :speak-jp "針を持たせてください" :wait t))
(send *ri* :stop-grasp :larm)
(send *ri* :wait-interpolation)
(ros::duration-sleep 2)
(send *ri* :start-grasp :larm)
(send *ri* :wait-interpolation)
;; TODO: FVが実装されたらここで正しく持てているかを確認して必要に応じて繰り返す.
;; lerobotのはじめの例みたいな感じで逆側のアームでつついて正しい姿勢にするみたいな？
;; パターンマッチング
(send *irtviewer* :draw-objects)

;; Suturing
;; Go to wait position
(setq *target-coords* (send *left-tube-rc* :copy-worldcoords))
(send *target-coords* :rotate #d90 :x) 
(send *target-coords* :rotate #d90 :z)
(send *target-coords* :rotate #d270 :y)
(send *target-coords* :rotate #d180 :x)
(send *target-coords* :translate #f(-30 0 0))
(send *pr2* :larm :inverse-kinematics *target-coords* :rotation-axis t)
(send *ri* :angle-vector (send *pr2* :angle-vector) 5000)
(send *ri* :wait-interpolation)
(send *irtviewer* :draw-objects)

;; Make trajectories
(setq *interval* 0.7)
(setq *traj_num* 10)
(setq *traj_len* (* *interval* *traj_num*))
(setq *curved-path* (instance curved-traj :init))
(send *curved-path* :translate (send *left-tube-rc* :worldpos))
;; TODO: translateする先の位置(*left-tube-rc*)をDepth cameraで取得したい(Vision)
(send *curved-path* :translate #f(0 0 33))
(send *curved-path* :rotate #d90 :z)
(send *curved-path* :rotate #d180 :x)
(send *irtviewer* :draw-objects)

;; Append *curved-path* to object list
(objects (append (list *pr2* *center* *desk* *table* *needle* *left-tube* *right-tube* *left-finger-extension* *right-finger-extension*) (send *curved-path* :points)))
(send *irtviewer* :draw-objects)


;; 左手手先座標系の作成
(setq *left-finger-tip* (make-cascoords))
(send *left-finger-tip* :transform (send *left-finger-extension* :copy-worldcoords))
(send *left-finger-tip* :translate (float-vector (/ *finger-length* 2) 0 0))
(send *left-finger-tip* :rotate #d90 :x)
(send (send *pr2* :larm :end-coords) :assoc *left-finger-tip*)

;; 針先座標系の設定
(setq *needle-tip* (make-cascoords 
                    :pos (send (car (last needle-bodies)) :worldpos)))
(send *needle-tip* :rotate #d90 :y)
(send *needle-tip* :rotate #d180 :x)
(send *needle-tip* :rotate #d270 :z)
;; 針本体に針先座標系をassoc
(send *needle* :assoc *needle-tip*)


;; 縫合実行関数

;; 縫合実行関数
(defun suture (trajectory time-interval)
  (let (
        (tm time-interval)
        (path (list))
        (time-vec (list 4000))
        )
    ;; 針先の座標系をleft-finger-tipにassoc
    (send *left-finger-tip* :assoc *needle-tip*)
    
    ;; 軌道に沿ってIKを解く
    (dolist (e (send trajectory :points))
      (setq ee (send e :copy-worldcoords))
      (send ee :translate (float-vector 0 0 0) :local)
      (send ee :rotate #d270 :y :local)
      (send ee :rotate #d90 :x :local)
      (send ee :translate (float-vector 0 0 0))
      (send ee :draw-on :flush t :size 100)
      
      ;; IKを解く（針先をターゲットとして指定）
      (push (print (send *pr2* :larm :inverse-kinematics ee 
			 :move-target *needle-tip*  ;; 針先を移動対象に
			 :link-list (send *pr2* :link-list 
					  (send *pr2* :larm :end-coords :parent))
			 :use-torso t 
			 :rotation-axis t)) 
            path)
      (push tm time-vec)
      )

    ;; 経路を実行
    (pop time-vec)
    (nreverse time-vec)
    ;;(nreverse path)

    (dolist (i path)
      (send *pr2* :angle-vector i)
      (send *irtviewer* :draw-objects)
      (unix:usleep 100000))


    (send *ri* :angle-vector-sequence path time-vec
          :default-controller 0.001
          :min-time 0.0001
          :minjerk-interpolation t)
    (send *ri* :wait-interpolation)
    
    ;; 針先のassocを解除
    (send *left-finger-tip* :dissoc *needle-tip*)
    (send *pr2* :angle-vector (car (last path)))
    (send *irtviewer* :draw-objects)
    ) ;; let
  ) ;; defun

(suture *curved-path* 4000)

;; TODO: 引き出し (vision, 触覚)

;; TODO: 持ち替え
