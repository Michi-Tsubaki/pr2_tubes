#!/usr/bin/env roseus

#| DESCRIPTION
PR2で血管モデルを縫合する．

20250219 血管の縫い方
|#

;; PR2のインタフェースをロードする
(require "package://pr2eus/pr2.l")
(require "package://pr2eus/pr2-utils.l")
(require "package://pr2eus/pr2-interface.l") ;;*ri*
(require "package://pr2eus/speak.l") ;;pkg for speaking

;; jskのメッセージ型をロードする
(ros::load-ros-package "jsk_recognition_msgs")
(ros::roseus-add-msgs "speech_recognition_msgs")
(ros::roseus-add-msgs "geometry_msgs")

;; PR2の初期化
(if (not (boundp '*pr2*)) (pr2-init)) ;; 変数*pr2*が存在しなければ(pr2-init)を実行する

;; 指の長さのオフセットを決める
;; 指を
(setq *finger-offset* 80)

;; 実験環境の中心の設定
(setq *centerx* 700)
(setq *centery* 0)
(setq *centerz* 755)
(setq *center* (make-cube 10 10 10))
(send *center* :translate (float-vector *centerx* *centery* *centerz*))
(send *center* :set-color :black)
(setq *o* (send *center* :copy-worldcoords))

;; 実験机の設定
(setq *deskw* 500)
(setq *deskd* 800)
(setq *deskh* *centerz*)
(setq *desk* (make-cube *deskw* *deskd* *deskh*))
(send *desk* :translate (float-vector *centerx* *centery* (* *centerz* 0.5)))
(send *desk* :set-color :white)

;; 血管実験台
(setq *tablew* 50)
(setq *tabled* 80)
(setq *tableh* 10)
(setq *table* (make-cube *tablew* *tabled* *tableh*))
(send *table* :translate (float-vector (- *centerx* 200) *centery* (+ *centerz* (* *tableh* 0.5)) ))
(send *table* :set-color :red)

;; ベンド針モデル (コア技術) ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(setq *needle-radius* 10)
(setq *needle-thickness* 1)
(setq *needle-angle* pi)

;; 針を構成する微小片を作る
(setq segments 40)
(setq theta-step (/ *needle-angle* segments))
(setq points nil)
(dotimes (i (+ segments 1))
  (let ((theta (* i theta-step)))
    (push (float-vector 
           (* *needle-radius* (- 1 (cos theta)))
           0                                    
           (* *needle-radius* (sin theta)))     
          points)))
(setq points (reverse points))

(defun make-rotation-matrix (p1 p2)
  (let* ((diff (v- p2 p1))
         (z-axis (normalize-vector diff))
         (y-axis (float-vector 0 1 0))
         (x-axis (v* y-axis z-axis)))
    (setq y-axis (v* z-axis x-axis))
    (transpose (matrix (normalize-vector x-axis) 
                      (normalize-vector y-axis) 
                      (normalize-vector z-axis)))))

;; アーチ状にシリンダーを作っていく
(setq needle-parts nil)
(let ((prev-point (car points)))
  (dolist (point (cdr points))
    (let* ((diff (v- point prev-point))
           (height (norm diff))
           (center (midpoint 0.5 prev-point point))
           (cylinder (make-cylinder *needle-thickness* height :segments 12))
           (rot-mat (make-rotation-matrix prev-point point)))
      (setq coords (make-coords :pos center :rot rot-mat))
      (send cylinder :transform coords)
      (push cylinder needle-parts)
      (setq prev-point point))))

;; bodysetにする
(setq needle-bodies (reverse needle-parts))

;; assocする
(dolist (body (cdr needle-bodies))
  (send (car needle-bodies) :assoc body))

;; 針の色（銀色）にする
(dolist (body needle-bodies)
  (send body :set-color :yellow))

;; needleをbodyset-linkとして定義する
(setq *needle* 
      (instance bodyset-link :init (make-cascoords)
                :bodies needle-bodies))

(send *needle* :translate (float-vector (- *centerx* 200) *centery* (+ *centerz* 40)))
(send *needle* :rotate #d180 :y)
(send *needle* :rotate #d90 :z)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; メモ: euslispのscriptで作れる基本的な幾何構造
;; --> https://euslisp.github.io/jskeus/jmanual-node118.html

;; 復習: euslispの座標系
;; 右手座標系 red-> x, green-> y, blue-> z

;; 左血管
(setq *left-tube* (make-cylinder 2.5 20))
(send *left-tube* :translate (float-vector (- *centerx* 200) *centery* (+ *centerz* 10)))
(send *left-tube* :rotate #d90 :x)
(send *left-tube* :translate (float-vector 0 0 -21))
(send *left-tube* :set-color :white)
(setq *left-tube-rc* (send (send *left-tube* :copy-worldcoords) :translate (float-vector 0 0 20)))

;; 右血管
(setq *right-tube* (make-cylinder 2.5 20))
(send *right-tube* :translate (float-vector (- *centerx* 200) *centery* (+ *centerz* 10)))
(send *right-tube* :rotate #d90 :x)
(send *right-tube* :translate (float-vector 0 0 1))
(send *right-tube* :set-color :white)
(setq *right-tube-lc* (send (send *right-tube* :copy-worldcoords) :translate (float-vector 0 0 0)))

;; 描画の初期化
(make-irtviewer)
(objects (list *pr2* *center* *desk* *table* *needle* *left-tube* *right-tube*))
(send *irtviewer* :draw-objects)

;; ここからメイン
;; 初期姿勢に戻る
(setq *voice-mode* nil)
(if *voice-mode* (send *ri* :speak-jp "初期姿勢に戻ります" :wait t))
(send *ri* :stop-grasp) ;; 両指を開く
;;(send *ri* :angle-vector (send *pr2* :reset-pose) 2000)  
(send *ri* :wait-interpolation)
(send *pr2* :torso :waist-z :joint-angle 270) ;; 予め腰を上げておくことで腕が机に干渉するのを回避)
(send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
(send *ri* :wait-interpolation)
(send *irtviewer* :draw-objects)


;; 針を持たせる
(setq *larm-coords* (send *pr2* :larm :end-coords :copy-worldcoords))
(send *larm-coords* :rotate #d180 :y)
(send *larm-coords* :rotate #d180 :z)
(send *larm-coords* :translate (float-vector (+ 10 *finger-offset*) 0 -10))
(send *needle* :move-to *larm-coords* :world)
(send (send *pr2* :larm :end-coords) :assoc *needle*)
(send *irtviewer* :draw-objects)

(if *voice-mode* (send *ri* :speak-jp "針を持たせてください" :wait t))
(send *ri* :stop-grasp :larm)
(send *ri* :wait-interpolation)
(ros::duration-sleep 2) ;; 2秒間待機する 
(send *ri* :start-grasp :larm)
(send *ri* :wait-interpolation)
;; TODO: FVが実装されたらここで正しく持てているかを確認して必要に応じて繰り返す.
(send *irtviewer* :draw-objects)

;; 縫い始める
;; 針の端に座標系を作成
(setq *needle-tip-coords* (make-cascoords :pos (send (elt needle-bodies (1- (length needle-bodies))) :worldpos)))
(send *needle-tip-coords* :rotate #d225 :x)
(send *needle* :assoc *needle-tip-coords*)


;; 目標の軌道を作成
(setq *target-coords* (send *left-tube-rc* :copy-worldcoords))
(send *target-coords* :rotate #d90 :x)  ;; 血管の向きに合わせる
(send *target-coords* :rotate #d180 :z)  ;; 針の先端を血管に向ける
(send *target-coords* :rotate #d180 :x)  ;; 上下を反転
;; IKを解く（torsoを使用）
;; メモ: https://euslisp-docs.readthedocs.io/en/latest/jskeus/jmanual/
(send *pr2* :larm :inverse-kinematics *target-coords* 
      :move-target *needle-tip-coords*
      :link-list (send *pr2* :link-list (send *pr2* :larm :end-coords :parent))
      :rotation-axis t
      :use-torsi t
      :debug-view t)
(send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
(send *ri* :wait-interpolation)
(send *irtviewer* :draw-objects)

;; 計画中
;; 曲率を持ったtrajectory
;; 軌道のパラメータ設定
(setq *needle_len* 70)
(setq *interval* 10)
(setq *traj_num* 10)
(setq *traj_len* (* *interval* *traj_num*))

;; 曲線的な軌道のためのtrajクラスを定義
(defclass curved-traj
  :super cascaded-coords
  :slots (points))

(defmethod curved-traj
  (:init (&key (radius 30.0) (start-angle -45) (end-angle 45))
         (send-super :init)  ;; 修正：argumentsなしで初期化
         (let* ((angle-step (/ (- end-angle start-angle) (1- *traj_num*)))
                (current-angle start-angle))
           ;; 円弧上の点を生成
           (setq points nil)  ;; pointsを初期化
           (dotimes (i *traj_num*)
             (let* ((theta (deg2rad current-angle))
                    (x (* radius (sin theta)))
                    (z (* radius (cos theta)))
                    (point (make-cube 10 10 10 
                                    :pos (float-vector x 0 z))))
               (push point points)
               (setq current-angle (+ current-angle angle-step))))
           (setq points (reverse points)))
         ;; 点同士を関連付け
         (dotimes (i (1- (length points)))
           (send (elt points i) :assoc (elt points (1+ i))))
         (send self :assoc (car points))
         self)
  (:points () points))

;; 血管縫合のための関数
(defun stitch-blood-vessels ()
  ;; 針の端に座標系を作成
  (setq *needle-tip-coords* (make-cascoords :pos (send (elt needle-bodies (1- (length needle-bodies))) :worldpos)))
  (send *needle* :assoc *needle-tip-coords*)

  ;; 最初の血管（左）への軌道
  (setq *target-coords* (send *left-tube-rc* :copy-worldcoords))
  
  ;; 曲線軌道を生成
  (setq *curved-path* (instance curved-traj :init))
  (send *curved-path* :translate (send *target-coords* :worldpos))
  
  ;; すべてのオブジェクトを表示（軌道の点を含む）
  (objects (append (list *pr2* *center* *desk* *table* *needle* *left-tube* *right-tube*)
                  (send *curved-path* :points)))
)

;; 実行
(stitch-blood-vessels)


;; for debug
;; (send (send *left-tube* :copy-worldcoords) :draw-on :flush t :size 100)
;; (send *left-tube-rc* :draw-on :flush t :size 100)
;; (send (send *right-tube* :copy-worldcoords) :draw-on :flush t :size 100)
;; (send *right-tube-lc* :draw-on :flush t :size 100)
