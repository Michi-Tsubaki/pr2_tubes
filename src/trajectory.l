#!/usr/bin/env roseus

;; パラメタの初期値 (mainで上書きされる)
(setq *interval* 1.5)
(setq *traj_num* 5)
(setq *traj_len* (* *interval* *traj_num*))

;; 直線状の軌跡の定義
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


;; 曲がった軌跡の定義
;; 曲線的(円弧)の軌道
(setq *interval* 1.5)
(setq *traj_num* 5)
(setq *traj_len* (* *interval* *traj_num*))

(defclass curved-traj
  :super cascaded-coords
  :slots (points))
(defmethod curved-traj
  (:init (&key (radius 30.0) (start-angle -25) (end-angle 25)) ;; 初期値は適当
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
		    ;; 接線ベクトル（円の接線なので、半径方向に垂直）
		    (tangent-x (* radius (cos theta)))  ;; 接線のx成分
		    (tangent-z (* -1 radius (sin theta)))  ;; 接線のz成分
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
