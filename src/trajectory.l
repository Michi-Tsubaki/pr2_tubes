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
		    )
	       (send point  :set-color :black)
	       (send point :orient (- 0 (deg2rad current-angle)) :y)
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
