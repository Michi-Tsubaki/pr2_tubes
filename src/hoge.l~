#!/usr/bin/env roseus

(load "package://pr2eus/pr2-interface.l")
(load "models/arrow-object.l")

(pr2-init)
(setq *arrow* (arrow))

(setq *cube* (make-cube 200 10 10))

(objects (list *pr2* *arrow* *cube*))

(send *arrow* :newcoords (make-coords :pos #f(800 -400 1200)))
(send *cube* :newcoords (send (send *pr2* :rarm :end-coords) :copy-worldcoords))
(send *cube* :translate #f(100 0 0) :local)
(setq *cube-tip-pos* (make-cascoords :coords (send *cube* :copy-worldcoords)))
(send *cube-tip-pos* :translate #f(100 0 0) :local)
(send *cube-tip-pos* :assoc *cube*)
;; when :move-arm is set and :link-list is nil, '(send move-target :parent)' must be link
;; https://github.com/jsk-ros-pkg/jsk_pr2eus/blob/2b5af0cd5377c275f7c39975b9d46b574e8ee1e4/pr2eus/pr2-utils.l#L117
;; (send (send *pr2* :rarm :end-coords) :assoc *cube*)
(send (send *pr2* :rarm :end-coords :parent) :assoc *cube-tip-pos*)

;; default usage 1, set only target-coords, use :use-torso :t for option
(send *pr2* :reset-pose)
;;(format t "before ik torso joint is ~A~%" (send *pr2* :torso_lift_joint :joint-angle))
(format t "before ik torso joint is ~A~%" (send *pr2* :torso :waist-z :joint-angle))
(send *pr2* :rarm :angle-vector #f(0 0 0 -90 -180 -90 180.0)) ;; use nice initial pose
(send *pr2* :rarm :inverse-kinematics (send *arrow* :copy-worldcoords)
      :use-torso t
      :rotation-axis nil
      :debug-view :no-message) ;; 50.0
(send *irtviewer* :draw-objects)
(format t "after ik torso joint is ~A~%" (send *pr2* :torso_lift_joint :joint-angle))

(send *pr2* :reset-pose)
(format t "before ik torso joint is ~A~%" (send *pr2* :torso_lift_joint :joint-angle))
(send *pr2* :rarm :angle-vector #f(0 0 0 -90 -180 -90 180.0)) ;; use nice initial pose
(send *pr2* :rarm :inverse-kinematics (send *arrow* :copy-worldcoords)
      :use-torso nil
      :rotation-axis t
      :debug-view :no-message) ;; 50.0
(send *irtviewer* :draw-objects)
(format t "after ik torso joint is ~A~%" (send *pr2* :torso_lift_joint :joint-angle))

;; default usage 2, use move-target. we expect '(send move-target :parent)' is a link
(send *pr2* :reset-pose)
;;(format t "before ik torso joint is ~A~%" (send *pr2* :torso_lift_joint :joint-angle))
(format t "before ik torso joint is ~A~%" (send *pr2* :torso :waist-z :joint-angle))
(send *pr2* :rarm :angle-vector #f(0 0 0 -90 -180 -90 180.0)) ;; use nice initial pose
(send *pr2* :rarm :inverse-kinematics (send *arrow* :copy-worldcoords)
      :move-target *cube-tip-pos*
      :use-torso t
      :rotation-axis nil
      :debug-view :no-message) ;; 50.0
(send *irtviewer* :draw-objects)
(format t "after ik torso joint is ~A~%" (send *pr2* :torso_lift_joint :joint-angle))

(send *pr2* :reset-pose)
(format t "before ik torso joint is ~A~%" (send *pr2* :torso_lift_joint :joint-angle))
(send *pr2* :rarm :angle-vector #f(0 0 0 -90 -180 -90 180.0)) ;; use nice initial pose
(send *pr2* :rarm :inverse-kinematics (send *arrow* :copy-worldcoords)
      :move-target *cube-tip-pos*
      :use-torso nil
      :rotation-axis t
      :debug-view :no-message) ;; 50.0
(send *irtviewer* :draw-objects)
(format t "after ik torso joint is ~A~%" (send *pr2* :torso_lift_joint :joint-angle))

;; advanced usage 1, specify link-list, in this case, you do not need to set :use-torso, because it is set within link-list
(send *pr2* :reset-pose)
(format t "before ik torso joint is ~A~%" (send *pr2* :torso :waist-z :joint-angle))
(send *pr2* :rarm :angle-vector #f(0 0 0 -90 -180 -90 180.0)) ;; use nice initial pose
(send *pr2* :rarm :inverse-kinematics (send *arrow* :copy-worldcoords)
      :move-target *cube-tip-pos*
      :link-list (send *pr2* :link-list (send *pr2* :rarm :end-coords :parent))
      ;; :use-torso t
      :rotation-axis t
      :debug-view :no-message) ;; 50.0
(send *irtviewer* :draw-objects)
(format t "after ik torso joint is ~A~%" (send *pr2* :torso_lift_joint :joint-angle))

(send *pr2* :reset-pose)
(format t "before ik torso joint is ~A~%" (send *pr2* :torso_lift_joint :joint-angle))
(send *pr2* :rarm :angle-vector #f(0 0 0 -90 -180 -90 180.0)) ;; use nice initial pose
(send *pr2* :rarm :inverse-kinematics (send *arrow* :copy-worldcoords)
      :move-target *cube-tip-pos*
      :link-list (send *pr2* :link-list (send *pr2* :rarm :end-coords :parent) (send *pr2* :rarm :root-link))
      ;; :use-torso nil
      :rotation-axis t
      :debug-view :no-message) ;; 50.0
(send *irtviewer* :draw-objects)
(format t "after ik torso joint is ~A~%" (send *pr2* :torso_lift_joint :joint-angle))

;; advanced usage 2, you do not need call from  :rarm object, call :ik from *pr2* object
(send *pr2* :reset-pose)
(format t "before ik torso joint is ~A~%" (send *pr2* :torso :waist-z :joint-angle))
(send *pr2* :rarm :angle-vector #f(0 0 0 -90 -180 -90 180.0)) ;; use nice initial pose
(send *pr2* :inverse-kinematics (send *arrow* :copy-worldcoords)
      :move-target *cube-tip-pos*
      :link-list (send *pr2* :link-list (send *pr2* :rarm :end-coords :parent))
      ;; :use-torso t
      :rotation-axis t
      :debug-view :no-message) ;; 50.0
(send *irtviewer* :draw-objects)
(format t "after ik torso joint is ~A~%" (send *pr2* :torso_lift_joint :joint-angle))

(send *pr2* :reset-pose)
(format t "before ik torso joint is ~A~%" (send *pr2* :torso_lift_joint :joint-angle))
(send *pr2* :rarm :angle-vector #f(0 0 0 -90 -180 -90 180.0)) ;; use nice initial pose
(send *pr2* :inverse-kinematics (send *arrow* :copy-worldcoords)
      :move-target *cube-tip-pos*
      :link-list (send *pr2* :link-list (send *pr2* :rarm :end-coords :parent) (send *pr2* :rarm :root-link))
      ;; :use-torso nil
      :rotation-axis t
      :debug-view :no-message) ;; 50.0
(send *irtviewer* :draw-objects)
(format t "after ik torso joint is ~A~%" (send *pr2* :torso_lift_joint :joint-angle))
