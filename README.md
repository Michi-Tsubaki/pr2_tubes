# pr2_tubes

このリポジトリでは，PR2 (双腕ロボット) を用いて，ベンド針で動脈血管モデル(直径5mmのチューブ)の縫合を行う目的で作成したの様々なツールとコードを管理している．このリポジトリをcloneすれば，ROSのpkgとして使えるはず．

## はじめに
2024年度Aセメスターの機械工学少人数ゼミ（岡田教授）の続きの位置づけ．ゼミでは，直針を用いて傷口に見立てたスポンジを縫合した． --> [pr2_surgery](https://github.com/Michi-Tsubaki/jsk_demos/tree/20a08aeaf60930f3e58cbce8068f3236fe40f4a6/jsk_2024_10_semi/pr2_surgery) を参照

## 主要な機能（クラス・関数）
- 縫合に用いるTrajectoryの生成
  - `traj`真っ直ぐな軌跡 (引数: 軌跡の長さ, 位置) <-- pr2_surgeryから継承
  - `curved_traj`曲率を持った軌跡 (引数: 入り角, 出角, 半径, 位置)
- 縫合する関数 (for 曲率を持った軌跡)
  - `suture` angle-vector-sequence によって縫合．針の先端がターゲットのtrajectoryを縫う．（関数内部で針の先端の座標系をassoc, dissocしたりしている．）

## eus
2月25日現在

https://github.com/user-attachments/assets/d5409fd7-fd25-4de7-9d43-941d721092f9
