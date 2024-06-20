このモジュールはIV測定のデータファイルの読み込みとSchottky解析を行います。

### 利用方法

#### 必要なモジュール
- numpy, scipy, pandas, matplotlib, jupyter
- pynverse 
```
pip install git+https://github.com/alvarosg/pynverse.git
```
このモジュールのインストール方法
```
pip install git+https://github.com/s-yagyu/test_IV.git
```
#### テンプレート
- それぞれのテンプレートをコピーして、名前を変更する
- テンプレートに従って進める
- IV_data_read_template.ipynb  (IV data の読み込みと可視化)
- schottky_simulation_templae.ipynb (Schottkyのシミュレーション)
- IV_data_read_and_schottky_template_.ipynb (IV data の読み込みと可視化の後、SchottkyモデルでFiting)


#### 読み込むDataについて

Example
```
IG88--- IG88_C-V Sweep [_ 2 -17_Subsite_0__Subsite_(57) ; 2022-07-06 11_53_44 AM].csv
     |   ...
     |- IG88_Generic C-f [_ 0 -2_Subsite_0__Subsite_(4) ; 2022-07-06 11_21_58 AM].csv
     |   ...
     |- IG88_I_V Sweep [_ 1 0_Subsite_0__Subsite_(21) ; 2022-07-06 11_30_17 AM].csv
     |   ...
```

今回利用するデータはI_V

#### I_Vのファイルの中身

- ヘッダー : 1行目から248行目まで
- コラム名 : 249行目 列名 : V1, I1
- IV測定は　-5V-->5V, 5V-->-5Vとして連続で往復測定
（可視化では、行きと帰りの2つのスペクトルに分けて描画）




