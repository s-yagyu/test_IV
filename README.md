#### IV data の読み込みと可視化

#### Dataについて

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

#### 利用方法

##### 必要なモジュール
- numpy, scipy, pandas, matplotlib, 

- Data_read.ipynbのテンプレートを利用する
- テンプレートをコピーして、名前を変更する
- テンプレートと同じ階層にdata_reader2.pyを配置する
（将来的には、githubからpipでインストールできるようにする。gitのインストールが必要）
- テンプレートに従って進める


