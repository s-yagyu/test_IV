#### Pythonを導入する方法

##### インストールするアプリケーション

- Miniconda
- VS Code
- Git

#### Miniconda
Condaを利用するのに一番手っ取り早いのがAnacondaの導入ですが、数年前からAnacondaのライセンス条件がきつくなりっており、オープンソースのminicondaを導入することにします。

Minicondaのアプリケーションのインストールについては、他のサイトでの説明があるのでそちらを参考に導入してください。

[Windowsでコマンドライフ２：minicondaでpython環境構築｜うちの実験的Web (makeintoshape.com)](https://makeintoshape.com/windows-commandlife2/)

Minicondaをインストールしたら、仮想環境を作ります。

Baseには何も加えず仮想環境を作りそちらで運用します。

Anaconda Promptを立ち上げ、仮想環境を作成します。その後、その仮想環境でパッケージやモジュールはすべてpipでインストールします。（Conda installは使いません。）
```powershell
# 例えば、仮想環境の構築 -nはオプションで　名前を記入します
conda create -n pip11 python=3.11

# 仮想環境ができたら、仮想環境に移動
conda activate pip11

# 仮想環境から抜けるには
conda deactivate

# 仮想環境に入り　pipでインストール
pip install numpy scipy matplotlib jupyter seaborn pandas openpyxl 

# ----　その他に必要なものなど
pip install scikit-learn python-opencv opencv-contrib-python Flet

```
なお仮想環境は一般的には、`C:/Users/ユーザー名/miniconda3/envs`の中に作られます。



#### VS Code

[Download Visual Studio Code - Mac, Linux, Windows](https://code.visualstudio.com/Download)

VSCodeでは作成した仮想環境を利用してください。



#### Git

[Git - Downloads (git-scm.com)](https://git-scm.com/downloads)



