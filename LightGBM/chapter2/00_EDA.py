import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    return mo, pd, plt, sns


@app.cell
def _(pd):
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data", header=None, sep="\s+")
    df.columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
    df.head()
    return (df,)


@app.cell
def _(df):
    # データの形状を確認
    df.shape
    return


@app.cell
def _(df):
    # 欠損値の有無を確認
    df.isnull().sum()
    return


@app.cell
def _(df):
    # データ型の表示
    df.info()
    return


@app.cell
def _(df):
    # 住宅価格の統計情報一覧
    df['MEDV'].describe()
    return


@app.cell
def _(df):
    # ヒストグラム作成
    df['MEDV'].hist(bins=30)
    return


@app.cell
def _(df, plt, sns):
    # 相関係数の確認
    plt.figure(figsize=(12, 10))
    df_corr = df.corr()
    sns.heatmap(df_corr, vmax=1, vmin=-1, center=0, annot=True, cmap='Blues')
    return


@app.cell
def _(df, sns):
    # 散布図を用いて相関関係を可視化する
    num_cols = ['LSTAT', 'RM', 'MEDV']
    sns.pairplot(df[num_cols], height=2.5)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### MSE (平均二乗誤差)
    - 誤差を二乗するため、予測が外れて大きな誤差が出た時、外れ値の誤差が強調されて、MSEは大きくなる。

    $$
    MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
    $$

    ### RMSE (二乗平均平方誤差)
    - 数値の単位が目的変数と一致する。ただし外れ値の影響を受けやすい
    - 小さな誤差より、外れ値など大きな誤差への当てはまりを重視したい時に採用するとよい
    - RMSEは誤差が正規分布に従う時の誤差説明に有効⇒RMSEと標準偏差は比較可能で、予測モデルにより、誤差がどの程度改善したかを説明できる。

    $$
    RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
    $$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### $R^1$ (決定係数)
    - MSEと分散を比較してスコアかした評価指標
    - 通常0~1の範囲になる
    - 決定係数はRMSEと標準偏差の誤差を比較するイメージ
    - 予測モデルの予測値が平均値よりも正解値への当てはまりが悪い場合、決定係数はマイナスになる。

    $$
    R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}{\sum_{i=1}^{n}(y_i-\bar{y})^2}
    $$

    ### MAE (平均絶対誤差)
    - 誤差の絶対値の平均で平均的な残差を示す
    - 誤差を二乗しないので、評価指標が残差の大きさを示す
    - RMSEと比べて小さな誤差を重視するので、外れ値の影響を重視しない場合に有効

    $$
    MAE = \frac{1}{n}\sum_{i=1}^n|y_i-\hat{y}_i|
    $$
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
