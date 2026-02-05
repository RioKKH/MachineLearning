import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""## 単回帰の予測値の可視化""")
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error
    return mo, np, pd, plt


@app.cell
def _(pd):
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data", header=None, sep='\s+')
    df.columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
    df.head()
    return (df,)


@app.cell(hide_code=True)
def _(df):
    X_train = df.loc[:99, ["RM"]] # 特徴量に100件のRM(平均部屋数)を設定
    y_train = df.loc[:99, "MEDV"] # 正解値に100件のMEDV(住宅価格)を設定
    print("X_train:", X_train[:3])
    print("y_train:", y_train[:3])
    return X_train, y_train


@app.cell
def _(X_train, y_train):
    # モデルの学習
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(X_train, y_train)
    model.get_params()
    return (model,)


@app.cell
def _(X_train, model):
    # 予測値
    model.predict(X_train)
    return


@app.cell
def _(X_train, model, np, plt, y_train):
    # データと予測値の可視化
    plt.figure(figsize=(8, 4))
    X = X_train.values.flatten() # numpy配列に変換し、さらに1次元配列に変換
    y = y_train.values # numpy配列に変換

    # Xの最小値から最大値まで0.01刻みのX_pltを作成し、2次元配列に変換する
    # X_pltがnumpy配列のままなので、警告が出るが無視する
    X_plt = np.arange(X.min(), X.max(), 0.01)[:, np.newaxis]
    y_pred = model.predict(X_plt) # 住宅価格を予測

    # 学習データ(平均部屋数と住宅価格)の散布図と予測値のプロット
    plt.scatter(X, y, color='blue', label='data')
    plt.plot(X_plt, y_pred, color='red', label='LinearRegression')
    plt.ylabel('Price in $1000s [MDEV]')
    plt.xlabel('average number of rooms [RM]')
    plt.title('Boston house-prices')
    plt.legend(loc='upper right')
    # plt.show()
    return


@app.cell
def _(model):
    print('傾き w1:', model.coef_[0])
    print('切片 w0:', model.intercept_)
    return


if __name__ == "__main__":
    app.run()
