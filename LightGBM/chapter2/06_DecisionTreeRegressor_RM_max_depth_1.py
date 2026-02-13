import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    # ライブラリのインポート
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error

    return mean_squared_error, mo, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 深さ1のLightGBM回帰の可視化
    """)
    return


@app.cell
def _(pd):
    # データセットの読み込み
    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
        header=None,
        sep="\s+",
    )
    df.columns = [
        "CRIM",
        "ZN",
        "INDUS",
        "CHAS",
        "NOX",
        "RM",
        "AGE",
        "DIS",
        "RAD",
        "TAX",
        "PTRATIO",
        "B",
        "LSTAT",
        "MEDV",
    ]
    return (df,)


@app.cell
def _(df):
    # 特徴量と目的変数の設定
    X_train = df.loc[:99, ["RM"]]  # 特徴量に100件のRM（平均部屋数）を設定
    y_train = df.loc[:99, "MEDV"]  # 正解値に100件のMEDV（住宅価格）を設定
    print("X_train: ", X_train[:3])
    print("y_train: ", y_train[:3])
    return X_train, y_train


@app.cell
def _(X_train, y_train):
    # ハイパーパラメータの設定
    import lightgbm as lgb

    lgb_train = lgb.Dataset(X_train, y_train)

    params = {
        "objective": "mse",  # 1次微分と2次微分を計算する損失関数
        "metric": "mse",  # objectiveと異なる評価指標を使用するときに指定する
        "learning_rate": 0.8,  # 学習率
        "max_depth": 1,  # 決定木の深さの最大値
        "min_data_in_leaf": 1,  # 葉の作成に必要な最小レコード数
        "min_data_in_bin": 1,  # ヒストグラムの1つのbinに含まれる最小レコード数
        "max_bin": 100,  # ヒストグラムのbiの件数の最大値
        "seed": 0,
        "verbose": -1,
    }
    return lgb, lgb_train, params


@app.cell
def _(lgb, lgb_train, params):
    # モデルの学習
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=1,
        valid_sets=[lgb_train],
        valid_names=["train"],
    )
    return (model,)


@app.cell
def _(X_train, mean_squared_error, model, y_train):
    # 学習データの予測と評価
    y_train_pred = model.predict(X_train)
    print("MSE train: %.2f" % (mean_squared_error(y_train, y_train_pred)))
    return


@app.cell
def _(X_train, model):
    # 予測値
    model.predict(X_train)
    return


@app.cell
def _(lgb, model):
    # 木の可視化
    lgb.plot_tree(model, tree_index=0, figsize=(10, 10))
    return


@app.cell
def _(X_train, model, np, plt, y_train):
    # データと予測値の可視化
    plt.figure(figsize=(8, 4))  # プロットのサイズ指定
    X = X_train.values.flatten()  # numpy配列に変換し、1次現配列に変換
    y = y_train.values  # numpy配列に変換

    # Xの最小値から最大値まで0.01刻みのX_pltを作成し、2次元配列に変換
    X_plt = np.arange(X.min(), X.max(), 0.01)[:, np.newaxis]
    y_pred = model.predict(X_plt)  # 住宅価格を予想

    # 学習データ (平均部屋数と住宅価格)の散布図と予測値のプロット
    plt.scatter(X, y, color="blue", label="data")
    plt.plot(X_plt, y_pred, color="red", label="LightGBM")
    plt.ylabel("Price in $1000s [MEDV]")
    plt.xlabel("average number of rooms [RM]")
    plt.title("Boston house-price")
    plt.legend(loc="upper right")
    plt.show()
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 深さ1のLightGBM回帰の予測値の検証
    1回ブースティングする場合の予測値

    $$
    \hat{y} = \hat{y}^{(0)} + \eta w_1(\bold{x})
    $$
    """)
    return


@app.cell
def _(y):
    # 初期値
    print("samples: ", len(y))  # レコード数
    pred0 = sum(y) / len(y)  # 予測値(平均)
    print("pred0: ", pred0)
    return (pred0,)


@app.cell
def _(X, y):
    # 左葉のレコード
    threshold = 6.793  # 左右に分割する分割点
    X_left = X[X <= threshold]  # 左葉の特徴量
    y_left = y[X <= threshold]  # 左葉の正解値
    print(len(X_left), "/", len(X))
    print("X_left: ", X_left)
    print("y_left: ", y_left)
    return (y_left,)


@app.cell
def _(pred0, y_left):
    # 左葉の予測値
    print("samples left: ", len(y_left))  # 左葉のレコード数
    residual_left = y_left - pred0  # 残差
    weight_left = sum(residual_left) / len(y_left)  # 重み
    print("weight_left: ", weight_left)
    y_pred_left = pred0 + 0.8 * weight_left  # 左葉の予測値
    print("y_pred_left: ", y_pred_left)
    return


if __name__ == "__main__":
    app.run()
