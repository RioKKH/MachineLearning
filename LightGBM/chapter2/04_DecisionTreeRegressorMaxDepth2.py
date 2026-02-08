import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    # ライブラリのインポート
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import graphviz
    from sklearn.metrics import mean_squared_error
    return graphviz, mo, np, pd, plt


@app.cell
def _(pd):
    # データセットの読み込み
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data", header=None, sep="\s+")
    df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    return (df,)


@app.cell
def _(df):
    # 特徴量と目的関数の設定
    X_train = df.loc[:99, ['RM']] # 特徴量に100件のRM(平均部屋数)を設定
    y_train = df.loc[:99, 'MEDV'] # 正解値に100件のMEDV(住宅価格)を設定
    return X_train, y_train


@app.cell
def _(X_train, y_train):
    # モデルの学習
    from sklearn.tree import DecisionTreeRegressor

    model = DecisionTreeRegressor(criterion='squared_error', max_depth=2, min_samples_leaf=1, ccp_alpha=0, random_state=0) # 深さ2の回帰モデル
    model.fit(X_train, y_train)
    model.get_params()
    return (model,)


@app.cell
def _(X_train, model):
    # 予測値
    # 予測値は4つになる(2^2なので)
    model.predict(X_train)
    return


@app.cell
def _(graphviz, mo, model):
    # 木の可視化
    from sklearn import tree
    dot_data = tree.export_graphviz(
        model,
        out_file=None,
        rounded=True,
        feature_names=['RM'],
        filled=True
    )
    # 明示的にSVGとしてレンダリングする方法はうまくいった
    svg_data = graphviz.Source(dot_data).pipe(format='svg').decode('utf-8')
    mo.Html(svg_data)
    return


@app.cell
def _(X_train, model, np, plt, y_train):
    # データと予測値の可視化
    plt.figure(figsize=(8, 4)) # プロットのサイズ指定
    X = X_train.values.flatten() # numpy配列に変換し、1次元配列に変換
    y = y_train.values # numpy配列に変換

    # Xの最小値から最大値まで0.01刻みのX_pltを作成し、2次元配列に変換する
    X_plt = np.arange(X.min(), X.max(), 0.01)[:, np.newaxis]
    y_pred = model.predict(X_plt) # 住宅価格を予想

    # 学習データ (平均部屋数と住宅価格)の散布図と予測値のプロット
    plt.scatter(X, y, color='blue', label='data')
    plt.plot(X_plt, y_pred,
             color='red', label='DecisionTreeRegressor')
    plt.ylabel('Price in $1000s [MEDV]')
    plt.xlabel('average number of rooms [RM]')
    plt.title('Boston house-prices')
    plt.legend(loc='upper right')
    plt.show()
    return


app._unparsable_cell(
    r"""
    import
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
