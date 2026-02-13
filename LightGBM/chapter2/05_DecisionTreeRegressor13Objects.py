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
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    return graphviz, mean_squared_error, mo, np, pd, plt, train_test_split


@app.cell
def _(pd):
    # データセットの読み込み
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data", header=None, sep="\s+")
    df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    return (df,)


@app.cell
def _(df):
    # 特徴量と目的変数の設定
    # 目的変数としてMEDV以外の13個全てを用いる
    X = df.drop(['MEDV'], axis=1)
    y = df['MEDV']
    X.head()
    return X, y


@app.cell
def _(X, train_test_split, y):
    # 学習データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)
    print('X_trainの形状: ', X_train.shape, ' y_trainの形状: ', y_train.shape,
          ' X_testの形状: ', X_test.shape, 'y_testの形状: ', y_test.shape)
    return X_test, X_train, y_test, y_train


@app.cell
def _(mo):
    mo.md(
        """
    ### DecisionTreeRegressorのハイパーパラメータ
    - criterison: squared_error 分割点を計算するときの誤差を指定する。二乗誤差が基本
    - max_depth: None 決定木の深さの最大値
      - 深さ4を指定している
    - min_samples_leaf: 1 葉の作成に必要な最小レコード数
      - 過学習を防ぐために10を指定して葉のレコード数を10個以上とする。
    - ccp_alpha: 0 端数に対する正則化の強さ
      - 正則化の強さに5を指定して葉が増えないよう制約を課す。
    """
    )
    return


@app.cell
def _(X_train, y_train):
    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor(
        criterion='squared_error',
        max_depth=4,
        min_samples_leaf=10,
        ccp_alpha=5,
        random_state=0,
    ) # 深さ4の回帰木モデル
    model.fit(X_train, y_train)
    model.get_params()
    return (model,)


@app.cell
def _(X_test, mean_squared_error, model, y_test):
    # テストデータの予測と評価
    # 決定木は獲得超量を個別に閾値と比較するだけなのでスケールの影響を受けない。
    # 従って、決定木では標準化では不要である。
    # 一方で、決定木以外のモデル（線形回帰、ロジスティック回帰、SVM、ニューラルネットワーク、k-NN）
    # などは標準化が必要である
    # 線形回帰のRMSEは5.78なので、わずかに精度が悪化している。
    y_test_pred = model.predict(X_test)
    print("RMSE test: %.2f" % (mean_squared_error(y_test, y_test_pred) ** 0.5))
    return


@app.cell
def _(X_test, model):
    # 予測値
    model.predict(X_test)
    return


@app.cell
def _(X, graphviz, mo, model):
    # 正則化のために深さが4の決定木ではあるが、葉の作成を4つに抑えることが出来ている。
    from sklearn import tree
    dot_data = tree.export_graphviz(
        model,
        out_file=None,
        rounded=True,
        feature_names=X.columns,
        filled=True,
    )
    # 明示的にSVGとしてレンダリングする方法はうまくいった
    svg_data = graphviz.Source(dot_data).pipe(format='svg').decode('utf-8')
    mo.Html(svg_data)
    return


@app.cell
def _(X, model, np, plt):
    # 決定木は学習の時に計算した特徴量の重要度を持っている
    importances = model.feature_importances_ # 特徴量の重要度
    indices = np.argsort(importances)[::-1] # 特徴量の重要度をキーとして降順にソート
    plt.figure(figsize=(8, 4)) # プロットのサイズ指定
    plt.title('Feature Importance') # プロットのタイトルを作成
    plt.bar(range(len(indices)), importances[indices]) # 棒グラフを追加
    plt.xticks(range(len(indices)), X.columns[indices], rotation=90) # X軸に特徴量の名前を追加
    plt.show() # プロットを表示
    return


if __name__ == "__main__":
    app.run()
