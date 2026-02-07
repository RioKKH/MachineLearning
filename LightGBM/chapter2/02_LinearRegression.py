import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 重回帰の学習⇒予測⇒評価""")
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    return mean_squared_error, mo, np, pd, plt, train_test_split


@app.cell
def _(pd):
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, sep='\s+')
    df.columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    df.head()
    return (df,)


@app.cell
def _(df):
    # 特徴量と目的変数の設定
    #print(df.describe())
    X = df.drop(['MEDV'], axis=1)
    y = df['MEDV']
    X.head()
    return X, y


@app.cell
def _(X, train_test_split, y):
    # 学習データとテストデータに分割
    # 比率でデータセットのレコードを分割する方法をホールドアウト法という
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)
    print('X_trainの形状: ', X_train.shape, 'y_trainの形状: ',  y_train.shape, 'X_testの形状: ', X_test.shape, 'y_testの形状: ', y_test.shape)
    return X_test, X_train, y_test, y_train


@app.cell
def _(X, X_test, X_train):
    # 特徴量の標準化
    # 標準化も変換器を作ってfit()を用いるのはscikit-learnの作法
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler() # 変換器の作成
    num_cols = X.columns[0:13] # 全て数値型の特徴量なので全てのインデックスを取得
    scaler.fit(X_train[num_cols]) # 学習データでの標準化パラメータの計算
    X_train[num_cols] = scaler.transform(X_train[num_cols]) # 学習データの変換
    X_test[num_cols] = scaler.transform(X_test[num_cols]) # テストデータの変換
    X_train.iloc[:2] # 標準化された学習データの特徴量
    return


@app.cell
def _(X_train, y_train):
    # モデルの学習
    from sklearn.linear_model import LinearRegression
    model = LinearRegression() # 線形回帰モデル
    model.fit(X_train, y_train)
    model.get_params()
    return (model,)


@app.cell
def _(X_test, mean_squared_error, model, y_test):
    # テストデータの予測と評価
    y_test_pred = model.predict(X_test)
    print('RMSE test: %.2f' % (mean_squared_error(y_test, y_test_pred) ** 0.5))
    return (y_test_pred,)


@app.cell
def _(y_test):
    # テストデータの目的関数の統計情報
    # RMSEは標準偏差と比較できる。テストデータの標準偏差は下に示すようにおおよそ9.07。RMSEは5.78なので、予測モデルはテストデータの平均を予測値とした9.07よりも優れた予測値となっている
    y_test.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    1. 標準偏差(何も学習しない場合)：データの平均値を予測値とする

    $$
    \hat{y}_i = \bar{y}
    $$

    この場合のRMSEは

    $$
    RMSE_{baseline} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\bar{y})^2} = \sigma_y
    $$

    2. 学習したモデルのRMSE

    $$
    RMSE_{model} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y})^2}
    $$

    3. 比較の意味
      - RMSE < 標準偏差 $\rightarrow$ モデルは平均値予測より優れている
      - RMSE $\simeq$ 標準偏差 $\rightarrow$ モデルは殆ど学習できていない
      - RMSE < 標準偏差 $\rightarrow$ モデルが逆に悪化(過学習など)
    """
    )
    return


@app.cell
def _(model):
    # 標準化済みの特徴量を用いているので回帰係数のスケールもそろっている
    # パラメータ
    print('回帰係数 w = [w1, w2, ..., w3]: ', model.coef_)
    print('定数項 w0:', model.intercept_)
    return


@app.cell
def _(X):
    # 特徴量の列をテキスト表示する
    X.columns
    return


@app.cell
def _(X, model, np, plt):
    # 回帰係数の可視化
    importance = model.coef_ # 回帰係数
    indices = np.argsort(importance)[::-1] # 回帰係数を降順にソートしてそのインデックスを取得する
    plt.figure(figsize=(8, 4)) # プロットのサイズ指定
    plt.title('Regression coefficient') # プロットのタイトルを作成
    plt.bar(range(X.shape[1]), importance[indices]) # 棒グラフを追加
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90) # X軸に特徴量の名前を追加
    plt.show()
    return


@app.cell
def _(y_test_pred):
    # 予測値のリスト
    y_test_pred[:20]
    # 15件目の予測値は39.99で定数項の22.61に比べると住宅価格が高くなっている。この理由を知りたい
    return


@app.cell
def _(y_test_pred):
    # 15件目の予測値
    y_test_pred[14]
    return


@app.cell
def _(X_test):
    # 15件目の特徴量を表示する
    # 特徴量の数値は標準化済みの数値であることに注意。
    print('15件目の特徴量 X = [x1, x2, ..., x13: ', X_test.values[14]) # pandasをnumpyに変換
    # 6個目と15個目の特徴量が大きい(標準化しているので、σが1であることに留意すること)
    # これはRMとLSTATと回帰係数も大きく、掛け合わせた結果がどちらもプラスになることから15件目がこの2つにおいて回帰直線から外れているということが分かる
    return


@app.cell
def _(X_test, model, np):
    # 15件目の予測値の検証
    # y = w * X + w0
    np.sum(model.coef_ * X_test.values[14]) + model.intercept_
    return


if __name__ == "__main__":
    app.run()
