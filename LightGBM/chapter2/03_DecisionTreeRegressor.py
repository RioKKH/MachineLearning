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
    # 特徴量と目的変数の設定
    X_train = df.loc[:99, ['RM']] # 特徴量に100件のRM(平均部屋数)を設定
    y_train = df.loc[:99, 'MEDV'] # 正解値に100件のMDEV(住宅価格)を設定
    print('X_train: ', X_train[:3])
    print('y_train: ', y_train[:3])
    return X_train, y_train


@app.cell
def _(X_train, y_train):
    # モデルの学習
    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor(criterion='squared_error', max_depth=1, min_samples_leaf=1, random_state=0) # 回帰木モデル
    model.fit(X_train, y_train)
    model.get_params()
    return (model,)


@app.cell
def _(y_train):
    # 目的変数
    y_train
    return


@app.cell
def _(X_train, model):
    # 予測値
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
    return (dot_data,)


@app.cell
def _(dot_data, graphviz):
    # 画像として保存して表示することも可能
    from PIL import Image
    import io

    png_data = graphviz.Source(dot_data).pipe(format='png')
    image = Image.open(io.BytesIO(png_data))
    image
    return


@app.cell
def _(X_train, model, np, plt, y_train):
    # データと予測値の可視化
    plt.figure(figsize=(8, 4)) # プロットのサイズ指定
    X = X_train.values.flatten() # numpy配列に変換し、1次元配列に変換
    y = y_train.values # numpy配列に変換

    # Xの最小値から最大値まで0.01刻みのX_pltを作成し、2次元配列に変換する
    # print(X_plt.shape) # shape of X_plt: (268,)
    X_plt = np.arange(X.min(), X.max(), 0.01)[:, np.newaxis]
    # print(X_plt.shape) # shape of X_plt: (268, 1)
    y_pred = model.predict(X_plt) # 住宅価格の予測

    # 学習データ (平均部屋数と住宅価格)の散布図と予測値のプロット
    plt.scatter(X, y, color='blue', label='data')
    plt.plot(X_plt, y_pred, color='red', label='DecisionTreeRegressor')
    plt.ylabel('Price in $1000s [MEDV]')
    plt.xlabel('average number of rooms [RM]')
    plt.title('Boston house-prices')
    plt.legend(loc='upper right')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 深さ1の回帰木の予測値の検証

    scikit-learnの関数を使わずに、自前で決定木(回帰木)を実装する
    """
    )
    return


@app.cell
def _(X_train, y_train):
    # データのソート
    X_train_sorted = X_train.sort_values('RM') # 特徴量RMの分割点計算前にソート
    y_train_sorted = y_train[X_train_sorted.index] # 正解もソート

    X_train_array = X_train_sorted.values.flatten() # numpy化して2次元配列->1次元配列
    y_train_array = y_train_sorted.values # numpy化

    print(X_train_array[:10])
    print(y_train_array[:10])
    return X_train_array, y_train_array


@app.cell
def _(X_train_array, np, y_train_array):
    # 分割点の計算、というか探索かな？
    index = []
    loss = []
    # 分割点毎の予測値、SSE、MSEを計算する
    for i in range(1, len(X_train_array)):
        X_left = np.array(X_train_array[:i])
        X_right = np.array(X_train_array[i:])
        y_left = np.array(y_train_array[:i])
        y_right = np.array(y_train_array[i:])

        # 分割点のインデックス
        print('*****')
        print('index', i)
        index.append(i)

        # 左右の分割
        print('X_left: ', X_left)
        print('X_right: ', X_right)
        print('')

        # 予測値の計算
        print('y_pred_left: ', np.mean(y_left))
        print('y_pred_right: ', np.mean(y_right))
        print('')

        # SSEの計算
        y_error_left = y_left - np.mean(y_left)
        y_error_right = y_right - np.mean(y_right)
        SSE = np.sum(y_error_left * y_error_left) + np.sum(y_error_right * y_error_right)
        print('SSE: ', SSE)
        loss.append(SSE)

        # MSEの計算
        MSE_left = 1/len(y_left) * np.sum(y_error_left * y_error_left)
        MSE_right = 1/len(y_right) * np.sum(y_error_right * y_error_right)

        print('MSE_left: ', MSE_left)
        print('MSE_right: ', MSE_right)
        print('')
    
    return index, loss


@app.cell
def _(index, loss, np, plt):
    # 分割点とSSEの可視化
    _X_plt = np.array(index)[:, np.newaxis] # 1次元配列⇒2次元配列
    plt.figure(figsize=(10, 4)) # プロットのサイズ指定
    plt.plot(_X_plt, loss)
    plt.xlabel('index')
    plt.ylabel('SSE')
    plt.title('Split Point index of feature RM')
    plt.grid()
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
