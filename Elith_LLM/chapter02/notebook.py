import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import math

    return math, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Softmax関数

    $\begin{aligned}
    &\mathrm{softmax}(a_1, a_2, \cdots, a_n) \\
    &= \left(
    \frac{\exp(a_1)}{\sum_{i=1}^{n}\exp(a_i)},
    \frac{\exp(a_2)}{\sum_{i=1}^{n}\exp(a_i)},
    \cdots,
    \frac{\exp(a_n)}{\sum_{i=1}^{n}\exp(a_i)},\right)
    \end{aligned}$

    1. オーバーフロー対策として、入力の最大値を引いている
    2. `axis=-1`: 行ごとに正規化する為、入力が(バッチサイズ, クラス数)の２次元配列の時、softmaxは各サンプル(各行)ごとに合計が1になる必要がある。`e_x.sum()`だと配列全体の総和となり、全要素の合計が1になってしまう。`axis=-1`は最後の軸＝クラス方向の和なので、行ごとの和になる。
    3. `keepdims=True`: ブロードキャストを正しく効かせるため
    """)
    return


@app.cell
def _(np):
    def softmax(x):
        print(x)
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    return (softmax,)


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Attention
    1. 値がマイナスの場合がある
    2. クエリによって注意度の合計値が異なる
    この問題の大差置くとして、$\mathrm{softmax}$関数を用いて合計が1で値が正になるようにしている
    3. attentionを数式で書くと以下の様になる。
      - $\begin{aligned}
        \bold{o} = \mathrm{softmax}(\bold{q}K^T)V
        \end{aligned}$
    """)
    return


@app.cell
def _(np, softmax):
    def attention(q, K, V):
        attention_weights = softmax(np.matmul(q, K.T))
        return np.matmul(attention_weights, V), attention_weights

    return (attention,)


@app.cell
def _(attention, math, np):
    n = 10
    vectors = []
    # 360°をn等分した角度(ラジアン)
    theta = 2 * math.pi / n
    for i in range(n):
        x = math.cos(theta * i)
        y = math.sin(theta * i)
        vectors.append([x, y])

    vectors = np.array(vectors)
    query = np.array([1 / math.sqrt(2), 1 / math.sqrt(2)])
    output, attention_weights = attention(query, vectors, vectors)
    print("出力ベクトル:", output)
    print("注意度の形:", attention_weights.shape)
    print("注意度の和:", attention_weights.sum())
    return


if __name__ == "__main__":
    app.run()
