import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3-6: 位置埋め込みの可視化

    Vision Transformerの学習済みモデルから**位置埋め込み（Positional Embedding）**を取り出し、
    パッチ間のコサイン類似度を計算することで可視化します。

    位置埋め込みが正しく学習されていれば、**空間的に近いパッチ同士は類似した埋め込みベクトルを持つ**はずです。
    この可視化によって、モデルが位置情報をどの程度捉えているかを確認できます。

    ## ライブラリのインポート

    | ライブラリ | 説明 |
    |-----------|------|
    | `math` | `sqrt` を使ってパッチ数から格子サイズ（`H_and_W`）を計算する |
    | `torch` | モデルの読み込みと state_dict の取得 |
    | `torch.nn.functional` | コサイン類似度の計算（`F.cosine_similarity`） |
    | `matplotlib` | 類似度マップの可視化 |
    | `vit.Vit` | 学習済みモデルのクラス定義 |
    """)
    return


@app.cell
def _():
    import math
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F

    from vit import Vit

    return F, Vit, math, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## モデルの準備

    `fashionMNIST.py` で保存した学習済みの重みファイル `vit_fashionmnist.pth` を読み込みます。

    | 操作 | 関数 | 説明 |
    |------|------|------|
    | 重みファイルの読み込み | `torch.load(path, map_location=device)` | `map_location` を指定することで、GPU学習済みモデルをCPU環境でも読み込める |
    | 重みの適用 | `model.load_state_dict(state_dict)` | 保存されたパラメータをモデルに反映する |
    | 評価モード | `model.eval()` | Dropoutを無効化し、推論モードにする |

    > **注意**: `vit_fashionmnist.pth` が存在しない場合は、先に `fashionMNIST.py` を実行してモデルを学習・保存してください。
    """)
    return


@app.cell
def _(Vit, torch):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Vit(
        in_channels=1,
        num_classes=10,
        emb_dim=64,
        num_patch_row=2,
        image_size=28,
        num_blocks=3,
        head=4,
        hidden_dim=64 * 4,
        dropout=0.1,
    ).to(device)

    model.load_state_dict(
        torch.load("vit_fashionmnist.pth", map_location=device, weights_only=True)
    )
    model.eval()
    print("モデルを読み込みました")
    return device, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 位置埋め込みの取得

    `model.state_dict()` はモデルの全パラメータを辞書形式で返します。
    位置埋め込みは `VitInputLayer` の `pos_emb` に格納されているため、
    キー名は **`'input_layer.pos_emb'`** となります（書籍コードの `'pos_embed'` とは異なります）。

    | 変数 | 形状 | 説明 |
    |------|------|------|
    | `pos_embed` | `(1, N, D)` | N = パッチ数 + 1（クラストークン）、D = `emb_dim` |
    | `H_and_W` | スカラー | パッチの格子サイズ。`sqrt(N - 1)` で求める |

    本実装では `num_patch_row=2` のため：
    - N = 4（パッチ）+ 1（クラストークン）= **5**
    - H_and_W = $\sqrt{4}$ = **2**
    - D = **64**

    インデックス `[0, 1:]` でクラストークンを除外し、パッチトークンのみを取り出します。
    """)
    return


@app.cell
def _(math, model):
    # state_dict から位置埋め込みを取得
    # vit.py では VitInputLayer.pos_emb として定義されているため
    # キー名は 'input_layer.pos_emb'
    pos_embed = model.state_dict()["input_layer.pos_emb"]  # shape: (1, N, D)

    # クラストークン（index=0）を除いたパッチ数の平方根 = 格子の1辺のサイズ
    H_and_W = int(math.sqrt(pos_embed.shape[1] - 1))

    print(f"pos_embed の形状: {pos_embed.shape}")
    print(f"パッチ数: {H_and_W**2}（{H_and_W}×{H_and_W} の格子）")
    print(f"埋め込み次元数: {pos_embed.shape[2]}")
    return H_and_W, pos_embed


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## コサイン類似度の計算と可視化

    各パッチの位置埋め込みベクトル $\boldsymbol{p}_i$ と、全パッチの位置埋め込みベクトル $\boldsymbol{p}_j$ との
    コサイン類似度を計算します。

    $$\text{sim}(i, j) = \frac{\boldsymbol{p}_i \cdot \boldsymbol{p}_j}{\|\boldsymbol{p}_i\| \|\boldsymbol{p}_j\|}$$

    値は $[-1, 1]$ の範囲となり、**1に近いほど類似**（空間的に近い）、**-1に近いほど非類似**（空間的に遠い）を表します。

    | 関数 | 説明 |
    |------|------|
    | `F.cosine_similarity(a, b, dim=1)` | テンソル `a` と `b` の各ベクトル間のコサイン類似度を計算 |
    | `pos_embed[0, i:i+1]` | i番目のパッチの位置埋め込み（形状 `(1, D)`）|
    | `pos_embed[0, 1:]` | クラストークンを除く全パッチの位置埋め込み（形状 `(N, D)`）|
    | `.reshape((H_and_W, H_and_W))` | 1次元の類似度ベクトルを格子状に整形 |
    | `ax.imshow(sim, ...)` | 類似度マップをヒートマップとして表示 |

    格子の各セルは「**このパッチは他のどのパッチと位置的に近いか**」を表します。
    自分自身との類似度は必ず1（最も明るい）になります。
    """)
    return


@app.cell
def _(F, H_and_W, plt, pos_embed):
    fig, axes = plt.subplots(H_and_W, H_and_W, figsize=(6, 6))

    patch_labels = ["top-left", "top-right", "bottom-left", "bottom-right"]

    for i in range(1, H_and_W**2 + 1):
        # i番目のパッチと全パッチとのコサイン類似度を計算
        sim = F.cosine_similarity(pos_embed[0, i : i + 1], pos_embed[0, 1:], dim=1)
        # 1次元ベクトルをH_and_W × H_and_Wの格子に整形
        sim = sim.reshape((H_and_W, H_and_W)).detach().cpu().numpy()

        row, col = (i - 1) // H_and_W, (i - 1) % H_and_W
        ax = axes[row][col] if H_and_W > 1 else axes
        im = ax.imshow(sim, cmap="viridis", vmin=-1, vmax=1)
        ax.set_title(patch_labels[i - 1], fontsize=9)
        ax.axis("off")

    fig.suptitle("Position Embedding Cosine Similarity", fontsize=13)
    fig.colorbar(im, ax=axes, shrink=0.7, label="cosine similarity")
    plt.tight_layout()
    plt.savefig("position_embedding.png", dpi=150, bbox_inches="tight")
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
