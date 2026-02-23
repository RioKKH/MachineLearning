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
    # Vision Transformer × FashionMNIST

    [Vision-Transformer入門](https://www.amazon.co.jp/Vision-Transformer%E5%85%A5%E9%96%80-Computer-Library/dp/4297130580)
    のchapter 2で解説されているVision Transformerのコードを用いて、
    PyTorchのQuickstartチュートリアルをベースに、FashionMNISTを学習してみます。

    ## ライブラリのインポート

    | ライブラリ | 説明 |
    |-----------|------|
    | `torch` | PyTorchのコアライブラリ |
    | `torch.nn` | ニューラルネットワークの構成要素（損失関数・レイヤーなど） |
    | `DataLoader` | データセットをバッチ単位で読み込む |
    | `torchvision.datasets` | FashionMNISTなどの標準データセットを提供 |
    | `ToTensor` | 画像をTensorに変換し、ピクセル値を`[0, 1]`に正規化 |
    | `vit.Vit` | 本ディレクトリのVision Transformerモデル |
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from torchvision.transforms import ToTensor

    from vit import Vit

    return DataLoader, ToTensor, Vit, datasets, nn, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## データセットの読み込み

    `datasets.FashionMNIST` を使ってFashionMNISTデータセットをダウンロード・読み込みます。

    **FashionMNIST**: Zalando社が公開したファッション商品の画像データセット。28×28ピクセルのグレースケール画像で、10クラス・70,000枚（訓練60,000枚 + テスト10,000枚）。

    | 引数 | 説明 |
    |------|------|
    | `root="data"` | データの保存先ディレクトリ |
    | `train=True/False` | 訓練データ（`True`）かテストデータ（`False`）かを指定 |
    | `download=True` | ローカルに存在しない場合は自動ダウンロード |
    | `transform=ToTensor()` | 画像をTensorに変換し、ピクセル値を`[0.0, 1.0]`に正規化 |
    """)
    return


@app.cell
def _(ToTensor, datasets):
    # データセットの読み込み
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    return test_data, training_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## DataLoaderの作成

    `DataLoader` はデータセットをミニバッチに分割し、イテレーション可能にします。

    | 引数 | 説明 |
    |------|------|
    | `batch_size=64` | 1ステップで使用する画像の枚数 |
    | `shuffle=True` | 各エポック開始前にデータをシャッフル（訓練時のみ。過学習防止のため） |

    データの形状（最初のバッチで確認）：

    - `X.shape`: `[64, 1, 28, 28]`（バッチサイズ, チャンネル数, 高さ, 幅）
    - `y.shape`: `[64]`（各画像のクラスラベル、整数）
    """)
    return


@app.cell
def _(DataLoader, test_data, training_data):
    # DataLoaderの作成
    batch_size = 64
    train_dataloader = DataLoader(
        training_data, batch_size=batch_size, shuffle=True
    )

    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Sahpe of y: {y.shape}, {y.dtype}")
        break
    return test_dataloader, train_dataloader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## データセットの可視化

    訓練データからランダムに9枚の画像を3×3グリッドで表示します。


    | 関数 | 説明 |
    |------|------|
    | `torch.randint(n, size=(1,))` | 0からn-1の乱数を1つ生成し、ランダムなインデックスを取得 |
    | `img.squeeze()` | Tensorの形状`[1, 28, 28]`からサイズ1の次元を除去して`[28, 28]`に変換 |
    | `figure.add_subplot(rows, cols, i)` | 指定したグリッド位置にサブプロットを追加 |
    | `plt.imshow(..., cmap="gray")` | グレースケールで画像を表示 |
    | `plt.axis("off")` | 軸の目盛りを非表示にする |
    """)
    return


@app.cell
def _(plt, torch, training_data):
    # データセットの可視化
    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }

    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## デバイス設定とモデルの作成

    ### デバイス設定

    `torch.cuda.is_available()` でGPUが利用可能か確認します。利用可能であれば `"cuda"`、そうでなければ `"cpu"` を使用します。

    ### Vitモデルのパラメータ

    FashionMNISTはデフォルト設定（3チャンネル・32×32）と異なるため、以下のように変更します。

    | パラメータ | デフォルト | FashionMNIST用 | 理由 |
    |-----------|-----------|---------------|------|
    | `in_channels` | 3 | 1 | グレースケール画像（1チャンネル） |
    | `image_size` | 32 | 28 | FashionMNISTの画像サイズ |
    | `num_patch_row` | 2 | 2 | 28÷2=14ピクセルのパッチに分割 |
    | `emb_dim` | 384 | 64 | 軽量化 |
    | `num_blocks` | 7 | 3 | 軽量化 |
    | `head` | 8 | 4 | Multi-Head Attentionのヘッド数 |
    | `dropout` | 0.0 | 0.1 | 正則化のため |

    `.to(device)` でモデルを指定デバイスに転送します。
    """)
    return


@app.cell
def _(Vit, torch):
    # デバイス設定 & モデル作成
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

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

    print(model)
    return device, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 損失関数とオプティマイザの定義

    ### `nn.CrossEntropyLoss()`

    多クラス分類の標準的な損失関数。内部で **Softmax** と **NLLLoss（負の対数尤度損失）** の2ステップを組み合わせており、モデルのlogits（生の出力値）をそのまま受け取れます。

    #### ステップ1: Softmax — logitsを確率に変換

    クラス数を $C$、モデルの出力（logits）を $\boldsymbol{z} = (z_1, z_2, \ldots, z_C)$ とすると、各クラスの予測確率は：

    $$p_k = \text{Softmax}(z_k) = \frac{e^{z_k}}{\displaystyle\sum_{j=1}^{C} e^{z_j}}$$

    全クラスの確率の和は1になります（$\sum_{k=1}^{C} p_k = 1$）。

    #### ステップ2: Negative Log-Likelihood — 正解クラスの確率から損失を計算

    正解クラスのラベルを $y$ とすると、1サンプルの損失は：

    $$\ell = -\log p_y$$

    $p_y$ が1（完全に正解）に近いほど損失は0に近づき、$p_y$ が0（完全に不正解）に近いほど損失は大きくなります。

    #### 合わせるとCross-Entropy Loss

    バッチサイズ $N$ のミニバッチ全体の平均損失：

    $$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{z_{i,y_i}}}{\displaystyle\sum_{j=1}^{C} e^{z_{i,j}}}$$

    | 記号 | 意味 |
    |------|------|
    | $N$ | バッチサイズ（ここでは64） |
    | $C$ | クラス数（FashionMNISTでは10） |
    | $z_{i,j}$ | $i$番目のサンプルの$j$番目クラスに対するlogits |
    | $y_i$ | $i$番目のサンプルの正解クラスラベル |

    > **Note**: PyTorchの `nn.CrossEntropyLoss` はSoftmaxを内部で処理するため、モデルの出力にSoftmaxを適用する必要はありません。また、Softmaxを分離せず一体で計算する方が数値的に安定します（[log-sum-exp trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/)）。

    ### `torch.optim.Adam`

    適応的な学習率を持つ最適化アルゴリズム。パラメータごとに学習率を自動調整するため、SGDより学習が安定しやすく、Transformerとの相性が良いため、チュートリアルのSGDの代わりに採用します。

    | 引数 | 説明 |
    |------|------|
    | `model.parameters()` | 学習対象のパラメータ（重み・バイアス）をすべて渡す |
    | `lr=1e-3` | 初期学習率（0.001） |
    """)
    return


@app.cell
def _(model, nn, torch):
    # 損失関数とオプティマイザ
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return loss_fn, optimizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 学習関数

    1エポック分の学習を行う`train`関数を定義します。各バッチに対して「順伝播 → 損失計算 → 逆伝播 → パラメータ更新」を繰り返します。

    | 処理 | 関数 | 説明 |
    |------|------|------|
    | 学習モードに切替 | `model.train()` | Dropoutを学習時の動作（確率的に無効化）にする |
    | デバイスへ転送 | `.to(device)` | データをGPU/CPUに転送 |
    | 順伝播 | `model(X)` | 予測値（logits）を計算 |
    | 損失計算 | `loss_fn(pred, y)` | 予測と正解ラベルの誤差を計算 |
    | 逆伝播 | `loss.backward()` | 各パラメータに対する勾配を計算 |
    | パラメータ更新 | `optimizer.step()` | 勾配に基づいてパラメータを更新 |
    | 勾配リセット | `optimizer.zero_grad()` | 次のステップのために勾配を0にリセット |
    """)
    return


@app.cell
def _(device):
    # 学習関数
    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss_val, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss_val:>7f} [{current:>5d}/{size:>5d}]")

    return (train,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## テスト関数

    学習済みモデルをテストデータで評価する`test`関数を定義します。評価時は勾配計算が不要なため、`torch.no_grad()` で無効化してメモリと計算量を節約します。

    | 処理 | 関数 | 説明 |
    |------|------|------|
    | 評価モードに切替 | `model.eval()` | Dropoutを無効化し、推論時の動作にする |
    | 勾配計算を無効化 | `torch.no_grad()` | メモリ節約・推論の高速化（勾配グラフを構築しない） |
    | クラス予測 | `pred.argmax(1)` | 各サンプルのlogitsのうち最大値のインデックス（予測クラス）を取得 |
    | 正解数カウント | `.type(torch.float).sum()` | 正解したサンプル数を浮動小数点として合計 |
    """)
    return


@app.cell
def _(device, torch):
    # テスト関数
    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(
            f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}\n"
        )

    return (test,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 学習ループの実行

    `train` と `test` を `epochs` 回繰り返します。

    - **1エポック** = 訓練データ全体を1周して学習し、テストデータで評価すること
    - 各エポック終了後に精度と損失を表示し、モデルの改善を追跡します
    """)
    return


@app.cell
def _(
    loss_fn,
    model,
    optimizer,
    test,
    test_dataloader,
    train,
    train_dataloader,
):
    # 学習ループの実行
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t + 1}\n---------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## モデルの保存

    学習したモデルのパラメータをファイルに保存します。

    | 関数 | 説明 |
    |------|------|
    | `model.state_dict()` | モデルの全パラメータ（重み・バイアス）をPython辞書として返す |
    | `torch.save(obj, path)` | Pythonオブジェクトをファイルに保存（内部でpickleを使用） |

    保存したモデルは後から以下のように読み込めます：
    ```python
    model.load_state_dict(torch.load("vit_fashionmnist.pth", weights_only=True))
    ```
    """)
    return


@app.cell
def _(model, torch):
    # モデルの保存
    torch.save(model.state_dict(), "vit_fashionmnist.pth")
    print("Saved PyTorch Model State to vit_fashionmnist.pth")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
