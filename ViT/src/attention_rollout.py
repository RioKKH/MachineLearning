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
    # 3-7: Attention Rollout による判断根拠の可視化

    ViTのSelf-AttentionのAttention Weightを層をまたいで積算することで、
    入力画像のどのパッチにモデルが注目しているか（判断根拠）を可視化します。

    ## アルゴリズムの概要

    各層のAttention Weightを $A_h^b$（$b$: 層、$h$: ヘッド）として、以下の手順でAttention Mapを生成します。

    $$
    \hat{A}^b = \frac{1}{H}\sum_{h=1}^{H} A_h^b \quad \xrightarrow{+I} \quad
    \hat{A}^b + I \quad \xrightarrow{\text{normalize}} \quad
    \bar{A} = \prod_b \hat{A}^b
    $$

    最終的に $\bar{A}$ のクラストークン行からパッチトークンへのAttention Weightを取り出し、
    画像サイズにリサイズしてAttention Mapとします。

    ## ライブラリのインポート

    | ライブラリ | 説明 |
    |-----------|------|
    | `numpy` | Attention Weightの行列演算（平均・積算・正規化） |
    | `torch.nn.functional` | マスクの画像サイズへのリサイズ（`F.interpolate`） |
    | `torchvision.datasets` | FashionMNISTテストデータの読み込み |
    | `vit.Vit` | 学習済みモデルのクラス定義 |
    """)
    return


@app.cell
def _():
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    from torchvision import datasets
    from torchvision.transforms import ToTensor

    from vit import Vit

    return F, ToTensor, Vit, datasets, math, np, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## モデルの準備

    `fashionMNIST.py` で保存した学習済みの重みファイル `vit_fashionmnist.pth` を読み込みます。

    | 操作 | 関数 | 説明 |
    |------|------|------|
    | 重みの読み込み | `torch.load(path, map_location=device)` | GPU学習済みモデルをCPU環境でも読み込める |
    | 重みの適用 | `model.load_state_dict(state_dict)` | 保存されたパラメータをモデルに反映 |
    | 評価モード | `model.eval()` | Dropoutを無効化し推論モードにする |

    > **注意**: `vit_fashionmnist.pth` が存在しない場合は、先に `fashionMNIST.py` でモデルを学習・保存してください。
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
    ## テストデータの準備

    可視化対象の画像をFashionMNISTテストセットから取得します。

    | ID | ラベル | ID | ラベル |
    |----|--------|----|--------|
    | 0 | T-Shirt | 5 | Sandal |
    | 1 | Trouser | 6 | Shirt |
    | 2 | Pullover | 7 | Sneaker |
    | 3 | Dress | 8 | Bag |
    | 4 | Coat | 9 | Ankle Boot |
    """)
    return


@app.cell
def _(ToTensor, datasets):
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    labels_map = {
        0: "T-Shirt", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
        5: "Sandal",  6: "Shirt",   7: "Sneaker",  8: "Bag",   9: "Ankle Boot",
    }
    print(f"テストデータ数: {len(test_data)}")
    return labels_map, test_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## forward hook による Attention Weight の取得と Attention Rollout の計算

    ### forward hook とは

    指定したモジュールの順伝播が実行されるたびに自動的に呼び出されるコールバック関数です。
    `register_forward_hook(fn)` で登録し、推論後は `handle.remove()` で必ず解除します。

    ```
    model(x) を実行
      └─ encoder[0].mhsa.attn_drop が実行される
           └─ hook(module, input, output) が呼ばれる  ← output = Attention Weight (B, H, N, N)
      └─ encoder[1].mhsa.attn_drop が実行される ...
    ```

    書籍コード（`ch3/attention_rollout.py`）との対応：

    | 書籍 | 本実装 |
    |------|--------|
    | `model.blocks[i].attn.attn_drop` | `model.encoder[i].mhsa.attn_drop` |
    | `cv2.resize` でマスクをリサイズ | `F.interpolate` でリサイズ |

    ### 計算ステップと形状の変化

    | ステップ | 処理 | 形状 |
    |---------|------|------|
    | 収集 | 各層の Attn Weight を hook で取得 | L × `(1, 4, 5, 5)` |
    | スタック | `np.array(...)[:, 0]` で B 次元を除去 | `(3, 4, 5, 5)` |
    | ヘッド平均 | `np.mean(axis=1)` | `(3, 5, 5)` |
    | 単位行列加算 | `+ np.eye(5)` （残差接続の考慮） | `(3, 5, 5)` |
    | 正規化 | 各行列を合計値で割る | `(3, 5, 5)` |
    | 層積算 | `np.matmul` で最終層から順に乗算 | `(5, 5)` |
    | マスク抽出 | `v[0, 1:]`（cls トークン行・パッチ列のみ） | `(4,)` |
    | 整形 | `reshape(2, 2)` | `(2, 2)` |
    | リサイズ | `F.interpolate(..., size=(28, 28))` | `(28, 28)` |
    """)
    return


@app.cell
def _(F, device, math, model, np, torch):
    def get_attention_rollout(img_tensor):
        """
        1枚の画像に対してAttention Rolloutを計算する。

        引数:
            img_tensor: 入力画像 shape (1, C, H, W)
        返り値:
            mask_resized: Attention Map shape (H, W)、値域 [0, 1]
        """
        attention_weights = []

        # hookコールバック: attn_drop の出力（Attention Weight）を記録する
        def make_hook():
            def hook(module, input, output):
                # output shape: (B, H, N, N)
                attention_weights.append(output.detach().cpu().numpy())
            return hook

        # 各 Encoder ブロックの MHSA 内 attn_drop に hook を登録
        handles = []
        for i in range(len(model.encoder)):
            handle = model.encoder[i].mhsa.attn_drop.register_forward_hook(
                make_hook()
            )
            handles.append(handle)

        # 推論（hook が発動し L 層分の Attention Weight が記録される）
        with torch.no_grad():
            _ = model(img_tensor.to(device))

        # hook を解除（推論後は必ず解除する）
        for handle in handles:
            handle.remove()

        # (L, B, H, N, N) -> (L, H, N, N)  ※ B=1 なので [:, 0] で B 次元を除去
        attn = np.array(attention_weights)[:, 0]

        # ---- Attention Rollout の計算 ----

        # Step 1: ヘッド方向に平均 -> (L, N, N)
        mean_head = np.mean(attn, axis=1)

        # Step 2: 単位行列を加算（残差接続を考慮）
        mean_head = mean_head + np.eye(mean_head.shape[1])

        # Step 3: 正規化（各層の行列を合計値で割る）
        mean_head = mean_head / mean_head.sum(axis=(1, 2))[:, np.newaxis, np.newaxis]

        # Step 4: 層方向に積算（最終層から初層へ向けて乗算）
        v = mean_head[-1]
        for n in range(1, len(mean_head)):
            v = np.matmul(v, mean_head[-1 - n])

        # Step 5: クラストークン行 (row=0) からパッチトークン列 (col=1:) を抽出
        mask = v[0, 1:]  # shape: (num_patch,) = (4,)

        # Step 6: H_and_W × H_and_W の格子に整形
        H_and_W = int(math.sqrt(len(mask)))
        mask = mask.reshape(H_and_W, H_and_W)  # shape: (2, 2)

        # Step 7: 画像サイズ (28×28) にバイリニア補間でリサイズ
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
        mask_resized = F.interpolate(
            mask_tensor, size=(28, 28), mode="bilinear", align_corners=False
        ).squeeze().numpy()

        # Step 8: [0, 1] に正規化
        mask_min, mask_max = mask_resized.min(), mask_resized.max()
        mask_resized = (mask_resized - mask_min) / (mask_max - mask_min + 1e-8)

        return mask_resized

    return (get_attention_rollout,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 可視化

    各クラスから1枚ずつサンプル画像を選び、Attention Rolloutを可視化します。

    | 行 | 内容 |
    |----|------|
    | Original | 入力画像（グレースケール） |
    | Attention Map | クラストークンから各パッチへの Attention 強度（jet カラーマップ） |
    | Overlay | 元画像と Attention Map を重ねた合成画像（`alpha=0.5`） |

    **見方**: 赤〜黄色の部分がモデルの注目箇所、青色の部分が注目していない箇所です。
    3-6の位置埋め込みの可視化がモデルの静的な位置情報を表すのに対し、
    こちらは**入力画像に依存した動的な注目マップ**となります。
    """)
    return


@app.cell
def _(get_attention_rollout, labels_map, plt, test_data):
    # 各クラスから最初の1枚を取得（クラス0〜9 の順）
    sample_indices = []
    found = set()
    for idx in range(len(test_data)):
        _, label = test_data[idx]
        if label not in found:
            sample_indices.append(idx)
            found.add(label)
        if len(found) == 10:
            break

    num_samples = len(sample_indices)
    fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 1.8, 6))

    # 行ラベル
    for row, label_text in enumerate(["Original", "Attention\nMap", "Overlay"]):
        axes[row, 0].set_ylabel(label_text, fontsize=9)

    for col, idx in enumerate(sample_indices):
        img, label = test_data[idx]
        img_tensor = img.unsqueeze(0)       # (1, 1, 28, 28)
        img_np = img.squeeze().numpy()      # (28, 28)

        # Attention Rollout の計算
        mask = get_attention_rollout(img_tensor)

        # 行1: 元画像（グレースケール）
        axes[0, col].imshow(img_np, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(labels_map[label], fontsize=8)
        axes[0, col].axis("off")

        # 行2: Attention Map（jet カラーマップ）
        axes[1, col].imshow(mask, cmap="jet", vmin=0, vmax=1)
        axes[1, col].axis("off")

        # 行3: オーバーレイ（元画像 + Attention Map を alpha ブレンド）
        axes[2, col].imshow(img_np, cmap="gray", vmin=0, vmax=1)
        axes[2, col].imshow(mask, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        axes[2, col].axis("off")

    fig.suptitle("Attention Rollout — FashionMNIST（各クラス1枚）", fontsize=12)
    plt.tight_layout()
    plt.savefig("attention_rollout.png", dpi=150, bbox_inches="tight")
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
