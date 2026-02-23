#!/usr/bin/env python
# -*- coding:utf-8 -*-

from sympy import Mul
import torch
import torch.nn as nn
import torch.nn.functional as F


class VitInputLayer(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        emb_dim: int = 384,
        num_patch_row: int = 2,
        image_size: int = 32,
    ):
        """
        引数:
            in_channels: 入力画像のチャンネル数
            emb_dim: 埋め込み後のベクトルの長さ
            num_patch_row: 高さ方向のパッチの数。
                例は2x2であるため、2をデフォルト値とした
            image_size: 入力画像の1辺の大きさ。
                入力画像の高さと幅は同じであると仮定
        """
        super(VitInputLayer, self).__init__()
        self.in_channels = in_channels
        self.emb_dim = emb_dim
        self.num_patch_row = num_patch_row
        self.image_size = image_size

        # パッチの数
        ## 例: 入力画像を2x2のパッチに分ける場合、num_patchは4
        self.num_patch = self.num_patch_row**2

        # パッチの大きさ
        ## 例: 入力画像の1辺の大きさが32の場合、patch_sizeは16
        self.patch_size = int(self.image_size // self.num_patch_row)

        # 入力画像のパッチへの分割 & パッチの埋め込みを一機に行う層
        self.patch_emb_layer = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.emb_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        # クラストークン
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))

        # 位置埋め込み
        ## クラストークンが先頭に結合されている為、
        ## 長さemb_dimの位置埋め込みベクトルを(パッチ数+1)個用意
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_patch + 1, emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        引数:
            X: 入力画像。形状は、(B, C, H, W)
                B: バッチサイズ
                C: チャンネル数
                H: 高さ
                W: 幅
        返り値:
            z_0: ViTへの入力。形状は (B, N, D)。
                B: バッチサイズ
                N: トークン数
                D: 埋め込みベクトルの長さ
        """
        # バッチの埋め込み & flatten
        ## パッチの埋め込み (B, C, H, W) -> (B, D, H/P, W/P)
        ## ここで、Pはパッチ1辺の大きさ
        z_0 = self.patch_emb_layer(x)

        ## パッチのflatten (B, D, H/P, W/P) -> (B, D, Np)
        ## ここでNpパッチの数 (=H x W / P^2)
        z_0 = z_0.flatten(2)

        # 軸の入れ替え (B, D, Np) -> (B, Np, D)
        z_0 = z_0.transpose(1, 2)

        # パッチの埋め込みの先頭にクラストークンを結合 (式4)
        ## (B, Np, D) -> (B, N, D)
        ## N = (Np + 1) であることに留意
        ## また cls_token の形状は (1, 1, D) であるため、
        ## repeat メソッドによって (B, 1, D) に変換してからパッチの埋め込みとの結合を行う
        z_0 = torch.cat([self.cls_token.repeat(repeats=(x.size(0), 1, 1)), z_0], dim=1)

        # 位置埋め込みの加算 [式5]
        ## (B, N, D) -> (B, N, D)
        z_0 = z_0 + self.pos_emb

        return z_0


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_dim: int = 384, head: int = 3, dropout: float = 0.0):
        """
        引数:
            emb_dim: 埋め込み後のベクトルの長さ
            head: ヘッドの数
            dropout: ドロップアウト率
        """
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // head
        self.sqrt_dh = self.head_dim**0.5  # D_hの二乗根。qk^Tを割るための係数

        # 入力をq, k, vに埋め込むための線形層 (式6)
        self.w_q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_k = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_v = nn.Linear(emb_dim, emb_dim, bias=False)

        # 式(7) にはないが、実装ではドロップアウト層も用いる
        self.attn_drop = nn.Dropout(dropout)

        # MHSAの結果を出力ｎ埋め込むための線形層 (式10)
        ## 式10にはないが、実装ではドロップアウト層も用いる
        self.w_o = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.Dropout(dropout))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        引数:
            z: MHSAへの入力。形状は (B, N, D)。
                B: バッチサイズ
                N: トークンの数
                D: ベクトルの長さ

        返り値:
            out: MHSAの出力。形状は (B, N, D)。[式10]
                B: バッチサイズ
                N: トークンの数
                D: 埋め込みベクトルの長さ
        """

        batch_size, num_patch, _ = z.size()

        # 埋め込み 式6
        ## (B, N, D) -> (B, N, D)
        q = self.w_q(z)
        k = self.w_k(z)
        v = self.w_v(z)

        # q, k, vをヘッドに分ける 式10
        ## 先ずベクトルをヘッドの個数(h)に分ける
        ## (B, N, D) -> (B, N, h, D//h)
        q = q.view(batch_size, num_patch, self.head, self.head_dim)
        k = k.view(batch_size, num_patch, self.head, self.head_dim)
        v = v.view(batch_size, num_patch, self.head, self.head_dim)

        ## Self-Attentionが出来るように
        ## (バッチサイズ、ヘッド、トークン数、パッチのベクトル) の形に変更する
        ## (B, N, h, D//h) -> (B, h, N, D//h)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 内積 式7
        ## (B, h, N, D//h) -> (B, h, D//h, N)
        k_T = k.transpose(2, 3)
        ## (B, h, n, D//h) x (B, h, D//h, N) -> (B, h, N, N)
        dots = (q @ k_T) / self.sqrt_dh
        ## 列方向にソフトマックス関数
        attn = F.softmax(dots, dim=-1)
        ## ドロップアウト
        attn = self.attn_drop(attn)
        # 加重和 式8
        ## (B, h, N, N) x (B, h, N, D//h) -> (B, h, N, D//h)
        out = attn @ v
        ## (B, h, N, D//h) -> (B, N, h, D//h)
        out = out.transpose(1, 2)
        ## (B, N, h, D//h) -> (B, N, D)
        out = out.reshape(batch_size, num_patch, self.emb_dim)

        # 出力層 式10
        ## (B, N, D) -> (B, N, D)
        out = self.w_o(out)
        return out


class VitEncoderBlock(nn.Module):
    def __init__(
        self,
        emb_dim: int = 384,
        head: int = 8,
        hidden_dim: int = 384 * 4,
        dropout: float = 0.0,
    ):
        """
        引数:
            emb_dim: 埋め込み後のベクトルの長さ
            head: ヘッドの数
            hidden_dim: Encoder BlockのMLPにおける中間層のベクトルの長さ
                        原論文に従ってemb_dimの4倍をデフォルト値としている
            dropout: ドロップアウト率
        """
        super(VitEncoderBlock, self).__init__()
        # 1つ目のLayer Normalization [2-5-2項]
        self.ln1 = nn.LayerNorm(emb_dim)
        # MHSA [2-4-7項]
        self.mhsa = MultiHeadSelfAttention(
            emb_dim=emb_dim,
            head=head,
            dropout=dropout,
        )
        # 2つ目のLayer Normalization [2-5-2項]
        self.ln2 = nn.LayerNorm(emb_dim)
        # MLP [2-5-3]
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        引数:
            z: Encoder Blockへの入力。形状は(B, N, D
                B: バッチサイズ
                N: トークン数
                D: ベクトルの長さ
        返り値:
            out: Encoder Blockへの出力。形状は (B, N, D) 式10
                B: バッチサイズ
                N: トークンの数
                D: 埋め込みのベクトルの長さ
        """
        # Encoder Blockの前半部分 式12
        out = self.mhsa(self.ln1(z)) + z
        # Encoder Blockの後半部分 式13
        out = self.mlp(self.ln2(out)) + out
        return out


class Vit(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        emb_dim: int = 384,
        num_patch_row: int = 2,
        image_size: int = 32,
        num_blocks: int = 7,
        head: int = 8,
        hidden_dim: int = 384 * 4,
        dropout: float = 0.0,
    ):
        """
        引数:
            in_channels: 入力画像のチャンネル数
            num_classes: 画像分類のクラス数
            emb_dim: 埋め込み後のベクトルの長さ
            num_patch_row: 1辺のパッチの数
            image_size: 入力画像の1辺の大きさ。入力画像の高さと幅は同じであると仮定
            num_blocks: Encoder Blockの数
            head: ヘッドの数
            hidden_dim: Encoder BlockのMLPにおける中間層のベクトルの長さ
            dropout: ドロップアウト率
        """
        super(Vit, self).__init__()
        # Input Layer [2-3]
        self.input_layer = VitInputLayer(
            in_channels, emb_dim, num_patch_row, image_size
        )

        # Encoder。Encoder Blockの多段 [2-5]
        self.encoder = nn.Sequential(
            *[
                VitEncoderBlock(
                    emb_dim=emb_dim, head=head, hidden_dim=hidden_dim, dropout=dropout
                )
                for _ in range(num_blocks)
            ]
        )

        # MLP Head [2-6-1]
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_dim), nn.Linear(emb_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        引数:
            x: ViTへの入力画像。形状は (B, C, H, W)
                B: バッチサイズ
                C: チャンネル数
                H: 高さ
                W: 幅
        返り値:
            out: ViTの出力。形状は (B, M)。式10
                B: バッチサイズ
                M: クラス数
        """
        # Input Layer 式14
        ## (B, C, H, W) -> (B, N, D)
        ## N: トークン数 (=パッチの数+1), D: ベクトルの長さ
        out = self.input_layer(x)

        # Encoder 式15, 16
        ## (B, N, D) -> (B, N, D)
        out = self.encoder(out)

        # クラストークンのみ抜き出す
        ## (B, N, D) -> (B, D)
        cls_token = out[:, 0]

        # MLP Head 式17
        ## (B, D) -> (B, M)
        pred = self.mlp_head(cls_token)
        return pred


if __name__ == "__main__":
    import torch

    num_classes = 10
    batch_size, channel, height, width = 2, 3, 32, 32
    x = torch.randn(batch_size, channel, height, width)
    vit = Vit(in_channels=channel, num_classes=num_classes)
    pred = vit(x)

    # (2, 10) = (B, M) になっている事を確認
    print(pred.shape)

    input_layer = VitInputLayer(num_patch_row=2)
    z_0 = input_layer(x)

    # (2, 5, 384) (=(B, N, D)) になっていることを確認
    print(z_0.shape)

    mhsa = MultiHeadSelfAttention()
    out = mhsa(z_0)

    # (2, 5, 384) = (B, N, D) になっている事を確認
    print(out.shape)

    vit_enc = VitEncoderBlock()
    z_1 = vit_enc(z_0)
    # (2, 5, 384) = (B, N, D) になっている事を確認
    print(z_1.shape)
