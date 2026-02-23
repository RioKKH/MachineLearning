import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


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


@app.cell
def _(model, nn, torch):
    # 損失関数とオプティマイザ
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return loss_fn, optimizer


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
