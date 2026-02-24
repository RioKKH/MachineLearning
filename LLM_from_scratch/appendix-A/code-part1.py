import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # A.1 What is PyTorch
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import torch

    print(torch.__version__)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
