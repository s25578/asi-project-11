import typer

app = typer.Typer()


@app.command()
def pycaret(data: str = "data/password_data.csv", sample: float = 0.1):
    from passstren.pycaret_model import run_pycaret
    run_pycaret(data, sample)


@app.command()
def optuna(data: str = "data/password_data.csv", sample: float = 0.1, trials: int = 20):
    from passstren.optuna_tuning import run_optuna
    run_optuna(data, sample, trials)


@app.command()
def predict(password: str):
    from passstren.pycaret_model import predict_password
    predict_password(password)


def main():
    app()


if __name__ == "__main__":
    main()
