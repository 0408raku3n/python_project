"""Минимальная точка входа."""

from importlib.metadata import version

import typer

app = typer.Typer(add_completion=False)


def show_version(value: bool) -> None:
    if value:
        typer.echo(version("semantic-knowledge"))
        raise typer.Exit()


@app.callback()
def main(
    version_flag: bool | None = typer.Option(
        None, "--version", callback=show_version, is_eager=True, help="Показать версию."
    ),
) -> None:
    """Работа с семантической моделью."""
