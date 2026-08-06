import typer

from uztts_asr.evaluate import evaluate
from uztts_asr.hub import app as hub_app
from uztts_asr.prepare import prepare
from uztts_asr.train import train

app = typer.Typer(add_completion=False, no_args_is_help=True)
app.command("prepare")(prepare)
app.command("eval")(evaluate)
app.command("train")(train)
app.add_typer(hub_app, name="hub")
