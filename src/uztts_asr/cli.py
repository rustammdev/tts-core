import typer

from uztts_asr.evaluate import evaluate
from uztts_asr.hub import app as hub_app
from uztts_asr.prepare import prepare

app = typer.Typer(add_completion=False, no_args_is_help=True)
app.command("prepare")(prepare)
app.command("eval")(evaluate)
app.add_typer(hub_app, name="hub")
