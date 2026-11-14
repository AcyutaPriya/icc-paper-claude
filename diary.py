"""
Trading diary: a human-readable Markdown log, one file per day under diary/.

Each bot run that does real work (market open) appends a short entry: what the day filter said,
account state, every near-setup / skip / AI veto with the reason, and any orders placed.
Runs while the market is closed write nothing. The GitHub workflow commits diary/ to the repo
so the history persists.
"""
import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
DIARY_DIR = Path(__file__).parent / "diary"


class Diary:
    def __init__(self, paper: bool, dry_run: bool):
        self.now = dt.datetime.now(ET)
        self.paper, self.dry_run = paper, dry_run
        self.silent = False          # True -> write nothing (e.g. market closed)
        self.headline: str | None = None
        self.facts: list[str] = []
        self.watch: list[str] = []   # near-setups, skips, AI vetoes, sizing rejects
        self.no_indication: list[str] = []
        self.trades: list[str] = []
        self.eod: list[str] = []

    # --- recording helpers
    def fact(self, text: str):
        self.facts.append(text)

    def note(self, symbol: str, text: str):
        self.watch.append(f"**{symbol}**: {text}")

    def quiet(self, symbol: str):
        self.no_indication.append(symbol)

    def trade(self, text: str):
        self.trades.append(text)

    def result(self, text: str):
        self.headline = text

    # --- output
    def render(self) -> str:
        mode = ("PAPER" if self.paper else "LIVE") + (" · dry run" if self.dry_run else "")
        head = self.headline
        if head is None:
            head = (f"Placed {len(self.trades)} order(s)" if self.trades and not self.dry_run
                    else f"Dry run: {len(self.trades)} order(s) would have been placed" if self.trades
                    else "No trades: nothing qualified this run")
        out = [f"\n## {self.now:%H:%M} ET · {mode}", f"**{head}**\n"]
        out += [f"- {f}" for f in self.facts]
        if self.trades:
            out += ["", "**Trades**"] + [f"- {t}" for t in self.trades]
        if self.watch:
            out += ["", "**Looked at**"] + [f"- {w}" for w in self.watch]
        if self.no_indication:
            out += ["", f"No setup forming: {', '.join(self.no_indication)}"]
        if self.eod:
            out += ["", "**End-of-day snapshot**"] + [f"- {e}" for e in self.eod]
        return "\n".join(out) + "\n"

    def flush(self):
        if self.silent:
            return
        DIARY_DIR.mkdir(exist_ok=True)
        path = DIARY_DIR / f"{self.now:%Y-%m-%d}.md"
        header = f"# Trading diary: {self.now:%A, %B %d, %Y}\n" if not path.exists() else ""
        with open(path, "a") as f:
            f.write(header + self.render())
