"""flybrain: MaleCNS connectome subgraph as a sparse recurrent controller for ViZDoom."""

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("FLYBRAIN_DATA", Path.home() / "flybrain-doom" / "data"))
OUT_DIR = Path(os.environ.get("FLYBRAIN_OUT", Path.home() / "flybrain-doom" / "outputs"))
MALECNS_DIR = DATA_DIR / "malecns_v1"
