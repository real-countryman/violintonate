import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_DIR))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

project = "Violintonate"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

autodoc_member_order = "bysource"
