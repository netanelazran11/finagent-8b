"""Root conftest — adds scripts/ to sys.path so tests can import agent modules."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
