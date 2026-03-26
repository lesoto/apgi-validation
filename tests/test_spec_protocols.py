import sys
from pathlib import Path

# Add subdirectories to path for module imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "Theory"))
sys.path.insert(0, str(Path(__file__).parent.parent / "Falsification"))
