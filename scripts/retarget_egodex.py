"""EgoDex geometry retargeting and occlusion experiments, separate from training."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from swam.retargeting.cli import main

if __name__ == '__main__':
    main()
