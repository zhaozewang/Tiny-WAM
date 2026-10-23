"""Launch Tiny-WAM scene previews with an Isaac Lab Python environment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from tinywam.preview import main
if __name__ == '__main__':
    main()
