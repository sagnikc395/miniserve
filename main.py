"""Development entry point: `python main.py <command>`.

The installed console script is `miniserve`, which points at
`miniserve.cli:main`. This wrapper also puts `src/` (and the repository root,
for `demos/`) on `sys.path`, so the repository works without an editable
install.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from miniserve.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
