"""Make the repository root importable without installing the package.

The `miniserve/` package lives at the project root, next to this `tests/`
directory. pytest imports this file before collecting any test module, so
every test can simply `from miniserve... import ...` without any install.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
