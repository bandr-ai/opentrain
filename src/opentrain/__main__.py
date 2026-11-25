"""``python -m opentrain``: the same as the ``opentrain`` command."""

import sys

from .cli import main

sys.exit(main())
