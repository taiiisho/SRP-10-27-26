"""Allow ``python -m handpredict ...`` as an alternative to the ``handpredict`` command."""

import sys

from .cli import main

sys.exit(main())
