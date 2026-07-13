"""Developer tools.

This module is supposed to be used by authors in their private scripts and notebooks.
This module must not be imported from the main lib/ and bin/.
"""

# Check that the main imports work and protect from importing dev from lib.
import lib

del lib
