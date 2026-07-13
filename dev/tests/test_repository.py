# NOTE
# This script should run as fast as possible, because it is used in the pre-commit hook.
# This is why it is run as a standalone script outside of the main project.
#
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

import subprocess
from pathlib import Path


def test_gitignore():
    branch = (
        subprocess.run(
            ['git', 'branch', '--show-current'], check=True, capture_output=True
        )
        .stdout.decode('utf-8')
        .strip()
    )

    # The following condition covers main-like branches, e.g. "main", "main-v2", etc.
    # if branch.lstrip().lower().startswith(('main', 'master')):
    #     exp_rules = {
    #         x
    #         for x in Path('.gitignore').read_text().splitlines()
    #         if 'exp' in x and not x.startswith('#')
    #     }
    #     assert exp_rules == {'**/exp/**/*.*'}, (
    #         'In main-like branches, experiment files are not allowed,'
    #         ' with the only exception being manually added files in exp/examples'
    #     )


if __name__ == '__main__':
    test_gitignore()
