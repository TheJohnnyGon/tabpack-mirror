# /// script
# requires-python = "==3.12.9"
#
# dependencies = []
# ///
import getpass
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(*args, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(*args, **kwargs, check=True)


def _git_get_current_branch() -> str:
    return (
        run(['git', 'branch', '--show-current'], capture_output=True)
        .stdout.decode('utf-8')
        .strip()
    )


def _try_create_directory(path: str | Path) -> None:
    path = Path(path)
    if path.exists():
        print(f'Already exists: {path}')
    else:
        print(f'Creating a directory: {path}')
        path.mkdir()


def _try_create_symlink(source: str | Path, target: str | Path) -> None:
    source = Path(source)
    target = Path(target)
    if source.exists(follow_symlinks=False):
        print(f'Already exists: {source}')
    else:
        print(f'Creating a symlink: {source} -> {target}')
        source.symlink_to(target, target_is_directory=target.is_dir())


def main() -> None:
    cwd = Path.cwd()

    if not cwd.joinpath('.git').exists():
        raise RuntimeError('The script must be run from the root of the repository')
    if cwd not in Path(__file__).parents:
        raise RuntimeError(
            'The script must be run from the repository where this script is located'
        )
    if _git_get_current_branch() != 'main':
        raise RuntimeError('The script must be run from the main branch')

    is_template = cwd.joinpath('dev/misc/TEMPLATE').exists()
    user = getpass.getuser()
    user_dir = Path(f'dev/users/{user}')
    example_user_dir = user_dir.with_name('example')

    _try_create_symlink(
        '.git/hooks/pre-commit', '../../dev/misc/git-pre-commit-hook.sh'
    )
    _try_create_directory('data')
    _try_create_directory('local')
    if user_dir.exists():
        print(f'Already exists: {user_dir}')
    else:
        # In Template, user directories are git-ignored.
        # However, they are still needed for running the code relying on Nirvana.
        print(f'Creating a directory:  {user_dir}')
        shutil.copytree(example_user_dir, user_dir)
        user_settings_path = user_dir / 'settings.toml'
        user_settings_path.write_text(
            user_settings_path.read_text().replace('{{ username }}', user)
        )

    print('\nConfiguring git')
    # Optimize git for the large number of files.
    run(['git', 'config', 'feature.manyFiles', 'true'])
    if sys.platform in ('darwin', 'win32'):
        run(['git', 'config', 'core.fsmonitor', 'true'])

    print('\nSetting up the Python virtual environment')
    # When this script is run with uv based on PEP 723
    # (i.e. `uv run <path to this script>`, which is usually the case),
    # VIRTUAL_ENV is pointing to a temporary script-specific environment,
    # which confuses uv when running its command in new processes.
    VIRTUAL_ENV = os.environ.pop('VIRTUAL_ENV', None)
    try:
        # `--managed-python` is needed when creating the environment,
        # but is not needed in future `uv run` calls.
        run(['uv', 'sync', '--managed-python'])
    finally:
        if VIRTUAL_ENV is not None:
            os.environ['VIRTUAL_ENV'] = VIRTUAL_ENV
    del VIRTUAL_ENV

    if not is_template and example_user_dir.exists():
        print(
            '\nFeel free to remove the example user directory'
            ' to simplify the access to your user directory in the VSCode UI:'
        )
        print(f'rm -r {example_user_dir.resolve().relative_to(cwd)}')


if __name__ == '__main__':
    main()
