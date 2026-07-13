# /// script
# requires-python = "==3.12.9"
#
# dependencies = []
# ///
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from urllib.parse import urlparse

_STYLE_GREEN = '\033[32m'
_STYLE_YELLOW = '\033[33m'
_STYLE_BLUE = '\033[34m'
_STYLE_CYAN = '\033[36m'
_STYLE_BOLD = '\033[1m'
_STYLE_END = '\033[0m'


def bold_green(x) -> str:
    return f'{_STYLE_BOLD}{_STYLE_GREEN}{x}{_STYLE_END}'


def bold_yellow(x) -> str:
    return f'{_STYLE_BOLD}{_STYLE_YELLOW}{x}{_STYLE_END}'


def bold_blue(x) -> str:
    return f'{_STYLE_BOLD}{_STYLE_BLUE}{x}{_STYLE_END}'


def bold_cyan(x) -> str:
    return f'{_STYLE_BOLD}{_STYLE_CYAN}{x}{_STYLE_END}'


def run(*args, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(*args, **kwargs, check=True)


def get_template_url() -> str:
    return (
        run(
            ['git', '-C', os.path.dirname(__file__), 'remote', 'get-url', 'origin'],
            capture_output=True,
        )
        .stdout.decode('utf-8')
        .strip()
    )


def get_repository_name(repository_url: str) -> str:
    return Path(urlparse(repository_url).path).stem


def get_commit_hash(path: str | Path) -> str:
    return (
        run(['git', '-C', str(path), 'rev-parse', 'HEAD'], capture_output=True)
        .stdout.decode()
        .strip()
    )


def find_and_replace(path: Path, old: str, new: str) -> None:
    assert path.exists()
    old_text = path.read_text()
    new_text = old_text.replace(old, new)
    path.write_text(new_text)
    if new_text == old_text:
        raise RuntimeError(f'Failed to update the file: {path}')


def main(repository_dir: str | Path) -> None:
    repository_dir = Path(repository_dir)

    if repository_dir.suffix != '':
        raise ValueError(
            'The repository path must not have extension, however:'
            f' {repository_dir.suffix=}'
        )
    if repository_dir.exists():
        raise RuntimeError(f'The directory already exists: {repository_dir}')
    if not repository_dir.parent.exists():
        raise RuntimeError(
            f'The parent directory does not exist: {repository_dir.resolve().parent}'
        )

    template_url = get_template_url()

    with tempfile.TemporaryDirectory() as tmp:
        print(f'\n{bold_blue("Creating a temporary Template clone")}')
        tmp_dir = Path(tmp).resolve() / get_repository_name(template_url)
        run(['git', 'clone', template_url, str(tmp_dir)])

        pyproject_path = tmp_dir / 'pyproject.toml'
        pyproject = tomllib.loads(pyproject_path.read_text())
        template_project_name = pyproject['project']['name']

        print()
        commit_hash = get_commit_hash(tmp_dir)
        if commit_hash != get_commit_hash(os.path.dirname(__file__)):
            raise RuntimeError(
                'The script must be run from the up-to-date main branch.'
                ' Switch to the main branch, update it and rerun the script.'
            )

        def save_commit_hash(path: Path) -> None:
            print(
                'Saving the template commit hash to'
                f' {bold_cyan(path.relative_to(tmp_dir))}'
            )
            path.write_text(commit_hash)

        save_commit_hash(tmp_dir / 'dev' / 'misc' / f'.{template_project_name}')
        save_commit_hash(tmp_dir / 'dev' / 'misc' / 'latest_update_commit.txt')

        def print_removing(path: Path) -> None:
            print(f'Removing {bold_yellow(path.relative_to(tmp_dir))}')

        def print_creating(path: Path) -> None:
            print(f'Creating {bold_blue(path.relative_to(tmp_dir))}')

        def print_updating(path: Path) -> None:
            print(f'Updating {bold_cyan(path.relative_to(tmp_dir))}')

        for item in [
            tmp_dir / '.git',
            tmp_dir / 'dev' / 'misc' / 'TEMPLATE',
        ]:
            assert item.exists()
            print_removing(item)
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

        dev_readme_template_path = tmp_dir / 'dev' / 'misc' / 'DEV_README_TEMPLATE.md'
        dev_readme_path = tmp_dir / 'dev' / 'README.md'
        print_creating(dev_readme_path)
        shutil.copyfile(dev_readme_template_path, dev_readme_path)
        find_and_replace(dev_readme_path, '{{ commit_hash }}', commit_hash)

        gitignore_path = tmp_dir / '.gitignore'
        print_updating(gitignore_path)
        gitignore_path.write_text(
            # Allow dev/users in the new repository
            '\n'.join(
                x
                for x in gitignore_path.read_text().splitlines()
                if not x.strip('!').startswith('dev/users')
            )
        )

        print_updating(pyproject_path)
        # Patching the TOML file as raw text to preserve comments and formatting.
        find_and_replace(pyproject_path, template_project_name, repository_dir.name)
        find_and_replace(
            pyproject_path,
            f'version = "{pyproject["project"]["version"]}"',
            'version = "0.0.1"',
        )

        print_updating(tmp_dir / 'uv.lock')
        run(['uv', '--project', str(tmp_dir), 'lock'])

        print()
        git_cmd = ['git', '-C', str(tmp_dir)]
        run([*git_cmd, 'init'])
        run([*git_cmd, 'add', '-A'])
        run([*git_cmd, 'commit', '-m', 'Initial commit'], capture_output=True)
        run([*git_cmd, 'branch', '-M', 'main'])

        tmp_dir.rename(repository_dir)

    print()
    print(bold_green('Successfully initialized the repository:'), repository_dir)
    print(f'Check out {bold_blue(f"{repository_dir}/dev/README.md")}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('repository_dir')
    args = vars(parser.parse_args(sys.argv[1:]))
    main(**args)
