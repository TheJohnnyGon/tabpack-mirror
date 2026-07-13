import argparse
import shlex
import subprocess
from pathlib import Path

_PROJECT_SPECIFIC_PATHS = (
    '.gitignore',
    'pixi.lock',
    'pyproject.toml',
    'uv.lock',
    'dev/docker/Dockerfile',
    'dev/settings.toml',
)

_STYLE_BOLD = '\033[1m'
_STYLE_YELLOW = '\033[33m'
_STYLE_CYAN = '\033[36m'
_STYLE_END = '\033[0m'


def _bold_cyan(s: str) -> str:
    return f'{_STYLE_BOLD}{_STYLE_CYAN}{s}{_STYLE_END}'


def _bold_yellow(s: str) -> str:
    return f'{_STYLE_BOLD}{_STYLE_YELLOW}{s}{_STYLE_END}'


def main() -> None:
    cwd = Path.cwd()

    # Validate the current working directory.
    if not cwd.joinpath('.git').exists():
        raise RuntimeError('The script must be run from the root of the repository')

    # Validate the script location.
    if cwd not in Path(__file__).parents:
        raise RuntimeError(
            'The script must be run from the repository where this script is located'
        )

    # The script can be run only in project repositories,
    # but not in the Template repository.
    template_marker_path = cwd / 'dev' / 'misc' / 'TEMPLATE'
    assert not template_marker_path.exists(), (
        'The script cannot be run in the YR Template repository'
    )

    # The script can be run only in a clean state without uncommitted changes.
    git_status_output = subprocess.run(
        ['git', 'status', '--porcelain'], capture_output=True, check=True
    ).stdout.decode()
    if git_status_output.strip():
        raise RuntimeError('Commit all changes before running the script')

    # The Template repository must be connected as a remote repository.
    remotes = (
        subprocess.run(['git', 'remote'], capture_output=True, check=True)
        .stdout.decode()
        .strip()
        .splitlines()
    )
    if 'yr-template' not in remotes:
        raise RuntimeError(
            'Add YR Template as a remote repository and fetch the latest updates'
            ' before running the script:'
            '\n\ngit remote add yr-template <YR Template HTTPS URL>'
        )

    # Get the Template commit used for the latest update.
    latest_update_commit_path = cwd / 'dev' / 'misc' / 'latest_update_commit.txt'
    assert latest_update_commit_path.exists(), (
        f'Write the latest update commit to {latest_update_commit_path} before running'
        ' the script.'
    )
    print(f'\n{_bold_cyan(f"Reading {latest_update_commit_path.relative_to(cwd)}")}')
    latest_update_commit = latest_update_commit_path.read_text()
    print(f'Latest update commit: {latest_update_commit}')

    # Get the latest Template commit.
    git_fetch_command = ['git', 'fetch', 'yr-template']
    print(f'\n{_bold_cyan(shlex.join(git_fetch_command))}')
    subprocess.run(git_fetch_command, check=True)
    latest_template_commit = (
        subprocess.run(
            ['git', 'show', 'yr-template/main', '--pretty=format:%H', '--no-patch'],
            capture_output=True,
            check=True,
        )
        .stdout.decode()
        .strip()
    )
    print(f'Latest Template commit: {latest_template_commit}')

    if latest_update_commit == latest_template_commit:
        print('\nThe code is already up-to-date.')
        return

    try:
        # Cherry-pick the new commits from Template.
        git_cherry_pick_command = [
            'git',
            'cherry-pick',
            # The commit range "A..B" does _not_ include A, but _does_ include B.
            f'{latest_update_commit}..{latest_template_commit}',
            # Accumulate changes from all the commits into one set of changes
            # without committing.
            '--no-commit',
            # In case of conflicts, accept all incoming changes,
            # and handle the conflicts later (revert the changes in project-specific
            # files and raise an exception on conflicts in Template files).
            '--strategy-option',
            'theirs',
        ]
        print(f'\n{_bold_cyan(shlex.join(git_cherry_pick_command))}')
        cherry_pick_stdout = (
            subprocess.run(git_cherry_pick_command, capture_output=True, check=True)
            .stdout.decode()
            .strip()
        )

        # Check that there are no conflicts in Template files.
        if cherry_pick_stdout:
            conflict_paths = set()
            conflict_line_prefix = 'Auto-merging '
            for line in cherry_pick_stdout.splitlines():
                if line.startswith(conflict_line_prefix):
                    conflict_paths.add(line.removeprefix(conflict_line_prefix))
                else:
                    # The output must contain nothing but the conflict descriptions:
                    # ```
                    # Auto-merging path/to/file1.txt
                    # Auto-merging path/to/file2.txt
                    # ...
                    # ```
                    # Anything else indicates something unexpected.
                    raise RuntimeError('Unexpected output of git cherry-pick')
            template_conflict_paths = [
                x for x in conflict_paths if x not in _PROJECT_SPECIFIC_PATHS
            ]
            if template_conflict_paths:
                raise RuntimeError(
                    'Conflicts in YR Template files:'
                    f' {", ".join(template_conflict_paths)}'
                )

        if template_marker_path.exists():
            raise RuntimeError(
                'The Template marker was found after the cherry-pick, which should'
                ' never be the case. A potential reason is a wrong commit in the'
                f' following file: {latest_update_commit_path}. Ensure that (1) it'
                'contains a commit hash from the _Template_ repository, not from the'
                ' _project_ repository, and (2) it does not contain a new line in the'
                ' end.'
            )

        # Revert changes in project-specific files.
        changed_paths = frozenset(
            x.split(maxsplit=1)[1]
            for x in (
                subprocess.run(
                    ['git', 'status', '--porcelain'], capture_output=True, check=True
                )
                .stdout.decode()
                .strip()
                .splitlines()
            )
        )
        changed_project_specific_paths = [
            x for x in _PROJECT_SPECIFIC_PATHS if x in changed_paths
        ]
        changed_project_specific_paths_str = '\n'.join(
            map(str, changed_project_specific_paths)
        )
        if changed_project_specific_paths:
            print(f'\n{_bold_cyan("Reverting changes in project-specific files:")}')
            print(changed_project_specific_paths_str)
            # First, unstage the changes.
            subprocess.run(
                ['git', 'restore', '--staged', *changed_project_specific_paths],
                check=True,
            )
            # Then, revert the actual file modifications.
            subprocess.run(
                ['git', 'restore', *changed_project_specific_paths], check=True
            )

        # Save the Template commit used for the update.
        latest_update_commit_path.write_text(latest_template_commit)
        subprocess.run(['git', 'add', str(latest_update_commit_path)], check=True)

        # Finish.
        git_log_command = [
            'git',
            'log',
            '--oneline',
            f'{latest_update_commit}..{latest_template_commit}',
        ]
        print(_bold_cyan(f'\n{shlex.join(git_log_command)}\n'))
        subprocess.run(git_log_command, check=True)

        git_status_command = ['git', 'status']
        print(_bold_cyan(f'\n{shlex.join(git_status_command)}\n'))
        subprocess.run(git_status_command, check=True)

        if changed_project_specific_paths:
            print(
                f'\n{_bold_cyan("NOTE")}'
                '\nThe following files were not updated, because they are'
                ' project-specific (update them manually if needed):'
                f'\n\n{_bold_yellow(changed_project_specific_paths_str)}'
            )
        print(
            f'\n{_bold_cyan("NOTE")}'
            '\nReview the output of `git log` and `git status` shown above, and commit'
            ' the changes.'
            '\n\nTo cancel the update for some of the files, run TWO commands:'
            '\ngit restore --staged path/to/file1 path/to/file2 path/to/dir'
            '\ngit restore path/to/file1 path/to/file2 path/to/dir'
            '\n\nTo fully cancel the update, run:'
            '\ngit reset --hard'
        )

    except Exception as err:
        # If anything goes wrong, abort the cherry-pick and reraise the exception.
        print(f'\n{_bold_yellow("Failed to update the project repository.")}')
        subprocess.run(['git', 'cherry-pick', '--abort'], check=True)
        print(f'{_bold_cyan("All changes were successfully reverted.")}\n')
        raise RuntimeError('Failed to update the project repository') from err


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(**vars(parser.parse_args()))
