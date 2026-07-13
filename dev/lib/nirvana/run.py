import functools
import itertools
import os
import shlex
import subprocess
import tomllib
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlparse

import cyclopts
import vh3

import dev.lib.env
import dev.lib.nirvana.operations
import lib.util

_REPOSITORY_PREFIX = 'git@'
_REPOSITORY_SUFFIX = '.git'


@functools.lru_cache(1)
def _get_repository() -> str:
    error_message = 'Failed to infer the repository'
    try:
        url = (
            subprocess.run(
                ['git', 'remote', 'get-url', 'origin'], capture_output=True, check=True
            )
            .stdout.decode('utf-8')
            .strip()
        )
    except Exception as err:
        raise RuntimeError(error_message) from err
    else:
        if not url:
            raise RuntimeError(error_message)

    if url.startswith(_REPOSITORY_PREFIX) and url.endswith(_REPOSITORY_SUFFIX):
        return url
    else:
        parsed_url = urlparse(url)
        repository = f'{_REPOSITORY_PREFIX}{parsed_url.netloc}:{parsed_url.path}'
        if not repository.endswith(_REPOSITORY_SUFFIX):
            repository += _REPOSITORY_SUFFIX
        return repository


def _get_repository_name(repository: str) -> str:
    return repository.rsplit('/', 1)[1].removesuffix('.git')


def _prepare_commands(
    main_commands: str,
    *,
    update_environment_variables: None | dict[str, None | str],
    data_commands: None | str,
    venv_commands: None | str,
    n_gpus: int,
    repository_name: str,
) -> str:
    hline = '─' * 100
    commands = [
        'echo "Setting environment variables"',
        *(
            # NOTE
            # If quoting is needed, then `shlex.quote` must be applied
            # in the dictionary, NOT in the following f-string.
            f'unset {k}' if v is None else f'export {k}={v}'
            for k, v in {
                'KMP_AFFINITY': None,
                'UV_FROZEN': '1',
                'UV_MANAGED_PYTHON': '1',
                'UV_NO_CACHE': '1',
                'UV_OFFLINE': '1',
                'CUDA_VISIBLE_DEVICES': ','.join(map(str, range(n_gpus))),
                'PROJECT_DIR': f'${{SOURCE_CODE_PATH}}/{repository_name}',
                **(
                    {}
                    if update_environment_variables is None
                    else update_environment_variables
                ),
            }.items()
        ),
        '',
        'echo "Changing the working directory to $PROJECT_DIR"',
        'cd $PROJECT_DIR',
        '',
        'echo "Setting up the project"',
        'mv $INPUT_PATH/data $PROJECT_DIR' if data_commands is None else data_commands,
        'uv sync --no-build-isolation' if venv_commands is None else venv_commands,
        '',
        'echo "Printing system information"',
        'echoerr() { echo "$@" 1>&2; }',
        *itertools.chain.from_iterable(
            [
                f'echoerr "{hline}"',
                f'echoerr {shlex.quote(cmd)}',
                f'echoerr "{hline}"',
                f'{cmd} 1>&2',
            ]
            for cmd in [
                'printenv',
                'pwd',
                'ls',
                'lscpu',
                'nvidia-smi',
                'uv run which python',
                'VIRTUAL_ENV=$UV_PROJECT_ENVIRONMENT uv pip freeze',
            ]
        ),
        '',
        'echo',
        'uv run dev/bin/nirvana_start.py',
        '',
        'echo',
        main_commands,
    ]
    return '\n'.join(commands)


DataId = Literal[
    # terra.vla.yp-c.yandex.net:/mnt/tabular/data/mango/v4
    'f5418e18-8f57-4fb6-8f71-d5719be898e6',
    #
    # terra.vla.yp-c.yandex.net:/mnt/tabular/data/mango/v5
    '5629ca3f-00bf-4844-b6f7-54696c43e5be',
    #
    # terra.vla.yp-c.yandex.net:/mnt/tabular/data/common/v5
    '89e7ab8f-1441-4bd9-9707-ca62eb56ad79',
    #
    # terra.vla.yp-c.yandex.net:/mnt/tabular/data/common-raw/v5
    '3b067cae-bbdc-4a01-ab8d-75290f02dbb8',
]
LayerId = Literal[
    # yr-template:0.4.0
    'd56fa5be-5296-453b-a587-b8a65cf6c2c3',
    # yr/template:0.5.0
    'fba14043-7963-4670-99c6-eef390a86122',
    # yr/template:0.6.0
    '56d96d9c-557f-49e7-ad0e-349eafcb55d8',
]


def run_pydl_graph(
    main_commands: None | str = None,
    *,
    data_commands: None | str = None,
    venv_commands: None | str = None,
    update_environment_variables: None | dict[str, None | str] = None,
    commands: None | str = None,
    #
    workflow_id: str,
    data_ids: list[str],
    layer_id: str,
    comment: None | str = None,
    #
    n_gpus: int,
    n_cpu_cores: int,
    disk_gb: None | int = None,
    memory_gb: None | int = None,
    gpu_memory_gb: None | int = None,
    ttl_hours: None | int = None,
    priority: Literal['low', 'normal', 'high'] = 'normal',
    #
    quota: str,
    job_scheduler_instance: str,
    job_scheduler_yt_pool: str,
    pool_tree: list[str],
    #
    pulsar_token_secret: str,
    yt_token_secret: str,
    ssh_key_secret: str,
    #
    verbose: bool = True,
    interactive: Annotated[bool, cyclopts.Parameter(alias='-i')] = False,
) -> None:
    """Run a Python 3 Deep Learning operation on Nirvana."""
    if (main_commands is None and commands is None) or (
        main_commands is not None and commands is not None
    ):
        raise ValueError(
            'Exactly one of `main_commands` and `commands` must be provided'
        )
    if commands is not None and data_commands is not None:
        raise ValueError(
            'When `commands` are provided, `data_commands` must not be provided'
        )
    if n_gpus < 0:
        raise ValueError(
            f'n_gpus must be a non-negative integer. The provided value: {n_gpus=}'
        )
    if n_cpu_cores is not None and not 1 <= n_cpu_cores < 255:
        raise ValueError(
            'n_cpu_cores must be in the range [1, 255].'
            f' The provided value: {n_cpu_cores=}'
        )

    repository = _get_repository()
    # NOTE
    # Caching repository_branch may be unsafe if graphs are launched from a long-running
    # interpreter session (e.g. Jupyter) and branches change.
    repository_branch = lib.util.git_get_current_branch()
    repository_name = _get_repository_name(repository)

    # NOTE
    # How `commands` are built by default
    # is the most project-specific part of this function.
    if commands is None:
        assert main_commands is not None  # Already checked.
        commands = _prepare_commands(
            main_commands,
            update_environment_variables=update_environment_variables,
            data_commands=data_commands,
            venv_commands=venv_commands,
            repository_name=repository_name,
            n_gpus=n_gpus,
        )

    if verbose or interactive:
        try:
            terminal_size = os.get_terminal_size().columns
        except OSError:  # Jupyter
            terminal_size = 80
        print('─' * terminal_size)
        print(f'Comment: {"<no comment>" if comment is None else comment}')
        print(f'Priority: {priority}')
        print('─' * terminal_size)
        if main_commands is not None:
            print(main_commands)

    if interactive:
        print('\nRun? [y/N] ', end='')
        if input().strip() != 'y':
            return

    quota_ = quota  # A hack for using the variable below.

    class Context(vh3.BaseContext):
        quota: str = quota_  # type: ignore
        execution_params = vh3.WorkflowExecutionParams(workflow_priority=priority)
        yt_token: vh3.Secret = yt_token_secret
        workflow = vh3.Workflow(id=workflow_id)

    del quota_

    pydl_kwargs = {}
    if memory_gb is not None:
        pydl_kwargs['max_ram'] = memory_gb * 1024
    if gpu_memory_gb is not None:
        pydl_kwargs['gpu_max_ram'] = gpu_memory_gb * 1024
    if disk_gb is not None:
        pydl_kwargs['max_disk'] = disk_gb * 1024
    if ttl_hours is not None:
        pydl_kwargs['ttl'] = ttl_hours * 60

    with (
        vh3.Profile(Context)
        .load_defaults()
        .build(
            vh3.WorkflowInstance,
            **({} if comment is None else {'comment': comment}),
        ) as workflow_instance
    ):
        dev.lib.nirvana.operations.python_3_deep_learning(
            # The order of arguments follows the function python_3_deep_learning.
            pulsar_token=pulsar_token_secret,
            script=dev.lib.nirvana.operations.clone_a_repo_ext(
                repo=repository,
                commit_or_branch=repository_branch,
                folder_name=repository_name,
                key=ssh_key_secret,
            ),
            data=[vh3.data(x) for x in data_ids],  # type: ignore
            image=[layer_id],
            pool_tree=pool_tree,  # type: ignore
            gpu_count=n_gpus,
            run_command=commands,
            cpu_cores_usage=n_cpu_cores * 100,
            **pydl_kwargs,
            job_scheduler_instance=job_scheduler_instance,
            job_scheduler_yt_pool=job_scheduler_yt_pool,
        )
    print()
    workflow_instance.run()


def run_pydl_graphs_from_toml(path: str | Path) -> None:
    defaults = (
        dev.lib.env.load_dev_settings()['nirvana']['run']
        | dev.lib.env.load_user_settings()['nirvana']['run']
    )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'File {path} does not exist')

    with open(path, 'rb') as f:
        graphs = tomllib.load(f)['graphs']
        del f

    n_graphs = len(graphs)
    print(
        f'Running {n_graphs} Python 3 Deep Learning graph{"" if n_graphs == 1 else "s"}'
    )
    for i, graph in enumerate(graphs):
        print(f'\n[{i + 1}/{n_graphs}]')
        run_pydl_graph(**(defaults | graph))


if __name__ == '__main__':
    lib.util.init(torch_=False)

    app = cyclopts.App(
        config=[
            cyclopts.config.Toml(
                dev.lib.env.get_dev_settings_path(),
                ['nirvana', 'run'],
                must_exist=True,
                use_commands_as_keys=False,
            ),
            cyclopts.config.Toml(
                dev.lib.env.get_user_settings_path(),
                ['nirvana', 'run'],
                must_exist=True,
                use_commands_as_keys=False,
            ),
        ]
    )
    app.default(run_pydl_graph)
    app.command(run_pydl_graphs_from_toml, 'from-toml', config=[])
    app()
