import contextlib
import json
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Annotated, Any

import cyclopts
import nirvana_api
import urllib3
from loguru import logger
from tqdm.std import tqdm

import dev.lib.env
import lib.util

_MIN_REQUIRED_DISK_SPACE_GB = 50
_NIRVANA_CLIENT = None


def _get_nirvana_client() -> nirvana_api.NirvanaApi:
    global _NIRVANA_CLIENT
    if _NIRVANA_CLIENT is None:
        token_path = Path.home() / '.nirvana' / 'token'
        if not token_path.exists():
            raise FileNotFoundError(
                f'The Nirvana token file is missing: "{token_path}"'
            )

        _NIRVANA_CLIENT = nirvana_api.NirvanaApi(
            token_path.read_text(), ssl_verify=False
        )
    return _NIRVANA_CLIENT


def _check_log_path(path: str | Path) -> Path:
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f'The log file {path} does not exist')
    return path


def _check_output_path(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_dir():
        raise RuntimeError(f'The output {path} must be an existing directory')
    return path


def _nirvana_get_instance_results(
    instance_id: str, code_block_pattern: str
) -> None | dict[str, str]:
    results = _get_nirvana_client().get_block_results(
        block_patterns={'code': code_block_pattern},
        workflow_instance_id=instance_id,
    )
    if not results:
        return None
    # x['endpoint'] is the output name (e.g. "data")
    # x['storagePath'] is URL.
    # Thus, the return type is dict[output_name, url]
    return {x['endpoint']: x['directStoragePath'] for x in results[0]['results']}


def _fetch_new_completed_instances(
    workflow_id: str,
    instance_ids: None | list[str],
    log_path: str | Path,
    *,
    code_block_pattern: str,
) -> list[dict[str, Any]]:
    log_path = _check_log_path(log_path)

    with open(log_path) as f:
        log = json.load(f)
        del f
    if instance_ids is None:
        instance_ids = [
            x['instanceId'] for x in _get_nirvana_client().find_workflows(workflow_id)
        ]

    instances = []
    print()
    for instance_id in tqdm(instance_ids, desc='Fetching new completed instances'):
        if instance_id in log.get(workflow_id, []):
            # Skip already downloaded instances.
            continue

        instance = vars(
            _get_nirvana_client().get_workflow_meta_data(
                workflow_instance_id=instance_id
            )
        )
        if instance['archived']:
            # Skip archived instances.
            continue
        if 'completed' not in instance:
            # Skip instances that are not yet ready.
            continue

        comment = instance.get('instanceComment', '')
        comment = ''.join(x if x.isalnum() else '_' for x in comment)
        instance['_fullname'] = f'{comment}:{instance_id}'
        instance['_result'] = _nirvana_get_instance_results(
            instance_id, code_block_pattern
        )
        instances.append(instance)

    instances.sort(key=lambda x: x['_fullname'])
    return instances


def _download_python_3_deep_learning_data_output(
    workflow_id: str,
    instance: dict[str, Any],
    output: str | Path,
    log_path: str | Path,
    unpack: bool,
) -> None:
    with contextlib.closing(
        urllib.request.urlopen(instance['_result']['json_output'])
    ) as f:
        json_output_text = f.read()
        del f
    json_output = json.loads(json_output_text)
    instance_branch = json_output['branch']
    branch = lib.util.git_get_current_branch()
    if branch != instance_branch:
        logger.warning(
            f'Skipping the instance, because the current active branch is "{branch}",'
            f' while the instance branch is "{instance_branch}"'
        )
        return

    output = _check_output_path(output).resolve()
    log_path = _check_log_path(log_path)

    with open(log_path) as f:
        log = json.load(f)
        del f
    fullname = instance['_fullname']
    workflow_id = instance['guid']
    instance_id = instance['instanceId']
    assert instance_id not in log.get(workflow_id, [])

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        archive_path = tmpdir.joinpath(fullname).with_suffix('.tar')
        try:
            subprocess.run(
                [
                    'wget',
                    instance['_result']['data'],
                    '-O',
                    str(archive_path),
                    '-q',
                    '--show-progress',
                ],
                check=True,
            )
        except Exception as err:
            try:
                archive_path.unlink()
            except Exception:
                pass
            print(f'FAILED to download the results (reason: {err})')
            return

        if unpack:
            with tarfile.open(archive_path) as archive:
                # In Python 3.14, the following line should be changed to:
                # archive.extractall(output)
                archive.extractall(output, filter='data')
        else:
            archive_path.rename(output / archive_path.name)

    log.setdefault(workflow_id, []).append(instance_id)
    log_path.write_text(json.dumps(log, indent=4))


def download(
    *,
    workflow_id: str,
    instance_ids: None | list[str] = None,
    output: None | str | Path = None,
    log_path: str | Path,
    unpack: bool = True,
    interactive: Annotated[
        bool, cyclopts.Parameter(name=['-i', '--interactive'])
    ] = False,
) -> None:
    """Download the "data" output of Python 3 Deep Learning Nirvana operations.

    Args:
        workflow_id: the workflow identifier.
        instance_ids: the instance identifiers.
            If passed, outputs of only these instances will be downloaded.
            Otherwise, of all not-yet-downloaded instances.
        output: where to download the results. Must be an existing directory.
            By default, it is a current working directory.
        log_path: the path to your personal nirvana download log in the JSON format.
            Its structure must be `dict[str, list[str]]` (a mapping from a workflow id
            to already downloaded instance ids for this workflow).
            If you don't have a log, create a JSON file with an empty dictionary: `{}`.
            **Note: the log should be stored in a VCS.**
        unpack: if False, the downloaded results will remain tar archives.
        interactive: if True, you will be asked for a confirmation before the download.
    """
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    instances = _fetch_new_completed_instances(
        workflow_id, instance_ids, log_path, code_block_pattern='python_3_deep_learning'
    )
    print()
    if not instances:
        print('Nothing to download.')
        return
    n_instances = len(instances)
    print(f'{n_instances} instance{"s" if n_instances > 1 else ""} can be downloaded:')
    print()
    for instance in instances:
        print(instance['_fullname'])

    if interactive:
        print('\nDownload? [y/N] ', end='')
        if input().strip() != 'y':
            return

    for i, instance in enumerate(instances):
        _, _, free_bytes = shutil.disk_usage('/')
        free_gb = free_bytes / 2**30
        if free_gb < _MIN_REQUIRED_DISK_SPACE_GB:
            print(
                '\nNot enough disk space to continue the download.'
                f' Required: {_MIN_REQUIRED_DISK_SPACE_GB}GB.'
                f' Available: {int(free_gb)}GB.'
            )
            return

        print(f'\n[{i + 1}/{n_instances}] Downloading {instance["_fullname"]}')
        _download_python_3_deep_learning_data_output(
            workflow_id,
            instance,
            Path.cwd() if output is None else output,
            log_path,
            unpack=unpack,
        )

    print('\nDone!')


if __name__ == '__main__':
    lib.util.init(torch_=False)
    app = cyclopts.App(
        config=[
            cyclopts.config.Toml(
                dev.lib.env.get_user_settings_path(),
                ['nirvana', 'download'],
                must_exist=True,
                use_commands_as_keys=False,
            ),
            # Take workflow_id from [nirvana.run]
            cyclopts.config.Toml(
                dev.lib.env.get_user_settings_path(),
                ['nirvana', 'run'],
                must_exist=True,
                use_commands_as_keys=False,
                allow_unknown=True,
            ),
        ]
    )
    app.default(download)
    app()
