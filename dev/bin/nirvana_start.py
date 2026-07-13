import datetime
import json
import os
import shutil
import sys
import time
from pathlib import Path

import nirvana_dl
import numpy as np
import torch
from loguru import logger

import lib.env
import lib.experiment
import lib.util


def _printerr(*args, **kwargs) -> None:
    print(*args, **kwargs, file=sys.stderr)


def _benchmark_numpy_cpu(d: int) -> None:
    a = np.random.randn(d, d)
    a = a @ a.T


def _benchmark_torch_cpu(d: int) -> None:
    b = torch.randn(d, d)
    b = b @ b.T


def _benchmark_torch_cuda(d: int) -> None:
    c = torch.randn(d, d, device='cuda')
    c = c @ c.T
    torch.cuda.synchronize()


def benchmark(
    *, n_iterations: int, n_warmup_iterations: int, gpu_multiplier: int, d: int
) -> None:
    lib.util.print_sep()
    print(f'{lib.util.get_function_full_name(benchmark)} | {datetime.datetime.now()}')
    lib.util.print_sep()

    print(f'{torch.cuda.is_available()=}')
    print(f'{os.environ.get("CUDA_VISIBLE_DEVICES")=}')

    summary = []
    for name, fn, n, nw in [
        ('numpy.cpu', _benchmark_numpy_cpu, n_iterations, n_warmup_iterations),
        ('torch.cpu', _benchmark_torch_cpu, n_iterations, n_warmup_iterations),
        *(
            [
                (
                    'torch.cuda',
                    _benchmark_torch_cuda,
                    gpu_multiplier * n_iterations,
                    gpu_multiplier * n_warmup_iterations,
                )
            ]
            if torch.cuda.is_available()
            else []
        ),
    ]:
        for i in range(nw):
            fn(d)

        _printerr()
        elapsed = 0.0
        measurements = {'time': [], 'throughput': []}
        for i in range(n):
            start = time.perf_counter()
            fn(d)
            elapsed += time.perf_counter() - start

            throughput = (i + 1) / elapsed
            measurements['time'].append(elapsed)
            measurements['throughput'].append(throughput)
            _printerr(
                f'[name] {name}'
                f' [i] {i:>2}'
                f' [time] {elapsed:>10.6f}'
                f' [it/s] {throughput:.3f}'
            )
        summary.append(
            {
                'name': name,
                'n': n,
                'd': d,
                'time': measurements['time'][-1],
                'throughput': measurements['throughput'][-1],
            }
        )

    print()
    header = f'{"name":>10}  {"n":>5}  {"d":>5}  {"time":>6}  {"throughput":>10}'
    print(header)
    print(f'{"-" * 10}  {"-" * 5}  {"-" * 5}  {"-" * 6}  {"-" * 10}')
    for row in summary:
        print(
            f'{row["name"]:<10}'
            f'  {row["n"]:>5}'
            f'  {row["d"]:>5}'
            f'  {row["time"]:>6.2f}'
            f'  {row["throughput"]:>10.3f}'
        )


def unpack_snapshot() -> None:
    print()
    lib.util.print_sep()
    print(
        f'{lib.util.get_function_full_name(unpack_snapshot)}'
        f' | {datetime.datetime.now()}'
    )
    lib.util.print_sep()

    # >>> If there is no snapshot, exit.
    snapshot_dir = lib.env.get_snapshot_dir()
    assert snapshot_dir is not None
    print(f'The expected snapshot path: "{snapshot_dir}"')
    if snapshot_dir.exists():
        print('The snapshot is found')
    else:
        print('The snapshot does not exist')
        return

    # >>> Print all files in the snapshot.
    _printerr(
        'The snapshot contains the following files (relative to the project directory):'
    )
    for directory, _, filenames in snapshot_dir.walk():
        for filename in filenames:
            _printerr(directory.joinpath(filename).relative_to(snapshot_dir))
            del filename
        del directory, filenames
    _printerr()

    # >>> If there are no exp/ directories in the snapshot, exit.
    exp_snapshots = [
        snapshot_dir / 'exp',
        snapshot_dir / 'dev' / 'exp',
        *filter(Path.is_dir, snapshot_dir.glob('dev/users/*/exp')),
    ]
    if not any(x.exists() for x in exp_snapshots):
        print('Nothing to restore from the snapshot.')
        return

    # >>> Restore the `Python 3 Deep Learning`-specific JSON output.
    json_output = snapshot_dir / 'json_output.json'
    if json_output.exists():
        shutil.copyfile(json_output, nirvana_dl.json_output_file())

    # >>> Restore completed and unfinished experiments.
    for exp_snapshot in exp_snapshots:
        for directory, _, _ in exp_snapshot.walk():
            # NOTE[exp]
            if lib.experiment.is_experiment(directory) and (
                lib.experiment.is_done(directory)
                or lib.experiment._is_running(directory)
            ):
                exp = lib.env.get_project_dir() / directory.relative_to(snapshot_dir)
                config_path = lib.experiment.get_config_path(exp)
                snapshot_config_path = lib.experiment.get_config_path(directory)
                if (
                    config_path.exists()
                    and snapshot_config_path.exists()
                    and (config_path.read_text() != snapshot_config_path.read_text())
                ):
                    logger.warning(
                        f'Skipping {directory} because of incompatible configs'
                    )
                    continue

                exp.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(directory, exp, dirs_exist_ok=True)  # NOTE[exp]
                if lib.experiment.is_done(exp):
                    lib.experiment.add_to_tmp_output(exp)
                print(f'Restored {exp}')
            del directory
        print()

    snapshot_dir.joinpath('SNAPSHOT_UNPACKED').touch()


def fill_json_output() -> None:
    json_output_path = Path(nirvana_dl.json_output_file())
    json_output = (
        json.loads(json_output_path.read_text()) if json_output_path.exists() else {}
    )
    json_output['branch'] = lib.util.git_get_current_branch()
    json_output_path.write_text(json.dumps(json_output))


def main() -> None:
    benchmark(n_iterations=100, n_warmup_iterations=10, gpu_multiplier=10, d=2048)
    unpack_snapshot()
    fill_json_output()


if __name__ == '__main__':
    assert not lib.env.is_local()
    lib.util.init()
    main()
