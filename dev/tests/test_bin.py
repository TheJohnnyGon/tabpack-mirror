import subprocess

import lib.env


def test_example():
    project_dir = lib.env.get_project_dir()

    example_script = lib.env.get_project_dir() / 'bin' / 'demo.py'
    assert example_script.exists()

    example_exp = lib.env.get_exp_dir() / 'examples' / 'demo'
    assert example_exp.exists()

    process = subprocess.run(
        [
            'uv',
            'run',
            str(example_script.relative_to(project_dir)),
            str(example_exp.relative_to(project_dir)),
            '--force',
        ],
        capture_output=True,
    )
    assert process.returncode == 0
