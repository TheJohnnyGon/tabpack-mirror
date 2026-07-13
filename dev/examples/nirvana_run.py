# NOTE: see README for usage instructions.
from dev.lib.env import load_dev_settings, load_user_settings
from dev.lib.nirvana.run import run_pydl_graph

options = {}
options |= load_dev_settings()['nirvana']['run']
options |= load_user_settings()['nirvana']['run']
options |= {
    'n_gpus': 1,
    'pool_tree': [
        # 'gpu_tesla_a100',  # A100 40GB
        'gpu_tesla_a100_80g',  # A100 80GB
        # 'gpu_hainan_80g',  # H100 80GB
        # 'gpu_hutang_141g_cloud',  # H200 141GB
    ],
}
research_gpu = False
if research_gpu:
    options['priority'] = 'high'
    options['job_scheduler_yt_pool'] = 'research_gpu'

run_pydl_graph(
    '\n'.join(
        [
            'echo Hello, World!',
            'uv run bin/demo.py exp/examples/demo --force',
        ]
    ),
    comment='Example',
    **options,
)
