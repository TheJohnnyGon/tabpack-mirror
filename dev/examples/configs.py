"""Generate configs based on a reference config."""

import random
from pathlib import Path

import dev.lib.config as devconfig
import dev.lib.data as devdata
import lib
import lib.env
import lib.experiment
import lib.util

lib.util.init(torch_=False)

# For the reference dataset, the config is composed manually.
reference_dataset = devdata.CALIFORNIA
template_exp = 'exp/examples/mlp/0/{}/tuning'
reference_exp = template_exp.format(devdata.wrap_dataset_name(reference_dataset))

# Shuffle the Why datasets for (hopefully) more uniform workload distribution
# after splitting the dataset list into groups.
why_datasets = devdata.DATASETS_WHY.copy()
random.shuffle(why_datasets)

commands = []
for dataset in [
    devdata.CHURN,
    devdata.CALIFORNIA,
    # *devdata.DATASETS_DEFAULT,
    # *devdata.DATASETS_TABRED,
    # *why_datasets,
]:
    exp = Path(template_exp.format(devdata.wrap_dataset_name(dataset)))
    evaluation_exp = exp.with_name('evaluation')
    if lib.experiment.is_experiment(evaluation_exp) and lib.experiment.is_done(
        evaluation_exp
    ):
        # Skip fully completed experiments.
        continue

    if (
        # Do not rewrite the reference config.
        dataset != reference_dataset
        and (
            # Do not rewrite configs of in-progress experiments.
            not lib.experiment.is_experiment(exp) or lib.experiment.is_fresh(exp)
        )
    ):
        # Generate a new config based on the reference config.
        config = lib.experiment.load_config(reference_exp)
        subconfig = config['space']
        subconfig['data'] = devconfig.make_data_config(
            dataset, cache=subconfig['data'].get('cache', False)
        )
        subconfig['batch_size'] = devconfig.get_nn_batch_size(dataset)
        lib.experiment.create(exp, config=config, parents=True, force=True)

    # Save the command to run the experiment.
    commands.append(f'uv run dev/bin/go.py {exp} --n-seeds 15 --resume')
commands.append('')

# Write the commands to <local dir>/commands/latest.sh
# Potential use cases (the file names are just suggestions):
# - For local runs, manually distribute the saved commands between
#   <local dir>/commands/{0,..,7}.sh files based on the expected CUDA device indices.
# - For Nirvana runs, manually distribute the saved commands between
#   Nirvana graphs in <local dir>/commands/nirvana.toml
#   and use the `from-toml` command of `dev.lib.nirvana.run`.
commands_dir = lib.env.get_local_dir() / 'commands'
commands_dir.mkdir(parents=True, exist_ok=True)
commands_dir.joinpath('latest.sh').write_text('\n'.join(commands))

# Suggest Git commands for adding the configs to the repository.
exp_prefix = template_exp.split('/{}/', 1)[0]
print(f"""\
To add the configs to Git, run:

git add {exp_prefix}/**/config.toml
git commit -m 'TODO[Add/Update] configs in {exp_prefix}'
""")
