# next version

*(The changes are listed in the order of commits: from the most recent to the least recent)*

**Highlights**

- Added `dev/misc/update.py` to automate pulling updates from Template to projects. See `README.md`
  for details.
- Implemented multi-process execution in `bin/evaluate.py` and `bin/tune.py` to accelerate
  experiments. Importantly:
  - For `bin/evaluate.py`, the results obtained in the multi-process and single-process regimes are
    bitwise equivalent.
  - For `bin/tune.py`, this is not the case. See the comments in `bin/tune.py` for details.
- The result summarization was significantly improved. Details:
  - The API of `dev/lib.results.py` was simplified. To learn it, go through its public part and read
    the docstrings.
  - `dev/examples/results.ipynb` is now a more useful example that can be copied and adjusted as
    needed.

**Other changes**

- Added `dev/examples/configs.py` as an example of config generation.
- Added `exp/examples/mlp` with full-fledged experiment configs and results on two datasets.
- In `bin/mlp.py`, moved nested functions from `main` to the global scope.
- Added `bin_policy` to `lib.data.build_dataset` to simplify the bin-to-cat conversion.
- Added `lib.util.get_amp_dtype` and `lib.types.AMPDType` to simplify dtype configuration. See
  `bin/mlp.py` for a usage example.
- Added `lib.util.adjust_gpu_memory_usage` for automatic batch size reduction when it does not fit
  into GPU.
- Now, summaries contain most of the original report content, including all experiment-specific
  fields. Technically, the summarization logic changed from "including specific report fields" to
  "transforming/excluding specific report fields", and the conditioning on report functions is gone.
- Plus, a number of bug fixes.

# 0.5.0

**Highlights**

- Now, configs are located *inside* experiment directories and are always called `config.toml`.
- Now, `report.json` does not store the config inside.
- Now, experiment files are not allowed in the main branch, with the only exceptions being
  `exp/examples/*/config.toml`.
- Renamed the `--continue`/`continue_` flag/argument to `--resume`/`resume`.
- Moved the experiment-related features from `lib.util` to `lib.experiment.py`.
- Now, the main functions must be called with `lib.experiment.run` or `lib.experiment.run_cli`,
  but not directly. E.g. see how it is done in `bin/demo.py` and `bin/evaluate.py`.
- Moved the Nirvala-related code to `dev/lib/nirvana`.
- Now, configs can be both dataclasses and typed dictionaries. Config dataclasses must be defined
  with `frozen=True` and `kw_only=True`, and only `None` is allowed as a default field value.
- Now, `dev/bin/go.py` must always be called with `--resume`.
- Added `bin/demo.py`.
- Added `dev/examples`.
- Added `dev/tests`.
- Added `dev/misc` and move some development-related stuff there.

# 0.4.0

**Highlights**

- YR Template was relaunched.
- Added a script for quickly creating new projects from Template.
- Moved README of Template to the root of the repository.
- Added multiple new sections to README, including those describing conventions, features,
  merging commits from Template to your project, contributing to Template, etc.
- Added the new Docker image `yr-template` instead of the previous `tabular-deep-learning`.
- Updated the dependencies.
- Starting from this release, all changes to Template are made through pull requests
  as described in the new "Contributing to Template" section of README.

**Details**

▪️ Added `dev/template/init_project.py` to automate creation of new projects.

▪️ Improved the layout and content of README.

- Move `dev/README.md` to the root of this repository.
- Make the "Environment" section more lightweight and distribute the extra content among new
  sections.
- Add the "Setting up a project" section.
- Add the "Conventions" section.
- Add the "Features" section.
- Add the "Publishing the project" section.
- Add the "Merging updates from Template to your project" section.
- Add the "Contributing to Template" section.

▪️ Improved `pyproject.toml`.
- Now, the core dependencies contain only the packages required by the "public" code,
  i.e. `bin/` and `lib/`.
- The `dev` dependency group can be viewed a starter pack with plotting and dataframe libraries,
  Jupyter-related things, technical tools like `ruff`, etc. Project owners are free to modify
  the `dev` group (and other `dependency-groups`) as needed.

▪️ Dependencies
- Add the missing `PyYAML` core dependency.
- Update all packages to their latest versions to drop yanked packages.

▪️ Other minor improvements

# v0.1.0

> [!NOTE]
> To set up a new project,
> you need to complete the "Environment" section of `dev/README.md`.

**Highlights**

- The repository is now more Git-friendly. For example, a typical number of git-tracked files
  per a model-dataset pair is reduced from 54 to 6.
- Improved Nirvana-related code to simplify running graphs and downloading results.
- Added functions for computing result statistics
  to ensure that all team members compute the same thing.
- Improved summaries.
- Updating Nirvana layers is now easy thanks to Docker images.

**Details**

▪️ Use `dev/bin/nirvana_run.py` to run PyDL Nirvana graphs from a terminal.
Running *one* graph with multiple commands:

```
uv run dev/bin/nirvana_run.py "cmd1 arg1" "cmd2 arg2 arg3"
```

(Experimental) Running *multiple* graphs from a JSON file storing a list of keyword arguments for
`dev.bin.nirvana_run.run_pydl_graph`:

```
uv run dev/bin/nirvana_run.py from-file path/to/spec.json
```

▪️ Use `dev/bin/nirvana_run_advanced_example.py` as an example of using `dev/bin/nirvana_run.py`
as a library.

▪️ Use `dev/bin/nirvana_download.py` to download outputs of PyDL Nirvana graphs from a terminal:

```
uv run dev/bin/nirvana_download.py
```

▪️ Use `dev/notebooks/results_example.ipynb` as an example of summarizing results.
<br>**Important:** use the computational logic from `dev.lib.results` to ensure that all team
members compute the same thing.

▪️ When possible, use `dev.lib.config.make_nn_config` to correctly generate configs for DL models.

▪️ `bin/evaluate.py` and `bin/ensemble.py` are now standard experiment scripts taking one config as
input and producing one report as output.

▪️ Use `lib.init` in scripts (and `dev.lib.init_notebook` in notebooks)
instead of `lib.configure_libraries`. In particular, 
this ensures that you don't accidentally run in a different project environment.

▪️ `DONE` is gone. Use `lib.is_done` to check if an experiment is finished.
<br>**Important:** do not reimplement `lib.is_done` in your code.

▪️ Summary is a now a first-class citizen.
Any experiment producing a report also produces a summary.
<br>**Important:** contrary to reports, summaries are optimized for humans,
are *not* stored in VCS and must *not* be used as input to any scripts.

▪️ Use `dev/bin/summarize.py` to (re)compute summaries for all reports in a given directory,
recursively.

▪️ No more patching of `sys.path` thanks to editable installs of the project
(achieved by `tool.uv.package = true` and `tool.setuptools.packages = [...]` in `pyproject.toml`).

▪️ Nirvana layers are now based on Docker images, which allows easily creating new layers.
`dev/README.md` provides an instruction for building new layers.

▪️ `vendor/` contains *unmodified* files from other projects
  (modulo top-level comments with the source information and things like `# fmt: off`).

▪️ (Experimental) `dev/ygorishniy/bin/configs_commands.py` is an experimental attempt to jointly
  generate configs and commands.

> [!IMPORTANT]
> Please, share feedback and propose ideas for improvements!
> This is especially important for `dev/{bin,lib,notebooks}`, since this code is expected to be
> heavily reused by the team.