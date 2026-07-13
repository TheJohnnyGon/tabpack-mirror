# YR Template

YR Template (or simply Template) is a template repository for research projects.
It works as follows:

- You start a new project by cloning Template and reinitializing it as a new Git repository
  (this process is automated).
- After that, your project is considered to be fully independent from Template. In particular, this
  implies full freedom to change the code as needed and full responsibility over the code.
- Although you never merge commits from your project to Template,
  **you can merge new commits from Template to your project**.

> [!NOTE]
> This document is fully applicable both to Template and to any repository created from it.
> The only exception is the [Contributing](#contributing-to-template) section, which is specific to
> Template.

# Prerequisites

- Linux or macOS
- Text editor supporting remote work
  (VSCode with the ["Remote Development"](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack)
  extension pack is recommended). If you choose VSCode, it must be *no newer* than v1.98.2.
- git
- uv ([install](https://github.com/astral-sh/uv?tab=readme-ov-file#installation) the latest version
  or update your existing uv installation by running `uv self update`)

# Creating a new repository

▪️ Connect to any Terra-like machine:

```
ssh terra.vla.yp-c.yandex.net
```

▪️ Create a new repository from Template:

```
git clone https://github.com/Yura52/yr-template-2025
uv run yr-template-2025/dev/misc/init_repository.py <REPOSITORY NAME>
```

▪️ Your repository is ready. The commit hash of Template used to initialize your repository is
stored at `dev/misc/.yr-template` in your repository.

# Setting up an existing repository

To set up an existing repository created from Template, complete all subsections in the given order.
Note that this applies equally to both cases:
- Setting up a fresh repository just created from Template in the previous section.
- Setting up a clone of an active repository created from Template earlier (e.g. you are setting up
  a new clone of your repository on a different machine, or you are joining someone else's
  repository, etc.).

> [!IMPORTANT]
> For the rest of this document, all commands must be run from the root of the repository:
> 
> ```
> cd <REPOSITORY DIRECTORY>
> ```

## Code and software

To install a Python virtual environment, configure some of the tools (e.g. git) and create files and
directories needed for day-to-day work, simply run:

```
uv run dev/misc/setup_repository.py
```

## Remote repository

If this is a fresh repository just created from Template, create an empty **PRIVATE** repository
on GitHub and connect your local repository to the remote one:

```
git remote add origin https://github.com/<account>/<repository name>.git
git push --set-upstream origin main
```

## Data

If the project is located on Terra:

```
ln -s /mnt/tabular/data/common/v<version> data
```

where `<version>` is the latest version available. On other machines, pull the data from Terra:

```
scp -r terra.vla.yp-c.yandex.net:/mnt/tabular/data/common/v<latest> data
```

## Check

At this point, the following commands should work:

> [!TIP]
> Always use `uv run` instead of `python`, and never explicitly activate the virtual environment
> of this repository.

```
export CUDA_VISIBLE_DEVICES="0"
uv run bin/demo.py exp/examples/demo --force
```

This example demonstrates the main pattern of this repository:

- `exp/examples/demo` is the experiment directory, or simply "experiment".
- `exp/examples/demo/config.toml` is the config file of the experiment. Think command line
  arguments, but in the form of a text file in the [TOML](https://toml.io/en) format.
- `exp/examples/demo/report.json` is the main output of the experiment.
- `exp/examples/demo/summary.txt` is a human-friendly summary of the report.
- `config.toml` and `report.json` are tracked by VCS (except for this test example, where
  `report.json` is *not* tracked), and everything else is ignored.

## Nirvana

[Nirvana](https://nirvana.yandex-team.ru)
is a service for running computational graphs on powerful remote clusters.

▪️ Get Nirvana token as mentioned [here](https://docs.yandex-team.ru/nirvana/api#requests).
Save it to `$HOME/.nirvana/token`. Ensure that there is no new line at the end of the file by
running the following:

```
uv run python -c """
from pathlib import Path
path = Path.home() / '.nirvana' / 'token'
path.write_text(path.read_text().rstrip())
print('\nNirvana token is ready')
"""
```

▪️ Fill `dev/users/$USER/settings.toml` following the instructions below.

> [!IMPORTANT]
> Tokens and keys must **NOT** be stored explicitly in `settings.toml`.
> Instead, the names of their *Nirvana secrets* are stored.
> If, after completing the following steps, you are unsure what to put in `settings.toml`,
> ask your colleagues before committing anything.

- Open the [Nirvana secrets](https://nirvana.yandex-team.ru/secrets) page for completing the next
  steps.
- `pulsar_token_secret`
    - Get Pulsar token as explained
      [here](https://docs.yandex-team.ru/pulsar/quickstart#authorization).
    - A secret for this token should be created automatically. Find its name on the Secrets page
      and add it to your `settings.toml`.
- `yt_token_secret`
    - Get YT token [here](https://oauth.yt.yandex.net/).
    - Store it in `~/.yt/token`. Ensure that there there is no new line in the end.
    - Create a Nirvana secret for this token and add its secret name to your `settings.toml`.
- `ssh_key_secret`
    - Get SSH key for your GitHub account [here](https://github.com/settings/keys).
    - Create a Nirvana secret for this token and add its secret name to your `settings.toml`.
- Close the Secrets page.
- `workflow_id`
    - On [Nirvana](https://nirvana.yandex-team.ru) website:
      `Create New` -> `Workflow` -> `Save` -> Enter a workflow name.
    - The part of the URL after `/flow/` is the workspace id:
      `https://nirvana.yandex-team.ru/flow/<WORKFLOW ID>/...`.

▪️ Check that Nirvana-related settings are configured correctly by running the following command:

```
uv run --isolated --no-dev --group yandex -m dev.lib.nirvana.run "uv run bin/demo.py exp/examples/demo --force" --n-gpus 1 --priority high --comment "CLI example"
```

Running multiple graphs and downloading results is described later in this document.

## Tools

### git

At this point, git has already been configured by the `dev/misc/setup_repository.py` script.

### ruff

[Configure `ruff`](https://docs.astral.sh/ruff/editors/setup)
to keep the code compatible between collaborators:

- `ruff` from the repository's environment must be used.
- `ruff` must use `pyproject.toml` of this project as a config.
- `ruff` must automatically format your code and sort imports on save.
- `ruff` must highlight various code-related issues.

An example of Ruff-related settings in VSCode as of March 2025:

```
{
    "[python]": {
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": "explicit"
        },
        "editor.defaultFormatter": "charliermarsh.ruff"
    },
    "ruff.fixAll": false,
    "ruff.importStrategy": "fromEnvironment",
}
```

### Language server

Use Pylance in VSCode (this is true by default, no need to configure anything) and
[Pyright](https://microsoft.github.io/pyright) otherwise. In both cases, `pyproject.toml` already
contains the necessary settings in the `[tool.pyright]` table.

### (Optional) Aliases

In your shell config, configure aliases to simplify running typical commands. Examples:

```
alias u="uv run"
alias nirvana-run="uv run --isolated --no-dev --group yandex -m dev.lib.nirvana.run"
alias nirvana-download="uv run --isolated --no-dev --group yandex -m dev.lib.nirvana.download"
```

In Fish, you can use abbreviations instead of aliases. For example:

```
abbr --add u "uv run"
```

### (Optional) VSCode tasks

Optionally, configure [tasks](https://code.visualstudio.com/docs/editor/tasks)
in `.vscode/tasks.json` to simplify running typical commands. Example:

```
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Nirvana run TOML",
            "type": "shell",
            "command": "uv run --isolated --no-dev --group yandex -m dev.lib.nirvana.run from-toml ${file}",
            "presentation": {
                "reveal": "always"
            },
        },
    ]
}
```

Usage example: `Ctrl+Shift+P (macOS: Cmd+Shift+P)` -> `Tasks: Run Task` -> `Nirvana run TOML`.
To skip the dialog about output scanning, add the following to your `settings.json`:

```
{
    "task.problemMatchers.neverPrompt": true
}
```

### (Optional) Jupyter

TODO(port forwarding)

To run Jupyter in the background, create a tmux session and run Jupyter there:

```
tmux new -s <PROJECT NAME>-jupyter
uv run jupyter lab --port <PORT>
<detach from the tmux session: Ctrl+B D>
```

## (Optional) Local development

> [!TIP]
> During your first walkthrough, skip this section.

In fact, all the above steps can be completed locally, with only one difference. Namely, when you
attempt to run Nirvana-related commands, you are likely to get an SSL-related error. To fix it, do
the following:

1. Download the Yandex SSL certificate: https://crls.yandex.net/YandexInternalRootCA.crt
2. Put it to the right place (details below).
3. Make your software aware of the new certificate (details below).

The points 2 and 3 above depend on OS (Linux vs macOS) and Python distribution
(uv vs conda vs pyenv vs python.org vs ...).
This project uses exclusively the uv distribution of Python (due to the `--managed-python` flag),
so the solutions below will differ only by OS.

### macOS + Apple Silicon

This solution works on macOS 14.5 with Apple Silicon, and hopefully on other macOS versions as well.
Start by running these two commands:

```
uv sync --no-dev --group yandex
uv sync
```

The first of the above commands ensures that all Yandex-related packages are cached by uv.
Now, put the downloaded certificate to any location, e.g. `$HOME/.yandex/YandexInternalRootCA.crt`.
Finally, set `SSL_CERT_FILE=<path to the certificate>` when running Nirvana-related commands,
and they will work. Example:

```
SSL_CERT_FILE=$HOME/.yandex/YandexInternalRootCA.crt uv run dev/bin/nirvana_run.py "uv run bin/demo.py exp/examples/demo --force"
```

### Other cases

TODO (please, share your solution if you find it). Potentially helpful links:
- [For Linux users](https://wiki.yandex-team.ru/security/ssl/sslclientfix/#vpython)
- [For macOS users](https://wiki.yandex-team.ru/security/ssl/sslclientfix/#vpythonnamacosx)
- [For conda users](https://wiki.yandex-team.ru/security/ssl/sslclientfix/#vanacondapython)

## Next steps

At this point, your environment is ready. The suggested next steps are as follows:

- Read the [Conventions](#conventions) and [Features](#features) sections.
- Skim through the rest of the document to get an idea of what it offers.
- If you are not familiar with `uv`, read the following pages from the uv documentation:
  - [Working on projects](https://docs.astral.sh/uv/guides/projects/)
  - [Running commands in projects](https://docs.astral.sh/uv/concepts/projects/run/)
  - [Running standalone Python scripts](https://docs.astral.sh/uv/guides/scripts/)

# Usage

## Conventions

This section describes conventions suggested for projects created from Template.

**Layout**

- The `lib/` directory contains common utilities shared across experiments.
- The `bin/` directory contains the main high-level scripts. Also, `bin/` is where research
  usually happens. That is, new experimental methods and architectures are more likely to be
  implemented in scripts in `bin` than in `lib`.
- The `exp/` directory is where experiments are stored.
  The layout of `exp/` itself can be arbitrary.
- The `vendor/` directory contains *unmodified* code from other projects. The only allowed
  (and actually recommended) modification is adding the following top-level comments:
  ```
  # ======================================================================================
  # Source: <parmalink to the file or codebase>
  # fmt: off
  # isort: off
  # ======================================================================================
  ```
- The `local/` directory is ignored by VCS and can be used for any purposes.
- The `dev/` directory is private to Yandex employees, i.e. it must not be released to public. The
  public part of the repository (i.e. `bin/`, `lib/`, etc.) must not depend on `dev/` in any way.
- The `dev/users` directory contains user directories with arbitrary personal stuff (scripts,
  notebooks, etc.). Users must use their actual usernames as the directory names. The main code and
  experiments must not depend on `dev/users` in any way.
- Browse the repository to get and idea of the remaining parts.

**Branches**

> [!NOTE]
> What is a good branching strategy in this repository is an open question.
> Treat the below list as suggestions, and share your ideas.

- Treat the main branch of your repository as a "branch template" to enable quick experiments on
  arbitrary ideas simply by creating new branches from the main branch. In particular, avoid storing
  experiments in the main branch.
- Don't store too many experiments in a single branch to avoid slowing down git. What is "too many"
  will be defined as you go.
- To enable merging experiments from different branches, avoid overlapping experiments in branches.
  The simplest way to achieve that is to store experiments in a given branch in `exp/<branch name>`.
- If you are working on something that can be useful to others (say, some improvements to
  the data loading code, or the Nirvana-related code, etc.), consider
  [contributing](#contributing-to-template) to the Template repository and then
  [merging the changes](#merging-updates-from-template-to-your-project) from Template to your
  repository.

How to distribute Python files and experiments across branches depending on the activity (quick
experiments, long-term projects, side quests, etc.) and whether to use branches for anything beyond
storing experiment files is an open question.

**Technical**

- `CUDA_VISIBLE_DEVICES` must always be set explicitly. It is handy to set it in your shell config.
- Respect the warnings raised by tools like ruff, Pylance, etc. Either fix the code or use
  appropriate comments to suppress warnings if they are irrelevant.

**Dependencies**

*TL;DR: avoid modifying existing dependencies.*

- To keep results reproducible, don't modify *existing* dependencies in `dependencies` and
  `project.optional-dependencies` in `pyproject.toml`, unless truly necessary. Adding *new*
  dependencies is fine (again, unless this affects existing dependencies).
- Modifying `dependency-groups`, and in particular adding new dependency groups, should be fine.
  Pay close attention to changes triggered by your modifications, and avoid modifications that
  affect existing dependencies.
- Updating dependencies requires discussion with your teammates.

**Template**

- Be subscribed to releases of Template.
  On the GitHub page of Template: Watch -> Custom -> Releases.
- If you make a change that must be adopted by other projects (e.g. changes to metric computation),
  contribute it to Template.
- Optionally, contribute universally useful improvements to Template
  (e.g. improvements to Nirvana scripts, the experiment system, etc.).

## Working with Yandex-specific code

There are two ways of running scripts relying on internal Yandex packages.

(1) Running in temporary isolated environments (this saves disk space, but scripts start slower):

```
uv run --isolated --no-dev --group yandex <YOUR COMMAND>
```

(2) Running in a permanent second virtual environment (this consumes disk space, but scripts start
faster):

```
UV_PROJECT_ENVIRONMENT=.venv-yandex uv run --no-dev --group yandex <YOUR COMMAND>
```

Note that if you need autocompletion in editors when working on Yandex-related code,
you will need the second environment anyway.

## Features

This section briefly describes some useful features of this repository not mentioned above.

> [!TIP]
> Some scripts support the `--help` flag.

▪️ Downloading results of finished Nirvana graphs
(the command is safe to run if there are no new finished graphs):

```
uv run --isolated --no-dev --group yandex -m dev.lib.nirvana.download
```

▪️ Running Nirvana graphs from a custom script:

```
uv run --isolated --no-dev --group yandex dev/examples/nirvana_run.py
```

▪️ Running Nirvana graphs from a TOML file:

```
uv run --isolated --no-dev --group yandex -m dev.lib.nirvana.run from-toml dev/examples/nirvana_run.toml
```

▪️ Running one Nirvana graph with multiple commands from a terminal:

```
uv run --isolated --no-dev --group yandex -m dev.lib.nirvana.run --n-gpus 1 --comment "CLI example (multiple commands)" "
echo Hello, World!
uv run bin/demo.py exp/examples/demo --force
"
```

▪️ Use `bin/mlp.py` as an example of training a tabular DL model.

▪️ Use `dev/examples/results.ipynb` as an example of summarizing results of experiments.

▪️ Use `dev/examples/configs.py` as an example of creating experiment configs for many datasets.

▪️ Use `dev/bin/summarize.py` to (re)compute `summary.txt` for all experiments in a given directory,
recursively.

▪️ Use `lib.experiment.{run,run_cli}` to run `main` experiment functions. See how it is done in
`bin/tune.py` and `bin/evaluate.py`.

▪️ Use `lib.util.init` before running the `main` function in scripts. See how it is done in scripts
in `bin/`.

▪️ Use `lib.util.init_notebook` after all imports in Jupyter notebooks. See how it is done in
`dev/examples/results.ipynb`.

▪️ Use `lib.experiment.create` to create an experiment.

▪️ Use `lib.experiment.is_done` to check if an experiment is finished.

▪️ Use `dev.lib.datasets` to simplify dataset management in personal scripts and notebooks.
In particular, use `dev.lib.datasets.load_extended_info` to look up detailed dataset properties.

▪️ For everything else, read the source code.

## Pulling updates from Template to your project

From time to time, Template receives useful updates with new features and bug fixes. This section
describes how to pull updates from Template to a repository created from Template.

### Automatic updates (recommended)

Automatic updates work only if you don't modify files already presented in Template, with the
only exception being project-specific files, such as `pyproject.toml`, `uv.lock`, `pixi.lock`,
`.gitignore`, `dev/docker/Dockerfile`. In other words, you should treat Template as a third-party
"package" to keep your project compatible with future Template updates. Note that there are no
restrictions on creating and editing new files.

When updating your repository for the first time, add Template as a remote repository to your
project:

```
git remote add yr-template https://github.com/Yura52/yr-template-2025
```

Then, if you follow the above conventions, the following command will pull updates from Template to
your repository:

> [!TIP]
> The command updates files (as if you did that manually) and stages the changes, but does
> _not_ commit them. Thus, you will have a chance to review the changes and revert them if needed.

```
uv run dev/misc/update.py
```

### Manual updates

Assume that Template receives new commits that you want to merge to your repository.
Also, you may be interested only in some specific commits, not in all updates.
The solution is the `git cherry-pick` command that allows merging commits from other branches
and repositories to the current branch.

▪️ Add Template as an additional remote repository:

```
git remote add yr-template https://github.com/Yura52/yr-template-2025
```

▪️ Check the result:

```
git remote -v
```

▪️ Fetch all branches of Template:

```
git fetch yr-template
```

▪️ Take a look at the commits in Template:

```
git log --oneline yr-template/main
```

Now, everything is ready for cherry-picking. Choose either (A) or (B).

▪️ (A) To cherry-pick *one* specific commit with the hash `aaaaaaa`:

```
git cherry-pick aaaaaaa
```

▪️ (B) To cherry-pick a *range* of commits from `bbbbbbb` to `ccccccc`, inclusive
(note the `^` in the command):

```
git cherry-pick bbbbbbb^..ccccccc
```

▪️ Resolve conflicts. Git UI in VSCode can be helpful here.

▪️ Tell Git that you are ready to continue:

```
git cherry-pick --continue
```

▪️ Verify the staged changes by running `git status` and make a commit. Done!

## Updating the Docker image

> [!NOTE]
> Below, `<PROJECT NAME>` is the name of the project created from Template.
> In the case of Template itself, `<PROJECT NAME>` is `yr-template`.

▪️ If doing this for the first time, configure Docker and authorize as explained
[here](https://wiki.yandex-team.ru/docker-registry/?revision=188178527#authorization).

▪️ Edit `dev/docker/Dockerfile` as needed and bump `LABEL version`.

> [!IMPORTANT]
> 
> If this is your first update of the Docker image for your project created from Template, then:
> - Change `LABEL description`.
> - Set `LABEL version` to a reasonable starting value, e.g. `0.0.1`.

▪️ Build an image:

```
uv lock
ln -f uv.lock dev/docker/uv.lock
ln -f pyproject.toml dev/docker/pyproject.toml
docker build -t registry.yandex.net/yr/<PROJECT NAME>:<VERSION> --network host dev/docker
```

▪️ (Optionally) To test the container locally and simulate an actual Nirvana run:

```
docker run --name <NAME> --gpus all -i -t --rm --mount src="$(pwd)",target=/nirvana-project-dir,type=bind registry.yandex.net/yr/<PROJECT NAME>:<VERSION> /bin/bash
```

Comments:

- **Important:** `type=bind` means passing the original host directory to the container,
  i.e. changing files in the container will change the original files on your disk and vice versa.
- `--network host` is *not* provided to keep the internet off (since it is also off on Nirvana).
- `<NAME>` can be anything.
- `--rm` deletes the container on exit.

Also, in the running container, you may want to run commands on behalf of a non-root user to closer
mimic what happens on Nirvana:

```
useradd testuser
su -c "whoami" testuser
```

▪️ Upload the image.

```
docker push registry.yandex.net/yr/<PROJECT NAME>:<VERSION>
```

> [!NOTE]
> Using the Docker terminology, `registry.yandex.net/yr/<PROJECT NAME>` is a "Docker repository",
> and `<VERSION>` is a "tag". However, when filling some forms mentioned below, you should enter
> simply `yr/<PROJECT NAME>` instead of the full repository name.

If creating a new layer, import your Docker repository in the Layers service:
1. Open https://layers.yandex-team.ru/layers
2. Create layer -> From Docker repository -> Fill the form.
3. Go to your layer page -> Button on top with the Docker logo -> Import -> Fill the form.
   All future images uploaded with the `docker push` command will be synced
   with the created Nirvana layer automatically.

▪️ Find your Nirvana layer on https://layers.yandex-team.ru/layers and update `layer_id` in
`dev/settings.toml`.

## Tips

▪️ Copying a file or directory from another branch to the current branch:

```
git restore --source branch-name --worktree path/to/file/or/directory
```

▪️ Working on multiple branches simultaneously:

```
git worktree add ../my-repo-branch-name branch-name
```

Removing the worktree when it is not needed anymore:

```
git worktree remove ../my-repo-branch-name
```

See [this page](https://git-scm.com/docs/git-worktree) for details.

▪️ To work in *one* VSCode workspace on the original Template repository, the main branch of your
repository, and non-main branch(es) of your repository, create worktrees for the non-main branches
as shown above, and bring the necessary directories to one workspace using the
`Add Folder To Workspace` command in VSCode.

▪️ Downloading the latest commit of a given branch with minimal history (much faster than naive
`git clone`):

```
git clone --branch some-branch --depth 1 <repository url>
```

▪️ Other useful commands for working with branches:
- `git rebase`
- `git merge` (in particular `git merge --squash`)
- `git cherry-pick`
- `git stash`

# Publishing the repository

To make your repository public or submit it to a conference as a supplementary material,
copy it and make the following changes to the copy.

**Required:**

- Remove the `dev/` directory.
- In `.gitignore`, remove the rules related to `dev/`.
- In `pyproject.toml`:
  - Change the project name.
  - Remove *all* Yandex-related things from:
    - `[dependency groups]`
    - `[tool.uv]`
    - `[tool.uv.index]`
    - `[tool.uv.sources]`
    - Maybe something else? Carefully review the rest of the file.
  - Remove `dev` from `[tool.setuptools]`.
  - Check the project name and description.
- Recompute `uv.lock` by running `uv lock`.
- Add the `LICENSE` file with a license of the project.
- Add `README.md`.
- For anonymous submissions, ensure anonymity of the repository.

**Optional:**

- Change the remaining code as needed. This part is highly project-specific
  and can take an arbitrary amount of time.

# Contributing to Template

Read this section if you are going to contribute changes to Template.

## Conventions

- Template is used only as a starting point for projects and must not contain any
  project-specific or user-specific code.
- Changes to Template are done through pull requests with at least one of
  `{@ygorishniy, @TODO, @TODO}` as a reviewer.
  For now, the only exception is minor changes made by `@ygorishniy`.
- Maintain a clean commit history with informative messages.
  Intuitively, the output of `git log` should look like a changelog.
  In particular, the first line of a commit message should result in a meaningful output of
  `git log --oneline`.
- Inform your teammates about your commits to Template that you find important.
- Be familiar with the [Versioning policy](#versioning). 

## Tests

```
.git/hooks/pre-commit
uv run pytest dev
```

## Versioning

> [!NOTE]
> The below policy is active since `v0.4.0`.

*TL;DR: for now, the project version only reflects changes to the environment.*

To ensure that the repository is always in a correct state, if *any* of the following happens,
then *all* of the following must happen **within the same commit**:
- Updating `project.version` in `pyproject.toml`.
- Updating `layer_id` in `dev/settings.toml`.
- Updating the Docker image version to the new value of `project.version`.
- (If applicable) Updating dependencies in `pyproject.toml`.

Commits where `project.version` is updated to `X.Y.Z` must be tagged:

```
<do the commit where project.version is updated>
git tag -a vX.Y.Z -m "vX.Y.Z"
git push
git push origin vX.Y.Z
```

**The project version**

> [!NOTE]
> In `X.Y.Z`, `X` is the *major* version, `Y` is the *minor* version,
> and `Z` is the *patch* version.

- The *major* version always remains zero.
- The *minor* version must be updated if and only if any of the following happens:
  - Changes to the `project.dependencies` list in `pyproject.toml`.
  - Breaking changes to `dev/docker/Dockerfile`.
- The *patch* version must be updated if and only if any of the following happens:
  - Changes to `dependency-groups` in `pyproject.toml`.
  - Non-breaking changes to `dev/docker/Dockerfile`.
- Other types of changes must not result in version changes.

**The Docker image version**

`project.version` in `pyproject.toml` must always be equal to
`LABEL version` in `dev/docker/Dockerfile`.

## Tips

**Recomputing the lockfile from scratch**

- Remove the lockfile.
- In `pyproject.toml`, comment Yandex-related dependency groups
  and the `default-groups` option in `tool.uv`.
- Run `uv sync`.
- In `pyproject.toml`, uncomment what you previously commented.
- Run `uv sync`.
- Done. The above trick ensures that the non-yandex environment receives the latest package
  versions.
