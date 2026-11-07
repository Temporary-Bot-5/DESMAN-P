# Install DESMAN(P)

The installers create a per-user git checkout, install uv and Python 3.12 if needed, install
DESMAN without development extras, create Desktop/menu shortcuts and open `desman gui`.
No administrator shell is needed for DESMAN itself. On Windows the agents do not run code
because there is no sandbox; Linux code execution requires bubblewrap and the person's permission.

## 1. Linux

Install git first if needed (the installer prints the command for apt, dnf, pacman or zypper
and stops). Download the script, then run it from a terminal so you can answer its question:

```sh
curl -fsSL https://raw.githubusercontent.com/Temporary-Bot-5/DESMAN-P/master/install/install.sh -o install.sh
sh install.sh
```

The script explains before downloading and running the official uv installer. uv installs
Python as needed. For unattended installation, explicitly accept with `--yes`:

```sh
curl -fsSL https://raw.githubusercontent.com/Temporary-Bot-5/DESMAN-P/master/install/install.sh | sh -s -- --yes
```

The site's shorter `curl … | sh` command asks questions through `/dev/tty` when stdin is
a pipe. Without an accessible terminal, pass `--yes` explicitly or it refuses.
Add `~/.local/bin` to PATH to use `desman` in a terminal. The shortcuts work without this.
The Desktop launcher may need to be marked trusted by your desktop environment.

Flags: `--yes` accepts each confirmation; `--no-start` skips opening the interface;
`--uninstall` removes the application; `--purge` also removes data and config (only with
`--uninstall`). `DESMAN_HOME` overrides `~/.local/share/desman-app`; use an absolute path.

## 2. Windows

Save [install.bat](https://raw.githubusercontent.com/Temporary-Bot-5/DESMAN-P/master/install/install.bat)
(browser: **Save link as**, keep the extension), then double-click it. The batch file prints
what it does, downloads `install.ps1` over HTTPS from raw.githubusercontent.com into `%TEMP%`
and runs it with your arguments. Its top-level `DESMAN_INSTALLER_REF` defaults to `master`;
a wrapper distributed for another ref must set it to that same ref. If `install.ps1` is
already beside the batch file, it runs that copy instead. It pauses on error.
You may also download `install.ps1` separately and run it in Windows PowerShell 5.1 or later:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
```

The installer asks before running the official uv installer. If git is missing, it asks before
installing Git through `winget install --id Git.Git -e --source winget`. Without winget it
asks before fetching the latest official git-for-windows **MinGit 64-bit zip** via the GitHub
API into `%LOCALAPPDATA%\DESMAN\mingit`. It uses that git for installation and adds its cmd directory to the user PATH for later signed
updates. Sign out and back in before using shortcuts for updates (or install Git normally). A winget installation may require
reopening PowerShell. MinGit is for x64 Windows.

The app lives at `%LOCALAPPDATA%\DESMAN\app` (`DESMAN_HOME` overrides it); Desktop and Start
menu shortcuts point to its virtual environment. For terminal commands use
`& "$env:LOCALAPPDATA\DESMAN\app\.venv\Scripts\desman.exe"` or add its Scripts directory to PATH.

Parameters: `-Yes`, `-NoStart`, `-Ref v0.1.0`, `-Uninstall`, `-Purge` (with `-Uninstall`).
`-DryRun` prints the plan without downloading, invoking commands or changing files.

## 3. Setup and releases

Add a free provider's API key in the setup window, then describe your task in the chat.
The installers select the newest `v*` tag by git version sorting. Until a tag exists they print
a note and use `master`. Set `DESMAN_REF=v0.1.0` (either platform) or `-Ref v0.1.0` (Windows)
to select another tag or branch. A full clone is retained on local branch `desman-installed` so
signed updates can fast-forward it. Re-running fetches and checks out the selected ref, reuses
the environment and recreates shortcuts. Tracked edits or a foreign directory are refused.
Reinstalling is an explicit reset of that install branch to the chosen ref; use signed updates
for normal upgrades rather than rerunning an installer against an older tag.

Updates are root-signed releases, not automatic pulls. Opt into desman-global with
`desman global join --registry URL` using the owner's registry address, then run `desman worker`
to receive releases. Inspect and install one explicitly:

```sh
desman global update --show
desman global update --apply NUMBER
# To undo the last installed update:
desman global update --revert
```

Use the installed executable on Windows as described above. Stop other chats/workers before
applying. Restart afterwards. If the command reports changed dependencies, run `uv pip install
--python .venv/bin/python -e .` inside the app checkout (Windows: `.venv\Scripts\python.exe`).
If it reports protected files, inspect them and run `desman protected approve`. Releases must
be descendants of the installed commit. A git tag alone is not a root-signed network release.

## Uninstall and data

Close DESMAN first. Run `sh install.sh --uninstall` or `install.bat -Uninstall` using the saved
installer. Confirmation lists the app and shortcuts. uv, Python, git and portable MinGit stay.
The Linux `~/.local/bin/desman` launcher is removed too.

Data locations on **both platforms** (Windows uses `%USERPROFILE%` for `~`, not LOCALAPPDATA):

| Location | Contents |
|---|---|
| `~/.local/share/desman/` | Database, history, workspaces, node key and local state |
| `APP/config.json` | Configuration and API keys in this editable checkout |
| `~/.config/desman/config.json` | Fallback configuration / preserved uninstall copy |

`DESMAN_DATA_DIR` and `DESMAN_CONFIG_DIR` override data and configuration directories.
Ordinary uninstall moves `APP/config.json` to `~/.config/desman/config.json` before removing
the checkout. If that backup already exists, it stops so you can preserve both yourself.
A later installation adopts this saved config on first use. An external config override is
left alone. Keep the same overrides when uninstalling.

`--uninstall --purge` / `-Uninstall -Purge` asks a second question listing exactly the data
directory, configuration directory, and `APP/config.json` to delete. This destroys keys,
history, workspaces and node identity. With a configuration override, the fallback home config
is left alone; remove it yourself if you intentionally kept an older backup there.
`--yes` / `-Yes` accepts both confirmations. Back up anything you need first.

## For the repository owner

1. Enable **Settings → Pages → GitHub Actions**. `.github/workflows/pages.yml` publishes only
   `site/` when a push to `master` changes `site/**`. Hosting needs this repository setting;
   no site is published by running an installer. After enabling, push a site change.
2. Create and push the first tested release tag, e.g. `git tag v0.1.0` then
   `git push origin v0.1.0`. Until then the installers explicitly fall back to master.
3. Publish root-signed releases through the registry workflow in
   [registry-deploy.md](registry-deploy.md#accepting-a-proposal-and-releasing). Keep tags and
   signed release ancestry compatible. Tags do not replace signature verification.

The static page uses only local images, CSS and JavaScript, with English/Russian selected from
the browser language and an explicit switch. The images are copies of the owner's app assets.
