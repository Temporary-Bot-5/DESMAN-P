#!/bin/sh
# DESMAN per-user installation. No sudo and no system Python required.
set -eu
DESMAN_REF=${DESMAN_REF:-}
REPOSITORY=https://github.com/Temporary-Bot-5/DESMAN-P.git
APP=${DESMAN_HOME:-$HOME/.local/share/desman-app}
DESMAN_TTY=${DESMAN_TTY:-/dev/tty}
YES=0
START=1
UNINSTALL=0
PURGE=0
for arg in "$@"; do
    case "$arg" in
        --yes) YES=1 ;;
        --no-start) START=0 ;;
        --uninstall) UNINSTALL=1 ;;
        --purge) PURGE=1 ;;
        *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done
confirm() {
    echo "$1"
    [ "$YES" = 1 ] && return 0
    if [ -t 0 ]; then
        printf 'Continue? (y/N) '
        read -r answer
    elif (exec 3<>"$DESMAN_TTY") 2>/dev/null; then
        # A piped installer still asks the person through the controlling terminal.
        exec 3<>"$DESMAN_TTY"
        printf '%s\nContinue? (y/N) ' "$1" >&3
        read -r answer <&3
        exec 3>&-
    else
        echo 'Non-interactive input: use --yes to confirm.' >&2
        exit 1
    fi
    case "$answer" in y|Y) ;; *) exit 1 ;; esac
}
fail() { echo "$1" >&2; exit 1; }
# A marker and repository identity are both required before touching an existing path.
case "$APP" in /*) ;; *) fail 'DESMAN_HOME must be an absolute path.' ;; esac
case "$APP" in ''|/|"$HOME"|"$HOME/"|*/../*|*/..|*/./*) fail 'Unsafe install directory.' ;; esac
[ ! -L "$APP" ] || fail 'Install directory must not be a symlink.'
if [ -e "$APP" ]; then
    [ -f "$APP/.desman-install" ] && [ -d "$APP/.git" ] &&
        [ -f "$APP/src/desman/cli.py" ] || fail "Refusing non-DESMAN directory: $APP"
fi
DATA=${DESMAN_DATA_DIR:-$HOME/.local/share/desman}
CONFIG=${DESMAN_CONFIG_DIR:-$HOME/.config/desman}
# An app containing data cannot be uninstalled while preserving that data.
APP=${APP%/}
for path in "$DATA" "$CONFIG" "$HOME/.config/desman"; do
    case "$path/" in "$APP/"*) fail 'Data/config must be outside the install directory.' ;; esac
done
if [ "$UNINSTALL" = 1 ]; then
    confirm "Remove DESMAN app $APP, ~/.local/bin/desman and DESMAN shortcuts. Keep data and config."
    if [ "$PURGE" = 1 ]; then
        # Reject dangerous overrides before recursive deletion.
        for path in "$DATA" "$CONFIG"; do
            case "$path" in ''|/|"$HOME"|"$HOME/"|*/../*|*/..|*/./*) fail 'Unsafe purge path.' ;; esac
            [ ! -L "$path" ] || fail 'Purge directory must not be a symlink.'
        done
        confirm "PURGE permanently removes: $DATA ; $CONFIG ; $APP/config.json (keys, history, workspaces, node key)."
    elif [ -f "$APP/config.json" ]; then
        [ ! -e "$HOME/.config/desman/config.json" ] || fail 'Config backup exists; preserve app/config.json manually first.'
        mkdir -p "$HOME/.config/desman"
        mv "$APP/config.json" "$HOME/.config/desman/config.json"
        chmod 600 "$HOME/.config/desman/config.json"
    fi
    rm -f "$HOME/.local/bin/desman" "${XDG_DATA_HOME:-$HOME/.local/share}/applications/desman.desktop"
    desktop=${XDG_DESKTOP_DIR:-}
    if [ -z "$desktop" ]; then
        dirs=${XDG_CONFIG_HOME:-$HOME/.config}/user-dirs.dirs
        if [ -f "$dirs" ]; then
            desktop=$(sed -n 's/^XDG_DESKTOP_DIR="\(.*\)"$/\1/p' "$dirs" | head -n 1)
            case "$desktop" in '$HOME'*) desktop=$HOME${desktop#'$HOME'} ;; esac
        fi
    fi
    desktop=${desktop:-$HOME/Desktop}
    rm -f "$desktop/desman.desktop"
    rm -rf -- "$APP"
    if [ "$PURGE" = 1 ]; then rm -rf -- "$DATA" "$CONFIG"; fi
    echo 'DESMAN removed. uv and git were kept.'
    exit 0
fi
[ "$PURGE" = 0 ] || fail '--purge requires --uninstall.'
if ! command -v git >/dev/null 2>&1; then
    hint='Install git with your distribution package manager.'
    if command -v apt-get >/dev/null 2>&1; then hint='sudo apt-get install git'
    elif command -v dnf >/dev/null 2>&1; then hint='sudo dnf install git'
    elif command -v pacman >/dev/null 2>&1; then hint='sudo pacman -S git'
    elif command -v zypper >/dev/null 2>&1; then hint='sudo zypper install git'; fi
    fail "git is missing. Run: $hint Then run this installer again."
fi
if [ -d "$APP/.git" ]; then
    [ "$(git -C "$APP" remote get-url origin)" = "$REPOSITORY" ] || fail 'Unexpected origin; refusing to change checkout.'
    [ -z "$(git -C "$APP" status --porcelain --untracked-files=no)" ] || fail 'Tracked changes exist; preserve them before reinstalling.'
fi
if ! command -v uv >/dev/null 2>&1; then
    confirm 'Install uv (and Python 3.12 as needed) using https://astral.sh/uv/install.sh.'
    command -v curl >/dev/null 2>&1 || fail 'curl is required to install uv.'
    # Download first so a failed curl cannot be hidden by a successful shell pipeline.
    installer=$(mktemp)
    trap 'rm -f "$installer"' EXIT HUP INT TERM
    curl -LsSf https://astral.sh/uv/install.sh -o "$installer"
    sh "$installer"
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv >/dev/null 2>&1 || fail 'uv was not found after installation.'
fi
if [ -z "$DESMAN_REF" ]; then
    tags=$(git ls-remote --tags --refs --sort=-version:refname "$REPOSITORY" 'v*')
    DESMAN_REF=$(printf '%s\n' "$tags" | sed -n '1s|.*refs/tags/||p')
    if [ -z "$DESMAN_REF" ]; then DESMAN_REF=master; echo 'No v* tag found; using master until the first release tag exists.'; fi
fi
# Never interpret an override as a git option or revision expression.
git check-ref-format "refs/heads/$DESMAN_REF" >/dev/null || fail 'Invalid DESMAN_REF.'
if [ ! -d "$APP/.git" ]; then
    mkdir -p "$(dirname "$APP")"
    git clone "$REPOSITORY" "$APP"
    touch "$APP/.desman-install"
fi
cd "$APP"
git fetch origin --tags
if git show-ref --verify --quiet "refs/tags/$DESMAN_REF"; then target=refs/tags/$DESMAN_REF
else target=refs/remotes/origin/$DESMAN_REF; fi
# Keep a branch: signed global updates cannot fast-forward a detached HEAD.
git checkout -B desman-installed "$target"
if [ ! -d .venv ]; then uv venv -p 3.12; fi
uv pip install --python .venv/bin/python -e .
mkdir -p "$HOME/.local/bin"
launcher=$HOME/.local/bin/desman
if [ -e "$launcher" ] && [ ! -L "$launcher" ]; then fail "Refusing to replace $launcher"; fi
ln -sfn "$APP/.venv/bin/desman" "$launcher"
.venv/bin/desman gui --shortcut
echo 'Installed. Updates: desman global update (inspect), then --apply NUMBER. Add ~/.local/bin to PATH for terminal use.'
if [ "$START" = 1 ]; then .venv/bin/desman gui; fi
