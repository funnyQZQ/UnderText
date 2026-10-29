#!/usr/bin/env bash
# Make this Mac look like a new user's, to test installing the .dmg and the first-run
# model download; then put everything back.
#
#   scripts/fresh_install.sh reset     # hide settings, models and llama-server
#   scripts/fresh_install.sh restore   # bring them back
#
# reset moves (never deletes) the app's settings folder, which also holds the downloaded
# models and llama-server, unlinks Homebrew's llama-server so the app can't find it, and
# resets the microphone permission so macOS asks again.
set -euo pipefail

ID=dev.undertext.desktop
SUPPORT="$HOME/Library/Application Support"
APP_DIR="$SUPPORT/$ID"
BACKUP="$SUPPORT/$ID.backup"
MARK="$BACKUP/.unlinked-llama.cpp"

quit_app() {
  osascript -e "tell application id \"$ID\" to quit" >/dev/null 2>&1 || true
  pkill -x undertext 2>/dev/null || true
  pkill -x llama-server 2>/dev/null || true
}

case "${1:-}" in
  reset)
    quit_app
    if [ -e "$BACKUP" ]; then
      echo "A backup already exists: $BACKUP"
      echo "Run '$0 restore' first (or remove the backup yourself)."
      exit 1
    fi
    if [ -d "$APP_DIR" ]; then mv "$APP_DIR" "$BACKUP"; echo "Moved settings to $BACKUP"
    else mkdir -p "$BACKUP"; fi

    if command -v brew >/dev/null && brew list llama.cpp >/dev/null 2>&1; then
      brew unlink llama.cpp >/dev/null && touch "$MARK" && echo "Unlinked Homebrew llama.cpp"
    fi
    if p=$(command -v llama-server); then
      echo "Note: llama-server is still found at $p, so the app won't download it."
    fi
    for p in /opt/homebrew/bin/llama-server /usr/local/bin/llama-server; do
      if [ -e "$p" ]; then echo "Note: $p still exists, so the app won't download llama-server."; fi
    done

    tccutil reset Microphone "$ID" >/dev/null 2>&1 && echo "Reset the microphone permission" || true
    echo
    echo "Ready. Delete /Applications/UnderText.app, install it from the .dmg and open it."
    echo "When you are done: $0 restore"
    ;;

  restore)
    quit_app
    if [ ! -d "$BACKUP" ]; then echo "Nothing to restore ($BACKUP not found)."; exit 1; fi
    if [ -f "$MARK" ]; then
      brew link llama.cpp >/dev/null && echo "Linked Homebrew llama.cpp again"
      rm "$MARK"
    fi
    if [ -d "$APP_DIR" ]; then
      TEST="$SUPPORT/$ID.test-$(date +%Y%m%d-%H%M%S)"
      mv "$APP_DIR" "$TEST"
      echo "Kept the test run (with what it downloaded) in $TEST; delete it when you no longer need it."
    fi
    if [ -n "$(ls -A "$BACKUP")" ]; then mv "$BACKUP" "$APP_DIR"; echo "Restored settings"
    else rmdir "$BACKUP"; fi
    ;;

  *)
    echo "usage: $0 reset | restore"
    exit 2
    ;;
esac
