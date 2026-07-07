#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/mac/polyfun"
PLIST="$HOME/Library/LaunchAgents/com.polyfun.log_maintenance.plist"
mkdir -p "$HOME/Library/LaunchAgents"

cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.polyfun.log_maintenance</string>

  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>-lc</string>
    <string>$ROOT/scripts/maintenance/weekly_log_maintenance.sh >> $ROOT/logs/maintenance_weekly.log 2>&1</string>
  </array>

  <key>StartCalendarInterval</key>
  <dict>
    <key>Weekday</key>
    <integer>0</integer>
    <key>Hour</key>
    <integer>3</integer>
    <key>Minute</key>
    <integer>15</integer>
  </dict>

  <key>RunAtLoad</key>
  <false/>
  <key>StandardOutPath</key>
  <string>$ROOT/logs/maintenance_weekly.launchd.out.log</string>
  <key>StandardErrorPath</key>
  <string>$ROOT/logs/maintenance_weekly.launchd.err.log</string>
</dict>
</plist>
PLIST

launchctl unload "$PLIST" >/dev/null 2>&1 || true
launchctl load "$PLIST"
launchctl list | grep -E 'com\.polyfun\.log_maintenance' || true

echo "installed: $PLIST"
