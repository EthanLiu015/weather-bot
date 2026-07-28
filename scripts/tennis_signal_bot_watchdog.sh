#!/bin/bash
# Restart the tennis signal bot if it is not running. Mirrors
# scripts/tennis_watchdog.sh's pattern for the (separate) tick recorder.
# Safe to cron even while disabled: the bot itself exits immediately (no
# pidfile written) if TENNIS_ENABLED is false, so this just becomes a
# harmless no-op restart attempt every cycle.
DIR="/Users/ethan/Documents/Duke Year 1/weather-bot"
PIDFILE="$DIR/data/capture/tennis_signal_bot.pid"
LOG="$DIR/data/capture/logs/tennis_signal_bot.log"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    exit 0
fi
rm -f "$PIDFILE"
cd "$DIR" || exit 1
PYTHONPATH=. nohup /opt/anaconda3/bin/python live/tennis_signal_bot.py >> "$LOG" 2>&1 &
echo "$(date -u +%FT%TZ) watchdog: restarted signal bot (pid $!)" >> "$LOG"
