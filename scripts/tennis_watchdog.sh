#!/bin/bash
# Restart the tennis tick recorder if it is not running. Cron runs this every
# 5 minutes; the recorder itself exits cleanly if its pidfile is deleted.
DIR="/Users/ethan/Documents/Duke Year 1/weather-bot"
PIDFILE="$DIR/data/capture/tennis_recorder.pid"
LOG="$DIR/data/capture/logs/tennis_recorder.log"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    exit 0
fi
rm -f "$PIDFILE"
cd "$DIR" || exit 1
PYTHONPATH=. nohup /opt/anaconda3/bin/python live/tennis_recorder.py >> "$LOG" 2>&1 &
echo "$(date -u +%FT%TZ) watchdog: restarted recorder (pid $!)" >> "$LOG"
