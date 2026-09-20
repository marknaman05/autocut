#!/bin/sh
# App on localhost only; the tunnel is the only way in.  If either process
# dies the container exits and the platform restarts it.
set -e
mkdir -p /data/work
/app/.venv/bin/uvicorn server.app:app --host 127.0.0.1 --port 8000 &
APP=$!
if [ -n "$CLOUDFLARE_TUNNEL_TOKEN" ]; then
  cloudflared tunnel --no-autoupdate run --token "$CLOUDFLARE_TUNNEL_TOKEN" &
  TUNNEL=$!
  wait -n $APP $TUNNEL 2>/dev/null || wait $APP
else
  echo "CLOUDFLARE_TUNNEL_TOKEN not set: app is reachable only inside the machine" >&2
  wait $APP
fi
