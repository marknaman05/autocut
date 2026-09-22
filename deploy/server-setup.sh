#!/bin/sh
# Bring up autocut on a fresh Ubuntu/Debian VM (Oracle Always Free, or any
# other box).  Run as a user with sudo, from anywhere:
#
#   curl -fsSL https://raw.githubusercontent.com/marknaman05/autocut/master/deploy/server-setup.sh | sh
#
# Then put the secrets in /opt/autocut/.env (a template is written) and run
# `sudo systemctl restart autocut`.  Data lives in /opt/autocut/data.
set -e
sudo apt-get update -q
sudo apt-get install -y -q docker.io git
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER" || true

sudo mkdir -p /opt/autocut/data
sudo chown -R "$USER" /opt/autocut
cd /opt/autocut
if [ -d src ]; then git -C src pull -q; else git clone -q https://github.com/marknaman05/autocut.git src; fi

if [ ! -f .env ]; then
  cat > .env <<'ENV'
# Fill these in, then: sudo systemctl restart autocut
OPENROUTER_API_KEY=
UPLOAD_POST_API_KEY=
DODO_PAYMENTS_API_KEY=
DODO_PRODUCT_ID=
DODO_PAYMENTS_WEBHOOK_KEY=
DODO_PAYMENTS_ENVIRONMENT=test_mode
# `cloudflared tunnel token autocut` on the laptop prints this.
CLOUDFLARE_TUNNEL_TOKEN=
# Sign in with Google (a *Web application* OAuth client whose redirect URI is
# <AUTOCUT_PUBLIC_URL>/auth/callback).  Session secret: any 32+ random chars.
AUTOCUT_GOOGLE_CLIENT_ID=
AUTOCUT_GOOGLE_CLIENT_SECRET=
AUTOCUT_SESSION_SECRET=
AUTOCUT_PRO=marknaman05@gmail.com
AUTOCUT_PUBLIC_URL=https://autocut.mynameisnaman.in
AUTOCUT_RETENTION_DAYS=7
AUTOCUT_MAX_UPLOAD_BYTES=2147483648
AUTOCUT_MAX_JOBS_PER_USER=5
AUTOCUT_MAX_BYTES_PER_USER=3221225472
ENV
  echo ">> wrote /opt/autocut/.env -- fill it in"
fi

sudo docker build -q -t autocut:latest src

sudo tee /etc/systemd/system/autocut.service >/dev/null <<'UNIT'
[Unit]
Description=autocut (web app + Cloudflare Tunnel)
After=docker.service
Requires=docker.service

[Service]
Restart=always
RestartSec=5
ExecStartPre=-/usr/bin/docker rm -f autocut
ExecStart=/usr/bin/docker run --name autocut --env-file /opt/autocut/.env -v /opt/autocut/data:/data --memory=6g autocut:latest
ExecStop=/usr/bin/docker stop autocut

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable --now autocut
echo ">> autocut is starting.  Logs: sudo docker logs -f autocut"
echo ">> to update later: cd /opt/autocut && git -C src pull && sudo docker build -q -t autocut:latest src && sudo systemctl restart autocut"
