# Hosting autocut off the laptop

The container runs the app on localhost and a Cloudflare Tunnel beside it, so
the security model is unchanged: the only way in is through the tunnel, and
Cloudflare Access does the login and sets the header the app trusts.
Transcription and retake detection run through OpenRouter; nothing needs a GPU.
The image is multi-arch (amd64 and arm64).

## Free: Oracle Cloud "Always Free" VM (recommended)

1. cloud.oracle.com -> sign up (card for verification only; the Always Free
   shapes never bill).  Home region: Mumbai or Hyderabad.
2. Compute -> Instances -> Create: image **Ubuntu 24.04**, shape
   **VM.Standard.A1.Flex** (Ampere), 2 OCPU / 12 GB is plenty (up to 4 / 24
   is free).  Boot volume 100 GB.  Add your SSH public key.  Create.
   (If "Out of capacity", retry later or the other Indian region.)
3. `ssh ubuntu@<public ip>` and run

       curl -fsSL https://raw.githubusercontent.com/marknaman05/autocut/master/deploy/server-setup.sh | sh

4. Fill `/opt/autocut/.env` (the laptop's `.env` has the keys;
   `cloudflared tunnel token autocut` on the laptop prints the tunnel token),
   then `sudo systemctl restart autocut`.
5. Stop the laptop copies: `pkill -f 'uvicorn server.app'` and
   `pkill -f 'cloudflared tunnel run'`.  Only one machine may run the tunnel.

No inbound ports need opening -- the tunnel is outbound-only.

Updates: `cd /opt/autocut && git -C src pull && sudo docker build -q -t autocut:latest src && sudo systemctl restart autocut`.
Logs: `sudo docker logs -f autocut`.  Data: `/opt/autocut/data`.

## Paid alternative: Fly.io

`fly.toml` is included: `fly launch --no-deploy --copy-config --name autocut-server --region bom`,
`fly volumes create autocut_data --region bom --size 20`, `fly secrets set ...` (same variables as
the .env above), `fly deploy`.  About $5-7/month.
