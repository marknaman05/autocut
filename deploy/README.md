# Hosting autocut off the laptop (Fly.io)

The container runs the app on localhost and a Cloudflare Tunnel beside it, so
the security model is unchanged: the only way in is through the tunnel, and
Cloudflare Access does the login and sets the header the app trusts.
Transcription and retake detection run through OpenRouter; nothing needs a GPU.

One-time, from this directory's parent:

```
brew install flyctl
fly auth login                                  # opens the browser; add a card once
fly launch --no-deploy --copy-config --name autocut-server --region bom
fly volumes create autocut_data --region bom --size 20
fly secrets set \
  OPENROUTER_API_KEY=... \
  DODO_PAYMENTS_API_KEY=... DODO_PRODUCT_ID=... DODO_PAYMENTS_WEBHOOK_KEY=... DODO_PAYMENTS_ENVIRONMENT=test_mode \
  UPLOAD_POST_API_KEY=... \
  CLOUDFLARE_TUNNEL_TOKEN="$(cloudflared tunnel token autocut)"
fly deploy
```

Then stop the laptop copies (`pkill -f 'uvicorn server.app'`, `pkill -f 'cloudflared tunnel run'`).
The same tunnel id keeps `autocut.mynameisnaman.in` pointing at whichever
machine runs it; run only one at a time.

Day to day: `fly logs`, `fly ssh console`, `fly deploy` after a push.
Data (uploads, renders, the SQLite database) lives on the `autocut_data`
volume at `/data`; `fly volumes snapshots list autocut_data` for backups.

Moving to any other Linux host is the same image: build with the Dockerfile,
mount a disk at `/data`, set the same environment, run.
