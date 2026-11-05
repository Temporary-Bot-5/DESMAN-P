# Running desman-registry on a server

`desman-registry` is the registry of desman-global (ADR 0004). Nodes register with it, send
heartbeats and fetch the records signed by the root. It runs on the owner's server and speaks
plain HTTP on `127.0.0.1`. **TLS comes from a reverse proxy** in front of it (Caddy or nginx,
both described below). Never expose the plain port to the internet.

What the registry keeps, in its own directory (`~/.local/share/desman-registry`, or
`$DESMAN_REGISTRY_DIR`, or `--dir`):

| File | What it is |
|---|---|
| `registry.db` | SQLite: nodes, signed records, nonces, the journal |
| `root.key` | The root's private key. **Readable by its owner only (0600). Keep an offline copy.** |
| `operator.key` | The registry's own operator key, created by the first `assign` |

The registry reads no `config.json` and keeps no API key. Nodes send only their public key,
what they lend (provider, model, fingerprint of the API key) and signatures.

## 1. Install

```bash
sudo useradd --system --create-home --home-dir /srv/desman desman
sudo -iu desman
git clone https://github.com/Temporary-Bot-5/DESMAN-P.git
cd DESMAN-P
python3 -m venv .venv && .venv/bin/pip install -e .
```

The clock matters: a request signed more than 5 minutes away from the server's time is
refused. Keep NTP on (`timedatectl` should show `System clock synchronized: yes`).

## 2. The root key

The root key signs assignments of operators and revocations. Its public half is
`src/desman/net/root.pub` in the repository, and nodes trust nothing else. Put the private half
in the registry's directory:

```bash
install -m 700 -d ~/.local/share/desman-registry
install -m 600 /path/to/root.key ~/.local/share/desman-registry/root.key
```

A key whose public half is not `root.pub` is refused before anything is signed. **Make an
offline copy of `root.key`** (an encrypted USB stick in a drawer, not another server). If it is
lost, nothing new can be signed. If it leaks, `root.pub` has to change in a release, and every
person has to approve that change on their machine (`desman protected approve`).

## 3. Assign the registry's operator

```bash
.venv/bin/desman-registry assign            # the registry's own key, all scopes, 90 days
.venv/bin/desman-registry log
```

`assign PUBLIC_KEY.pem --scope tasks --days 30` assigns someone else's key instead. Renew the
assignment before it ends: run the same command again.

### The trust list

Nodes take no work unless they hold a fresh **trust list**: a record signed with the root key
that names every assignment and revocation in effect. If the registry hid a revocation (if it
was broken into, say), nodes would notice something missing from the list and stop taking work.
Each `assign` and `revoke` signs a new list lasting 7 days. Nodes accept a list only for
2 days after it is signed, so the daily timer matters. Sign one again every day from a timer (`crontab -e` as `desman`):

```cron
0 4 * * * /srv/desman/DESMAN-P/.venv/bin/desman-registry trust sign --days 7
```

`desman-registry trust show` says which list is handed out and until when. A list lasts 30
days at most (`--days`), but concealment is bounded by the 2-day age limit.

## 4. Run it as a service

`/etc/systemd/system/desman-registry.service`:

```ini
[Unit]
Description=desman-registry
After=network-online.target
Wants=network-online.target

[Service]
User=desman
WorkingDirectory=/srv/desman/DESMAN-P
ExecStart=/srv/desman/DESMAN-P/.venv/bin/desman-registry serve --host 127.0.0.1 --port 8470
Restart=on-failure
NoNewPrivileges=yes
ProtectSystem=strict
ReadWritePaths=/srv/desman/.local/share/desman-registry
PrivateTmp=yes

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now desman-registry
journalctl -u desman-registry -f
```

## 5. TLS in front of it

The proxy must pass the request path and query **unchanged**: the node signs them, and a
rewritten path is refused as forged. It must also pass the `X-Desman-*` headers as they are.
Bodies are small (at most 64 KiB).

### Caddy (certificates automatic)

`/etc/caddy/Caddyfile`:

```caddyfile
registry.example.org {
    request_body {
        max_size 64KB
    }
    reverse_proxy 127.0.0.1:8470
}
```

### nginx (with certbot)

```nginx
server {
    listen 443 ssl http2;
    server_name registry.example.org;
    ssl_certificate     /etc/letsencrypt/live/registry.example.org/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/registry.example.org/privkey.pem;

    client_max_body_size 64k;

    location / {
        proxy_pass http://127.0.0.1:8470;   # no URI after the port: the path stays as sent
        proxy_set_header Host $host;
        proxy_read_timeout 30s;
    }
}
```

Only ports 80 and 443 need to be open: `sudo ufw allow 80,443/tcp`. Leave port 8470 closed.

### Tailscale Funnel (no domain, no open ports)

A machine behind a home router can be reached without port forwarding and without publishing
its address: Tailscale Funnel gives it a stable `https://NAME.TAILNET.ts.net` with TLS and
passes the path unchanged. Funnel serves ports 443, 8443 and 10000; take another one if 443 is
used already (`--https=8443`, then the address carries `:8443`).

```bash
tailscale funnel --bg 8470            # or: tailscale funnel --bg --https=8443 8470
tailscale funnel status
tailscale funnel --https=443 off      # to take it down
```

The first time, Tailscale may ask to allow Funnel for the tailnet in its admin console. Inside
the tailnet the name resolves to the machine's 100.x address, so to test the public path from
the machine itself, resolve it with a public DNS server (`dig @1.1.1.1 NAME.TAILNET.ts.net`)
and pass that address to `curl --resolve`. The registry is reachable only while the machine
is on; nodes wait with backoff and go on with their own work meanwhile.

People point their nodes at it with `desman global join --registry https://registry.example.org`.
Once the address is final it becomes `net.DEFAULT_REGISTRY` in a release.

## 6. Watching it

```bash
desman-registry nodes            # who takes part, what they lend, when seen last
desman-registry log --limit 100  # registrations, departures, assignments, revocations, refusals
desman-registry revoke FINGERPRINT --reason "why"   # a node or an operator, signed by the root
```

Every refused request is journaled with the reason and the key it claimed. A burst of
`forged` or `replay` entries from one key usually means a broken clock or a copied `node.key`.

### Jobs for the network (G.6)

The operator offers work to nodes as signed jobs; each job is one self-contained subtask.

```bash
desman-registry job add --description "What to do" --acceptance "How it is judged" \
    [--kind text|code|analysis] [--runs-code] [--input NAME=TEXT ...] [--days 7]
desman-registry jobs [--status operator]   # every job and where it is
desman-registry inbox                      # jobs waiting for your word
desman-registry job show JOB               # the job, each result and verdict handed in
desman-registry job settle JOB accept|again|drop
desman-registry owner NODE NAME            # nodes of one owner never recheck each other
```

`job add` signs with `operator.key`, which must be assigned the `tasks` scope (section 3).
A job's term ends with the operator's assignment at the latest; once it ends, `again` is
refused: add the job anew.
A node does the job (a lease of 10 minutes, kept alive by the node), then a node of **another
owner** reviews the result. Accepted, the job is done. The job comes to the inbox when the
review rejects the result (`disputed`), when nodes say twice that they cannot do it
(`failed`), or when three nodes take it and fall silent (`leases_lost`). Nothing in the inbox
is decided without you. `again` lends the job from the start.

Until you say otherwise, every node counts as its own owner. Name your own machines with
`owner`, or they will recheck each other's work.

## 7. Backups and restoring

`backup` takes a consistent copy while the server runs (SQLite backup API), readable by its
owner only:

```bash
desman-registry backup /srv/desman/backups/registry-$(date +%F).db
```

A daily timer (`crontab -e` as `desman`):

```cron
15 3 * * * /srv/desman/DESMAN-P/.venv/bin/desman-registry backup /srv/desman/backups/registry-$(date +\%F).db
```

Copy the backups off the server. To restore one:

```bash
sudo systemctl stop desman-registry
cd ~/.local/share/desman-registry
mv registry.db registry.db.broken; rm -f registry.db-wal registry.db-shm
install -m 600 /srv/desman/backups/registry-2026-11-01.db registry.db
sleep 300   # see below
sudo systemctl start desman-registry
```

The copy does not know the nonces seen after it was taken. A request is good for 5 minutes, so
start the restored registry at least 5 minutes after stopping the old one: by then no request
it saw can be replayed. Records signed after the backup are lost with it. Sign them again: a
revocation, for one, must be redone at once. Then sign a trust list (`trust sign`): nodes keep
the newest one they saw, which may name records the copy has lost, and they take no work until
the restored registry hands out every record a list still names.

## 8. Updating

```bash
sudo -iu desman
cd DESMAN-P && git pull && .venv/bin/pip install -e .
exit
sudo systemctl restart desman-registry
```

The registry migrates its database on start. Take a backup first.

Independent proposal checks (G.7b) appear in `desman-registry proposal show ID`: the checking
node, outcome, verdict, whether the metrics match the submission, reason, and both benchmark
snapshots. The checker belongs to another owner and runs only the fake benchmark; the
operator still decides. The base must be HEAD or an ancestor in the checker's checkout.
Fake children receive a minimal environment and an empty home and config. When both
snapshots include workflow limits, the checker uses those limits to compare metrics; older
snapshots run with defaults and leave the match unknown. Signed checks survive a registry
re-registration and are handed in on a later poll. `--json` includes the checks.


## Accepting a proposal and releasing

Inspect the submission and its independent checks, then preview acceptance into the owner's
explicit checkout (it must be on a branch, with clean target files):

```bash
desman-registry proposal show ID
desman-registry proposal accept ID --repo /path/to/DESMAN-P
# Inspect the diff, checks, branch and HEAD, then repeat with the displayed hash:
desman-registry proposal accept ID --repo /path/to/DESMAN-P --confirm SHA256
# Or reject it, recording why (only when no node is checking it):
desman-registry proposal reject ID --reason "Reason for declining"
desman-registry release list
desman-registry release show 1 --json
```

Confirmation needs `root.key` in the registry directory (or `--root-key PATH`). It builds a
commit containing only the proposed target texts, fast-forwards the supplied checkout and
records a release signed by the root, numbered consecutively. Repeating acceptance is refused.
The preview is read-only and needs no root key; `--json` is available for preview, acceptance,
and release list/show. A changed HEAD, branch, submission or set of checks requires a fresh
preview. Rejections and acceptances are final.

DESMAN never pushes. Push the resulting branch yourself, as the command's output explains:

```bash
git -C /path/to/DESMAN-P push origin BRANCH
```

Git and the registry database cannot commit atomically. A crash or database failure after the
checkout advances can leave an **unrecorded commit** (an error names it when possible). Inspect
`git log` and `desman-registry release list` before retrying. Revert that commit with git,
then preview again; do not assume it was signed or recorded. Back up the registry and root
key as described above. Delivery to nodes and installation are separate work in G.8.

The registry serves the newest five root-signed releases at signed GET `/v1/releases` for active nodes.

Inspect incoming bug reports and shared memory with `desman-registry reports [--kind bug|memory]`; read one with `desman-registry report show ID`.

Messenger messages are retained for 30 days (old messages are deleted on send). The registry
operator can read them; TLS protects transport, and end-to-end encryption comes later.
Complaints are kept with a copy of the cited message, so message retention does not erase evidence.
