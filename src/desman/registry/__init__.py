"""desman-registry: the registry of desman-global, on the owner's server (ADR 0004; ROADMAP G.4).

Its own program (`desman-registry`), its own SQLite (`store`), plain HTTP behind a reverse proxy
that gives TLS (`server`, docs/registry-deploy.md). It shares only what both sides must agree on
with the node — `net.sign`, `net.request`, `net.protocol` — and imports nothing that reads
`config.json`, nor keeps any API key.
"""
