# Your own search engine: SearXNG

DESMAN searches Wikipedia, Stack Overflow, arXiv and GitHub without configuration. For the
whole web, set up its local SearXNG component: no account or search key is needed.

## Set up and manage

Install and start [Docker](https://docs.docker.com/get-docker/) (Docker Desktop on Windows
and macOS), then run:

```bash
desman search setup
```

Without `--yes`, DESMAN previews the change and exits without changing anything. Add `--yes` to confirm downloading the pinned image (about 100 MB), writes settings under its
config directory, and starts `desman-searxng`, reachable only at `127.0.0.1`. It picks a free
port starting at 8888. Only after a JSON search answers does it save `SEARXNG_URL`. Run setup
again to repair or reuse the container. `--port 8889` chooses a port; `--yes` confirms setup.
The configuration wizard also offers setup under Web search. Only `settings.yml` is mounted
into the container, read-only; setup makes that file readable by the container while keeping
the settings folder private.

```bash
desman search status
desman search stop
desman search start
desman search remove
```

Removal previews the change; `desman search remove --yes` removes the container and keeps settings
and the downloaded image. `desman keys list` shows the URL; `desman keys check searxng`
checks whether it answers. `search status --json` returns the container state, URL and
search health for scripts, even without Docker (container state: unknown). Docker restarts the container with the computer unless stopped.

## Manual fallback

For an existing server, set its URL under SearXNG in `desman config`. To run Docker yourself,
write `settings.yml` in a folder of your choice:

```yaml
use_default_settings: true
general:
  contact_url: https://github.com/Temporary-Bot-5/DESMAN-P
server:
  secret_key: REPLACE_WITH_A_RANDOM_SECRET
  limiter: false
search:
  formats: [html, json]
```

Generate a random secret (for example `openssl rand -hex 32`), make the file readable
by the container (`chmod 644 settings.yml` on Linux), then run:

```bash
docker run -d --name searxng --restart unless-stopped \
  -p 127.0.0.1:8888:8080 -v /absolute/settings/folder/settings.yml:/etc/searxng/settings.yml:ro \
  searxng/searxng:2026.11.15-2d500cb6b
```

Open <http://127.0.0.1:8888>, then set that URL in `desman config`.
For installation without Docker, see the [SearXNG guide](https://docs.searxng.org/admin/installation.html).

## Troubleshooting and alternatives

A JSON-format refusal means `json` is missing from `search.formats`: correct settings and
restart the container. An unreachable server needs Docker running and the configured port
available. Individual engines can temporarily limit searches; DESMAN tries the next configured
engine. Tavily is an alternative with free monthly searches and an account/key
(`desman keys add tavily`). DESMAN does not scrape search pages meant for people.
