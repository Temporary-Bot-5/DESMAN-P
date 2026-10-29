# Your own search engine: SearXNG

DESMAN's `web_search` tool can always search Wikipedia, Stack Overflow, arXiv and GitHub. To
search the whole web without an account or an API key, run [SearXNG](https://docs.searxng.org)
on your machine: a free, open-source metasearch engine that asks Google, Bing, DuckDuckGo and
others for you and sends back one list of results.

## Start it (Docker, about two minutes)

1. Make a folder for its settings and write `settings.yml` into it. DESMAN reads results as JSON,
   which SearXNG has to be told to allow:

   ```bash
   mkdir -p ~/.config/searxng
   cat > ~/.config/searxng/settings.yml <<EOF
   use_default_settings: true
   server:
     secret_key: "$(openssl rand -hex 32)"
     limiter: false
   search:
     formats:
       - html
       - json
   EOF
   ```

2. Start it, reachable only from this machine (`127.0.0.1`), restarted with the computer:

   ```bash
   docker run -d --name searxng --restart unless-stopped \
     -p 127.0.0.1:8888:8080 \
     -v ~/.config/searxng:/etc/searxng \
     searxng/searxng
   ```

3. Open <http://localhost:8888> in a browser: a search page means it works.

4. In `desman config`, pick **SearXNG** under *Web search* and set its URL to
   `http://localhost:8888`. The wizard checks it right away.

Without Docker, see SearXNG's own
[installation guide](https://docs.searxng.org/admin/installation.html).

## When something is wrong

- **"SearXNG refuses the JSON format"**: `json` is missing under `search.formats` in
  `settings.yml`. Fix it, then `docker restart searxng`.
- **"cannot be reached"**: the container is not running (`docker ps`, `docker start searxng`) or
  the URL is not the one in step 2.
- **Few or no results**: the big engines sometimes refuse SearXNG for a while (too many
  searches). It recovers by itself; DESMAN falls back to the next engine you set up, if any.

## Other ways to search the whole web

Both are set in `desman config` under *Web search*, and are tried after SearXNG:

- **Tavily**: an API with free monthly searches; needs a free account and its key
  (`desman keys add tavily`).
- **DuckDuckGo**: no account, no key, nothing to run. DESMAN reads DuckDuckGo's results page the
  way a browser does. This is unofficial: it can break when the page changes, too many searches
  get a captcha or a temporary block, and DuckDuckGo's terms may not allow it. Off until you turn
  it on; at most one search every 5 seconds.
