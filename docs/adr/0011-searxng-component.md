# ADR 0011: SearXNG, DESMAN's own search engine for the whole web

Status: accepted (owner's request, 2026-11-15: "SearXNG … should become one of the components of
this program")

## Context

Many requests need the whole web: prices, news, schedules, today's rates. Without a search engine
for it, `web_search` has only Wikipedia, Stack Overflow, arXiv and GitHub, and such tasks fail
(user test 2026-11-15: the trip cost with today's AI-95 price ended waiting for the person after
~15 attempts). Tavily works but needs an account and gives a limited number of searches a month.
DESMAN never reads search engines' pages meant for people (`core/search.py`), so "googler"-like
scraping is out.

SearXNG is a free, open-source metasearch server that asks Google, Bing, DuckDuckGo, Brave and
others in the way their operators allow and answers JSON. Claude set it up on the owner's machine
from docs/searxng.md (Docker, bound to 127.0.0.1): the same task then found the price and the
cost with one NVIDIA key and no other key. Until now the person had to do that by hand from the
doc; the owner wants every DESMAN user to have it.

## Decision

1. **DESMAN sets SearXNG up and looks after it**, as it does local models: `desman search
   setup|status|start|stop|remove` (module `searxng.py`, bodies shared by the CLI, the wizard and
   the chat's catalog). Everything that downloads or starts something waits for the person's
   "yes" (the size of the image is shown first); `--yes` for scripts.
2. **Docker is the one way it runs** (Linux, macOS, Windows with Docker Desktop: the same
   commands). `setup` checks that the `docker` command exists and its daemon answers; otherwise it
   says how to get Docker (link) and that Tavily is the alternative, and changes nothing.
3. **What setup does:** writes `searxng/settings.yml` in DESMAN's config directory (a random
   `secret_key`, `json` among `search.formats`, `limiter: false`, a `general.contact_url` so its
   Wikipedia/Wikidata engines load), pulls an image pinned by version (the tag DESMAN was tested
   with, never `latest`), starts the container `desman-searxng` bound to `127.0.0.1` only, on a
   free port starting at 8888, with `--restart unless-stopped`, waits until it answers a JSON
   search (`search.check_searxng`), then sets `SEARXNG_URL`. Run twice, it repairs instead of
   duplicating (an existing container is reused or recreated).
4. **Docker is called with a fixed argument list** (no shell, no generated text in it) through
   one helper; it is DESMAN's own administration, not generated code, so it does not go through
   the sandbox. A failure is reported in plain words with Docker's own message.
5. **Visible everywhere it matters:** the wizard's *Web search* section offers "set up SearXNG on
   this computer" first; `keys list`/`keys check` show SearXNG and whether it answers; the run's
   warning about a missing whole-web engine names `desman search setup`; the installers offer it
   at the end when Docker is present; the manual's topic says all this.
6. **Results without a host are dropped** (SearXNG sometimes returns `http:///go/v1/…`).

## Not chosen

- **One public SearXNG for all users** (the owner's server): the engines limit a shared address
  quickly, every user's searches would pass through one machine, and it would be a single point
  of failure. A node lending its search through desman-global may come later, like lent models.
- **Installing SearXNG without Docker** (a Python virtual environment): officially Linux only,
  with more moving parts; may be added later for Linux.
- **Scraping search pages** (googler and the like): refused by the engines and against
  `core/search.py`'s rule.
