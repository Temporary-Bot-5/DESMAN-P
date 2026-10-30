# Demo D9 on real providers

Demo D9: a request typed in ordinary words in the chat becomes a task, you watch its progress,
and the result comes back to the same chat. `tests/test_demo_d9.py` runs this on a scripted
model. This page is the manual check with real free providers and real web search. It takes
about ten minutes.

## Before you start

- At least one model key: Groq, Google AI Studio or another free provider from `desman config`.
- A way to search the whole web: a free Tavily key (https://app.tavily.com), or your own
  SearXNG (`docs/searxng.md`). Without one, `web_search` covers only Wikipedia, Stack Overflow,
  arXiv and GitHub, which is not enough for tickets.
- Leave `network_access` off for now: you turn it on in the chat (step 2).

Use your own data folders, so the check leaves your real tasks and config alone:

```bash
export DESMAN_DATA_DIR=/tmp/desman-d9/data DESMAN_CONFIG_DIR=/tmp/desman-d9/config
desman config          # language, one model key, network_access left off
```

## The check

Start the chat with bare `desman`, then type the following, each in your own words:

1. **A key through the chat.** "I want you to search the web, here is my Tavily key"
   (do not paste the key into the message).
   - Expected: a hidden field opens for the key, and the key is never shown. Then
     `✓ Added tvly…xxxx to Tavily`.
   - If you paste the key into the message anyway, it is cut out with a warning.
2. **A permission through the chat.** "Let your agents go online."
   - Expected: a `Confirm` window, built by DESMAN and not by the model, showing
     `network_access` and `Apply this change?`. Answer `y`, and you see
     `✓ network_access granted`.
   - Try "n" once as well: the window says `Nothing changed.`, and the model does not insist.
3. **The request.** "Find me cheap flights from Berlin to Lisbon next month."
   - Expected: `Planning…`, then `Task … started: …`. While it runs you see a subtask bar and a
     line for each module: provider, model, key (masked), role, calls, tokens and the limit
     left. Then `Task … is done.` and a short retelling with flights, prices and links.
   - If the interlocutor asks a question back (dates, airports), answer it. That is a
     `clarify` turn and it is fine.

## What to look at afterwards

```bash
desman list                    # the task, linked to the chat session
desman audit <ref>             # tool_called web_search, and every status change
desman usage                   # calls and tokens per provider and key
grep -r "tvly-" "$DESMAN_DATA_DIR" || echo "no key in the data folder"
```

- The result's links lead to real pages. Prices may be out of date: the search engine's summary
  is not a booking site.

## If it goes wrong

- `429` or `quota`: the progress line says `waiting for a limit` or `quota used up`. The task waits
  or pauses, and `desman resume <ref>` continues it later. This is expected on free tiers, not a
  failure of the demo.
- The task waits for you (`needs_human`): the chat shows the decisions, the same as in
  `desman inbox`.
- Write down what happened, the provider and model, and `desman audit <ref> --json`, and open an
  issue.
