# Harness v02: HTTP API and the page

Checked against `paratroop_harness_02/server.py` and `page.html` on 2026-09-15, after the page redesign.

## The server

- **Server:** `server.py` runs a `ThreadingHTTPServer` on `127.0.0.1`, port 8771 by default (`--port`).
- **One harness:** it holds one `Harness` plus an `App` with:
  - **live runs:** the working memory of turns in progress. Once there are more than 50 (`KEEP_RUNS`), older finished
    runs are dropped from memory; they're still on disk in `runs/`;
  - **test status** per module: `{running, error}`.
- **Threads:** each turn runs in its own daemon thread (`Harness.think`), and so does each module test.

### Rules for every request

| Check | Response |
|---|---|
| `Host` isn't `127.0.0.1`, `localhost` or `::1` (any port) | 403 `{"error": "this page only answers on localhost"}` |
| POST with an `Origin` whose host:port isn't the `Host` | 403 `cross-site request refused` |
| POST `Content-Type` isn't `application/json` | 415 `send JSON` |
| POST body empty or over 64 KB | 413 `empty or too big` |
| POST body isn't a JSON object | 400 `bad JSON` |
| A `HarnessError` while handling | 400 `{"error": "<message>"}` |
| Unknown path | 404 `not found` |

- **Every response** has `Cache-Control: no-store` and `X-Content-Type-Options: nosniff`.
- **The page (`GET /`)** also gets `Content-Security-Policy: default-src 'none'; script-src 'unsafe-inline';
  style-src 'unsafe-inline'; img-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'none';
  frame-ancestors 'none'`.
- **Errors** are always `{"error": "..."}`.

## Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/` | | `page.html`, the redesigned page |
| GET | `/classic` | | `page-classic.html`, the page as it was before 2026-09-15 |
| GET | `/api/state` | | [the state object](#the-state-object) |
| POST | `/api/run` | `{message, conversation?}` | `{id, conversation}` |
| GET | `/api/runs/<id>/events` | | server-sent events with working memory, until the turn ends |
| GET | `/api/runs/<id>` | | working memory of a run |
| GET | `/api/conversations` | | `{conversations: [...]}`, the 50 most recent |
| GET | `/api/conversations/<id>` | | one conversation |
| POST | `/api/choose` | `{module, choice}` | the state object |
| POST | `/api/toggle` | `{kind, id, on}` | the state object |
| POST | `/api/settings` | `{module, key, value}` | the state object |
| POST | `/api/test` | `{module}` | `{ok: true}`, and the test runs in the background |
| POST | `/api/forget` | `{id}` | the state object |

Run and conversation ids match `\d{8}-\d{6}-[0-9a-f]{6}`. Other ids 404 at the router level.

### POST `/api/run`

Starts a turn.

- **`message`:** 1 to 8000 characters after trimming. Otherwise 400 "say something (up to 8000 characters)".
- **`conversation`:** optional. Without it, a new conversation is created.
  - A malformed id gives 400 "that isn't a conversation id".
  - An unknown one gives 400 "there's no conversation <id>".
- **One turn per chat:** a second message while the same conversation is still running gives 400 "still thinking about
  the last message in this chat".
- **Returns** right away with `{id, conversation}`. Follow the turn with the events stream.

### GET `/api/runs/<id>/events`

`text/event-stream` for runs still held in memory (404 "no such run" otherwise).

- **`data: <working memory JSON>`** is sent every time working memory changes. The server waits up to 15 s for a change.
- **`: still thinking`** is a comment line sent when 15 s pass with no change.
- **The stream ends** after the first snapshot whose `status` isn't `running`. The harness saves the run and the chat
  turn *before* that snapshot exists, so it's safe to fetch the conversation as soon as the stream ends.

### GET `/api/runs/<id>`

Returns the live snapshot if the run is in memory, else `runs/<id>.json`, else 404 "no such run". The fields are in
[architecture.md → Working memory](architecture.md#working-memory).

### GET `/api/conversations` and `/api/conversations/<id>`

- **The list:** the 50 most recently changed chats as `{id, first, turns, updated}`. `first` is the first message cut
  to 80 characters. Empty chats are skipped.
- **One chat:** `{id, started, updated, turns: [{run, message, reply, status, note, goal, asked, seconds, at}]}`, or
  404 "no such conversation".

### POST `/api/choose`

Sets a module's model. `choice` must be one of `options` for that module that's `up` right now; status is checked
fresh, not from the cache. Otherwise 400 "<choice> isn't offered for <module> right now". It's saved to `choices.json`.

### POST `/api/toggle`

`kind` is `"module"` or `"tool"`, and `on` must be a JSON boolean (else 400 "send kind (module or tool), id and on (true
or false)"). Unknown ids, unbuilt tools ("command_line isn't built yet") and other kinds give 400. Saved to
`switches.json`.

### POST `/api/settings`

`key` must be one of the module's settings. `value` must be a boolean for bool settings, or a number in range. Errors
give 400 with the label and range. Saved to `settings.json`; a default value is removed from the file. The full list is
in [architecture.md → Settings](architecture.md#settings).

### POST `/api/test`

Starts `test_module(module)` in a thread.

- **Refused at once (400):** unknown modules, "no checks written for <module>", and "<module> is already being tested".
- **Reported later:** errors raised inside the test, like "<Name> is switched off, so there's nothing to test" or
  "<Name>'s model isn't trained yet...", show up in the state object as that module's `test_error`.
- **The score** is saved to `scores.json` under the module and its current choice, and reappears as `score` in the state
  object.

### POST `/api/forget`

Deletes one long-term memory by id. Returns 404 "no memory with that id" if it isn't there.

## The state object

`GET /api/state` and every POST except `/api/run` and `/api/test` return this. It reloads `scores.json` first and uses
provider status cached for up to 15 s.

```text
{
  modules: [                                   // in harness.json order
    { id, name, color, does, on,
      choice,                                  // "provider/model" or null
      options: [ {choice, provider, label, model, up, loaded, note?} ],   // note: "not trained yet"
      score: {passed, total, avg_seconds, when, results: [{ok, input, why}]} | null,
      has_checks, testing, test_error,
      settings: [ {key, label, type, min, max, step, default, value} ] }
  ],
  tools: [ {name, on, built, kind, does} ],    // kind: "tool" or "store"
  providers: { name: {label, up, error} },
  memory: { count, recent: [ {id, text, created} ] },   // last 20, newest first
  examples: [ {label, message} ],
  max_cycles,
  conversations: [ {id, first, turns, updated} ]        // 20 most recent
}
```

- **`score`** is the saved score for the module's *current* choice, so switching models shows that model's score.
  Inputs are cut to 80 characters and reasons to 120.
- **`loaded`** comes from the provider's `/v1/models` listing, where the server reports it.

---

## What the page shows

`page.html` is one file of plain JavaScript that builds everything from text nodes, never HTML from the server.

### Header

The title, then one dot per provider: green when up, red when down. Hovering shows "reachable" or the error.

### The brain (left)

- **Layout:** the 11 module cards sit on a circle around a dashed "Working memory" hub, **ordered clockwise in pipeline
  order** (Router → Perception → Attention → Recall → Math Language → Math → Planner → Predictor → Critic → Decision →
  Language), not `harness.json` order. The geometry is computed, not fixed: angles are allocated in proportion to each
  card's footprint along the ring and the card scale shrinks (1.0 → 0.78) until everything fits, with a top-to-bottom
  pipeline fallback when it can't. Measured 0 overlaps from 340 px to 1000 px of panel width.
- **Arcs, not spokes:** an arrowed arc joins each switched-on module to the next one that's on. A hop over switched-off
  modules cuts inside the ring. A switched-off card is a dashed outline at 92% scale with no arc and no spoke.
- **The cycle** is a dashed inner arc from Decision to Planner labelled "loops up to N×" from `max_cycles`, faded when
  the Planner is off or no tools are on, and lit once a run has more than one cycle.
- **Each card carries** the name, the short model name, a score pill and a `role="switch"` toggle (POST `/api/toggle`).
- **Clicking a card opens an inspector popover** with the rest, and it survives a state reload:
  - on/off with an explanation;
  - a model dropdown grouped by provider (POST `/api/choose`). A model loaded on its server shows "● loaded"; untrained
    ones show "(not trained yet)" and are disabled; a provider that's down is labeled "(offline)"; a saved pick that
    isn't reachable shows as "<choice> (not reachable)";
  - the score badge, "passed/total · avg s": green when every check passed, yellow at half or more, red below half,
    **plus the per-check pass/fail list** — which the old page had only in a `title` tooltip;
  - Test (POST `/api/test`), which reads "testing…" while it runs. The page polls state every 3 s during a test and
    shows `test_error` as a toast;
  - the settings inline (POST `/api/settings` on every change, with "default" or "default: X" under each field). There
    is no separate ⚙ dialog any more.
- **While a turn runs:** the working module's card glows, its arc animates in the module's color, and a running tool's
  row glows and counts. The hub shows the goal plus round "N of M", observations, memories and model calls. When idle,
  it shows the long-term memory count, the cycle limit and the chat's turn count.
- **Below the circle:**
  - "Tools and memory (not models) · click to switch": one chip per tool, crossed out when off and disabled when not
    built;
  - "Long-term memory · N": the 20 newest memories, each with a Forget button (POST `/api/forget`).

### The chat (right)

- **Header:** the chat's title (its first message), a "Recent chats…" dropdown, and New chat.
- **Thread:** your messages on the right, replies on the left. A reply renders `**bold**` and `*italic*`, and nothing
  else. A stopped turn shows "No reply needed." plus the note; an error shows the note in red.
- **Composer:** Send, or ⌘↩ / Ctrl+↩. Example chips fill the box from `examples`.
- **Sending:** POST `/api/run`, then follow `/api/runs/<id>/events`. The live turn shows "thinking… <module>" and an open
  "Thinking live" board. When the stream ends, the page reloads the conversation and state.
- **Blocked while thinking:** switching modules or tools, New chat and Recent chats all toast "Wait for this reply
  first."
- **Remembered chat:** the open chat's id is kept in `localStorage` (`paratroop02.conversation`), so a reload reopens it.
- **Refresh:** state reloads every 20 s when the tab is visible, no dropdown is focused, and no turn is running.
- **Lost connection:** if the event stream drops, the toast says "Lost the live view. It keeps thinking on the server;
  reopen the chat in a bit."

### "How it got there"

Each reply has a collapsed "How it got there · Ns". Opening it fetches `GET /api/runs/<run>` once and renders:

| Section | From working memory |
|---|---|
| Status line | "done in Ns · K model calls", plus the note |
| Perception | each observation: kind tag, text, when, and a relevance bar. The subtitle has attention's "for me / not for me" and why |
| Math Language | each problem: id, expression, unit; "= answer unit · by Math/calculator"; "the calculator caught N"; "worked as ..."; calculator steps with their reason; the model's work lines; errors |
| Recall | the queries and found count (or "long-term memory is empty"); kept memories with relevance |
| Cycle N (one per cycle) | candidates with a ✓ on the pick; tool, action and target; predicted helpful, outcome and risk; "dropped by guards", "changed by guards" and "skipped"; Critic issues colored by severity, and its question; "Decision (model/rule/router): #n — why"; the action with ok or failed, seconds and the first 1500 characters of the result; for Math, "X of Y steps by <model>, K caught by the calculator" or "unchecked (the calculator is off)", each step, and the work lines |
| Remembered | memories saved this turn |
| Model calls | every trace entry: name (with part, like "Recall (rank)" or "Language (make)"), model, seconds, error or "retried" |

The router's own output (`route`) isn't rendered. It shows up only in the Model calls table and in decisions made
`by: router`.
