# Adding a trained-model module

How to plug a new *kind* of model trained in SMMOL into paratroop_harness_02, the way Router (`classifier`), Math
(`math`) and Math Language (`math_language`) were added. Checked against the code on 2026-09-15.

If your model is just another model of an existing kind, like a new router or math model, skip to
[step 3](#3-the-provider-entry): add a provider entry and you're done. `smMATH001-a` was added that way, plus one new
code path (`takes = "expression"`).

Names below: `myproject/` is the model's project folder, `mykind` the new kind, `mymodule` the module id. They're
placeholders.

## Checklist

| # | Step | File | Router | Math | Math Language |
|---|---|---|---|---|---|
| 1 | A loader class in the project | `myproject/<file>.py` | `route.py:Router` | `solve.py:Solver` (both math projects) | `read.py:Reader` |
| 2 | The `LOCAL_CODE` entry | `harness.py` | `classifier` | `math` | `math_language` |
| 3 | The provider entry | `harness.json` → `providers.smmol.models` | `smROUTER_01` | `smMATH01-*`, `smMATH001-a` | `smMATH_LANGUAGE_001` |
| 4 | The module config | `harness.json` → `modules` | `router`, grey | `math`, brown | `math_language`, lime |
| 5 | The contract | `contracts.json` | none | none | `math_language` |
| 6 | Wiring into the cycle | `harness.py` | `route`, `_route_hint`, `_router_overrule` | `work_math`, `_whole_math`, `math_on` | `read_math`, `solve_problems`, `_problems`, `math_language_on` |
| 7 | Settings | `harness.py` → `SETTINGS` | 3 thresholds | `check` | `only_with_numbers` |
| 8 | Checks and grading | `checks.json`, `harness.py` → `test_module`, `grade` | 5 checks | 6 checks | 6 checks |
| 9 | Tests with stubs | `tests/test_harness.py` | `StubRouter`, `Routing` | `StubMath`, `StubWholeMath`, `MathModule` | `StubReader`, `MathLanguage` |
| 10 | The page | `page.html` | color only | color, plus the Math block in a cycle card | color, plus the Math Language section |

---

## 1. The loader class

The harness imports `<file>.py` from the project root and calls `Class(checkpoint_path)`.

```python
# myproject/read.py
import torch
from model import Config, MyNet

class Reader:
    def __init__(self, path):
        ck = torch.load(path, map_location="cpu")   # every loader here stays on the CPU
        self.model = MyNet(Config(**ck["config"]))
        self.model.load_state_dict(ck["model"])
        self.model.eval()

    def read(self, text):
        ...                                          # plain Python types in, plain Python types out
```

Rules that come from `load_local()`:

- **The checkpoint must be at `myproject/out/<name>.pt`.** The harness finds the project folder as the checkpoint's
  grandparent (`dirname(dirname(path))`), and imports `<file>.py` from there.
- **Import siblings by plain name** (`from model import ...`). The project folder is put first on `sys.path` while
  loading. Your `model.py` won't clash with another project's `model.py`: same-named modules are taken out of
  `sys.modules` before the import and restored afterwards.
- **Load on the CPU** (`map_location="cpu"`, no `.to("mps")`), like `Router`, `Reader` and both `Solver`s, so the
  harness never competes with training for the MacBook Pro M5's GPU.
- **Any exception** while loading becomes `HarnessError` "couldn't load <path> (<error>)".
- **Give the project a `__main__` CLI** that loads `out/<name>.pt` and prints one result, like `route.py`, `read.py`
  and `solve.py`. It's the quickest manual check.
- **Test the loader** with a tiny random model saved to a temp folder, with its generation method replaced by a lambda,
  as `smMATH_LANGUAGE_001/tests/test_reader.py` and `smMATH001-a/tests/test_work.py` do.

## 2. The `LOCAL_CODE` entry

```python
LOCAL_CODE = {"classifier": "route.py:Router", "math": "solve.py:Solver",
              "math_language": "read.py:Reader", "mykind": "read.py:Reader"}
```

What this entry controls:

- a module with `"kind": "mykind"` only offers `local` providers, and only models whose entry kind is `mykind`;
- `test_module()` refuses to test it until its checkpoint exists.

A provider entry can override the loader with its own `"code": "file.py:Class"`.

## 3. The provider entry

```json
"smmol": {"kind": "local", "label": "Mac · trained in SMMOL", "models": {
  "myproject": {"path": "../myproject/out/mine.pt", "kind": "mykind"}
}}
```

- **`path`** is relative to `paratroop_harness_02/`.
- **Always give `kind`.** A plain string entry, or a missing `kind`, means `classifier`, which is the old router form.
- **Before the file exists,** the dropdown shows the model as "(not trained yet)" and it can't be chosen.
- **Once any checkpoint is written,** even mid-training, it's offered and loaded, and it's reloaded whenever the file
  changes ([architecture.md → The cache](architecture.md#the-cache)).

## 4. The module config

```json
{
  "id": "mymodule", "name": "My Module", "color": "lime", "kind": "mykind",
  "provider": "smmol", "model": "myproject",
  "does": "Small trained model: one short line about its job"
}
```

- **Position in `modules`** is the order of `Harness.order`: the card's place on the page's circle and in
  `/api/state`. It doesn't set when the module runs; `think()` does.
- **`does`** is the card's subtitle.
- **`color`** has to be a color `page.html` defines ([step 10](#10-the-page)).
- **Extra keys** are allowed. Math has `tool_does` and `tool_how`, which describe arithmetic to the Planner when the
  calculator is off.

## 5. The contract

If the model returns structured data that other modules will read, add a schema to `contracts.json` and check it in
code. Nothing is sent to a server for local models, so `validate()` and `normalize()` are the only enforcement:

```json
"mymodule": {
  "type": "object", "required": ["items"],
  "properties": {"items": {"type": "array", "maxItems": 5, "items": {"type": "object",
    "required": ["id", "text"],
    "properties": {"id": {"type": "string", "maxLength": 4}, "text": {"type": "string", "maxLength": 120}}}}}
}
```

`validate()` supports `type`, `required`, `properties`, `items`, `enum` and `minItems`. `normalize()` applies `maxLength`,
`maxItems`, `minimum` and `maximum`, and drops unknown keys. Keep limits short: see
[architecture.md → Contracts](architecture.md#3-contracts) for why.

## 6. Wiring into the cycle

Follow `read_math()`, `_problems()` and `math_language_on()`:

```python
def mymodule_on(self):
    return "mymodule" in self.modules and self.module_on("mymodule") and self.local_ready("mymodule")

def run_mymodule(self, live):
    s = live.state
    choice = self.choice("mymodule")
    entry = {"module": "mymodule", "name": self.modules["mymodule"]["name"], "part": "", "model": choice,
             "status": "running", "started": time.time(), "seconds": None, "retries": 0, "error": ""}
    live.apply(lambda st: st["trace"].append(entry))       # the page shows the card as busy
    started = time.monotonic()
    items, error = [], ""
    try:
        value = {"items": self.local(choice).read(s["message"])}
        wrong = validate(value, self.contracts["mymodule"])
        if wrong:
            raise HarnessError("broke its contract: %s" % "; ".join(wrong[:3]))
        items = normalize(value, self.contracts["mymodule"])["items"]
    except HarnessError as exc:
        error = str(exc)
    except Exception as exc:                                # a broken checkpoint shouldn't take the turn down
        error = "mymodule failed: %s" % exc
    seconds = round(time.monotonic() - started, 3)

    def close(st):
        entry.update(status="error" if error else "done", seconds=seconds, error=error)
        st["my_items"] = items
    live.apply(close)                                       # always change state through live.apply
    if self.on_event:
        self.on_event(entry)                                # the CLI prints the call

def _my_items(self, s):
    """What other modules see, or None when the module is off, so drop_none leaves it out."""
    return s.get("my_items") if self.module_on("mymodule") and s.get("my_items") else None
```

Then:

- **Add the new key to `new_state()`** (`"my_items": []`) so every run has it.
- **Call it from `think()`** at the right point, guarded by `mymodule_on()`.
  - Router runs first.
  - Math Language runs after Recall and before the planning loop.
  - Decide whether a failure should end the turn, as the router's does by raising, or be noted and skipped, as Math
    Language's is.
- **Hand the result to the modules that need it.** Add `"my_items": self._my_items(s)` to their payloads: `plan()` and
  `speak()` for Math Language's `problems`. Add a line to the prompts in `harness.json` if they need to know what it
  means.
- **If it should answer alone** with Language off, extend `_plain_reply()` as Math Language did.
- **If it changes decisions,** do it in `decide()` like the arithmetic rule and `_router_overrule`, and set `by` so the
  page shows who decided.
- **Pipes are optional.** Local modules aren't in `STAGES`; add yours there and to `run_stage()` if you want `./m
  mymodule`.

**Invisible when off.** A switched-off module must never run, and no other module's payload, prompt or contract may
mention it. A `_my_items()` helper that returns `None` gets that for free through `drop_none`.

## 7. Settings

Add a `SETTINGS` entry for the kind. Each tuple is `(key, label, type, low, high, step, default)`, and type is
`"number"`, `"int"` or `"bool"`:

```python
"mykind": [("only_with_numbers", "Only read messages that have numbers in them", "bool", None, None, None, True)],
```

Read it with `self.setting("mymodule", "only_with_numbers")`. The ⚙ dialog, validation, `/api/settings` and saving to
`settings.json` all come for free.

## 8. Checks and grading

1. **Write cases** in `checks.json` under the module id, like the `math_language` cases with `message` and `expected`.
2. **Dispatch the test** in `test_module()`. Local modules aren't stages, so add a branch next to `math_language` and
   `math`:

   ```python
   elif mid == "mymodule":
       self.run_mymodule(live)
   ```

3. **Grade** in `grade(mid, case, s)`: return `(True, "ok")` or `(False, "<short reason>")`. Grade what matters, not
   the exact form:
   - Math Language grades by the *values* its expressions work out to;
   - Math fails any check where a step went to the calculator.
4. **Test button:** the page shows it once `checks.json` has cases. `--test mymodule` works too. A switched-off or
   untrained module refuses to be tested, so no bogus score is saved.

## 9. Tests with stubs

In `tests/test_harness.py`:

1. **A stub** that plays the model with no PyTorch:

   ```python
   class StubMine:
       def __init__(self, items):
           self.items, self.calls = items, []

       def read(self, text):
           self.calls.append(text)
           return [dict(i) for i in self.items]
   ```

2. **An untrained entry** in `Rig.setUp()`'s `smmol` models (`{"path": os.path.join(self.dir, "mine-not-trained.pt"),
   "kind": "mykind"}`), so the default choice resolves to "not trained yet".
3. **A helper** that uses `Rig.trained()`. It writes an empty checkpoint, registers it in the config, and puts the stub
   in `_local_models` under the file's `(path, mtime)`:

   ```python
   def use_mine(self, items):
       return self.trained("myproject", StubMine(items), "mykind")
   ```

   If your model isn't the module's default choice, call `self.h.choose("mymodule", "smmol/<name>")` after it, as
   `use_whole_math()` does.
4. **A test class** covering at least:
   - untrained: skipped, offered as "not trained yet", can't be chosen or tested;
   - offered only in its own module's dropdown;
   - it runs, and its output reaches the modules that use it (check `self.calls("planner")[-1]`);
   - switched off: not in `trace`, and not in any payload;
   - a broken or failing model doesn't take the turn down, if that's the design;
   - settings change its behavior;
   - `test_module()` grades it.

   Use `self.only(...)` to switch everything else off.
5. **Update the counts** the tests assert. `Page.test_thinks_live_and_forgets_on_request` expects
   `len(state["modules"]) == 11`.

Run `python3 -m unittest discover -s tests`. The tests need no GPU and no network.

## 10. The page

- **Color.** `page.html` defines `--<color>` (card fill) and `--<color>-ink` (accents) for grey, red, orange, yellow,
  green, teal, blue, purple, pink, brown and lime, in both the light block and the dark block. Use one of those, or add
  both variables to **both** blocks. An undefined color leaves the card without a fill.
- **Card, dropdown, switch, ⚙ and Test** need no changes. They're built from `/api/state`.
- **"How it got there":** if the module writes new state, add a section to `renderBoard()` using
  `section(title, "mymodule", subtitle)` so it takes the module's color, as the Math Language section does. The Router
  never got one.

## 11. After it works

- **A live check** in a scratch data folder ([running.md](running.md#a-scratch-data-folder)), with only the modules you
  need switched on.
- **Update the docs:**
  - the harness README (the module section, settings table, test count);
  - [architecture.md](architecture.md) (module table, contracts, switches table, settings table, turn cycle);
  - a model file in [../smModels/](../smModels/);
  - [../research/results.md](../research/results.md).
