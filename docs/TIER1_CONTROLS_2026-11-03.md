# Tier-1 money-path controls (3 Nov 2026)

Follow-up to `AUDIT_2026-11-03.md`. An earlier pass had already frozen approvals, made a locked
manifest reject edits, and bounded amounts. This note covers the gaps that were still open and how
they were closed.

## What was broken

| Gap | What could happen |
|---|---|
| No sign-in; any website could POST to the portal | A page open in the officer's browser could approve every ATM (an empty or garbled request body defaulted to "approve all") |
| The role came from the request (`role: "CHECKER"` was the default) | Anyone could act as a checker |
| Same person could override and then approve | No real maker/checker split |
| Chat "confirm" was a keyword search | "I **know** the numbers" counted as "now", so it approved all 8 ATMs; "acknowledged" locked the manifest |
| `/approve` with no ATM approved all; `/lock` had no confirmation | One mistyped command moved the whole fleet |
| The AI model could call approve, lock, sweep and clear-bin itself | Nothing stood between the model and the money |
| Chat `/reset` unlocked a locked manifest | The lock could be undone from chat |
| A manifest with zero approvals could be locked | The day was frozen with a ₹0 pull slip |
| The manifest showed both refill cycles as one pull | Cash for a later visit appeared on day one |
| Tests reset the real `config_overrides.json` | Running the tests deleted the operator's approvals (that is why git shows the file as deleted) |
| The signing key was written in the source | Anyone with the code could forge a manifest signature |
| Sweeps could cross cities (Nashik to Mumbai) | Physically impossible transfers were accepted |

## What changed

- **Sign-in** (`serve.py`): every request needs a sign-in. People use the `/login` page (added
  2026-11-04), which sets an 8-hour HttpOnly, SameSite=Strict session cookie. 5 wrong passwords
  for one user name lock that name for 15 minutes. Sessions live in memory, so restarting the
  server signs everyone out. Scripts can still use HTTP Basic. Users live in
  `live_portal_v2/users.json`, which stores salted PBKDF2 password hashes.
  Never commit this file. Each user's name and role come from the sign-in. Any `role`,
  `officer_name` or `actor` in the request body is overwritten.
- **Cross-site requests blocked**: a POST must be JSON (`Content-Type: application/json`), come from
  the portal's own address, and have a valid JSON object as its body. CORS headers were removed.
- **Roles**: only CHECKER users can approve, lock or reset.
- **Separation of duties** (`masters_manager.py`): whoever overrode or amended an ATM's indent cannot
  approve it, and a different checker must do so. Refusals return HTTP **409**.
- **Chat** (`agent.py`):
  - The AI model only gets read-only tools.
  - Free text never executes anything. It replies with the exact command to send.
  - Action commands must end in the word `confirm`, for example `/approve 3 confirm`,
    `/sweep 4 3 700000 confirm` or `/lock confirm`. They run as the signed-in user.
  - `/approve` needs an ATM id or `all`. `/sweep` needs an amount in rupees.
- **Lock**: reset is refused while the manifest is locked. Locking needs at least one approved CIT
  indent and the `CMS_SIGNING_KEY` environment variable.
- **Manifest**: there is a new `vault_pull_by_date` field, so each approved cycle is pulled from the
  chest on its own date.
- **Sweeps**: the donor and recipient must be in the same city.
- **UI**: changing the slider on an ATM now only records an override. A different checker approves it
  afterwards. Save and reset now check the server's reply instead of always showing success, and
  reset asks for confirmation first.
- **Tests**: `tests/__init__.py` points the portal at a temp folder, so tests never touch live files.

## Found while testing in the browser (also fixed)

| Problem | Effect | Fix |
|---|---|---|
| The server shared its whole folder as static files | Any signed-in user could download `users.json` (password hashes) or `config_overrides.json`; HEAD requests needed no sign-in | Only `index.html` and the mockups page are served. Everything else returns 404, and HEAD needs sign-in |
| The "Authorize Indent" button called a function the earlier edit had deleted (along with two other button functions) | The button did nothing | Restored `approveCurrentAtmIndent`, `switchOperationsView` and `toggleMapTheme` |
| The hidden slider capped values at one ATM's cassette capacity (₹28.6 L) | Every 2-cycle approval was treated as an override and cut to ₹28.6 L | The slider's maximum now covers the full plan |
| An override changed the indent figure but not the refill cycles | Approving ATM 2 at ₹15 L would have frozen the maker's ₹44.8 L cycles; the simulation also loaded ₹44.8 L | Cycles are scaled to the override total, so cycles, simulation, approval and manifest all agree |
| The approve button compared against the maker's figure, not the current one | A checker approving someone else's override would have recorded a new override instead | It compares against the current proposal |
| A POST refused with 401/403 left its body unread | The connection could reset before the error reached the client (seen as a flaky test) | The body is read first, and capped at 1 MB (413 above that) |
| The manifest modal read fields the server never sends | It showed "TOTAL DISPATCH ₹0" and an empty table right before the lock | It now shows the real total, each ATM's status and the pull-by-date schedule |

The browser walkthrough, run against throwaway users and state, confirmed:
- maker approval is refused with 403
- "approve all now, I know the numbers" approves nothing, and the AI replies with the slash command
- a maker override followed by a different checker's approval works
- a checker cannot approve their own override
- `/lock` without `confirm` changes nothing
- after the lock, edits, reset and chat `/reset` are all refused

Note: some embedded browsers (including the app's preview pane) don't show the Basic-auth login box.
Chrome, Safari and Firefox do.

## How to run it now

```bash
cd live_portal_v2
python serve.py add-user alice MAKER
python serve.py add-user bob CHECKER
export CMS_SIGNING_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
python serve.py
```

Keep `CMS_SIGNING_KEY` the same across restarts, or earlier manifest signatures can't be checked.

## How to verify

```bash
python -m unittest discover -s tests -t .
```

`tests/test_tier1_controls.py` starts the real server on a spare port. It checks:
- requests without sign-in get 401
- cross-site and non-JSON POSTs get 403
- a MAKER cannot approve, even when claiming CHECKER in the request body
- the same person cannot override and approve
- "I know the numbers" approves nothing
- a locked manifest refuses changes and reset with 409

## Still open (not tier 1)

- Amounts are checked against the ATM's total cassette capacity, not its free space.
- There is no supported way to unlock a locked manifest. Starting a new cycle means removing the lock
  from `config_overrides.json` by hand.
- The sweep, clear-bin and emergency buttons were removed from the UI in the earlier pass. Their
  server endpoints still exist and are now gated by sign-in.
- The free-text sweep parser picks up the ATM number as the amount. It refuses safely, but
  `/sweep` is the only working path.
- If you want the deleted `config_overrides.json` back, restore it from git:
  `git checkout -- live_portal_v2/config_overrides.json`.
