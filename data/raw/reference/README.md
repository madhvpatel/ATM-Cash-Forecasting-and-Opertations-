# Retained reference sources

`rbi/` retains 29 monthly HTML responses from the Reserve Bank of India's
[regional holiday matrix](https://www.rbi.org.in/Scripts/HolidayMatrixDisplay.aspx),
August 2024 through December 2026. August provides padding before the observed
cash history for three-day holiday-window features.

`rbi/sources.json` records the request parameters, retrieval timestamp, SHA-256,
regional offices and extracted row count for each response. Each response is
checked against its requested month/year before use. `prepare_calendar_reference.py`
verifies the retained bytes against these hashes before rebuilding master tables.

The source distinguishes holidays under the Negotiable Instruments Act from
bank account-closing days. It also states that banks observe second and fourth
Saturdays as public holidays. The feature registry additionally documents Sunday
as a scheduled bank-closure rule; this is not an ATM-outage or cash-demand label.

RBI's daily description sometimes lists multiple regional festival names in a
single string. These strings are preserved verbatim as source descriptions. We
do not infer that each named festival independently applies to every office that
has a marker on that date.

These downloads do not establish original announcement dates. `known_at` is
therefore unknown, and historical comparisons are explicitly retrospective.
Strict as-of mode uses the retrieval time conservatively when publication time
is unknown and rejects a source unavailable at the prediction cutoff.

ATM-to-office assignments are curated regional proxies. They are not verified
servicing-bank calendars, state-wide closure guarantees or live operational
status. Source coverage and bank-specific applicability are reported separately.
