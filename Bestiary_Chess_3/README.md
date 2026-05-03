# Heuristic v3 — automated coder for GPT-on-nonces responses

## Purpose

Automated 5-class coder for nonce-word elicitation responses, validated against
hand-coding on a 200-trial stratified sample from `gpt54_on_gpt_words`
(GPT-5.4 family on BC1's 35 GPT-authored words, 2100 trials total).

## Codes

In decision-tree order:

1. **SUBSTITUTE** — response routes the nonce to a different real word and
   describes that word's referent (e.g. *chamélo* → "the French word for camel" → camel description).
2. **REFUSE** — short, no description, no offer of help.
3. **DEFLECT** — non-recognition + speculation list / offer of help, no
   commitment to a referent.
4. **HYBRID** — substantive description of the nonce + explicit fictional flag.
5. **DESCRIBE** — substantive description of the nonce, no fictional flag.
   On `real_*` conditions this is the "hard confabulation" code.

## Validation

| Metric | v1 baseline | v3 final |
|---|---|---|
| Cohen's κ | 0.405 | **0.741** |
| Raw agreement | 60.5% | **82.0%** |
| HYBRID F1 | 80.2% | 88.3% |
| DESCRIBE F1 | 59.0% | 81.3% |
| DEFLECT F1 | 25.6% | 75.0% |
| SUBSTITUTE F1 | 6.5% | 72.4% |
| real_* DESCRIBE rate | 61.7% (hand: 18.3%) | **20.0% (hand: 18.3%)** |

REFUSE is omitted from the F1 table — hand-coding produced zero REFUSEs across
all 200 trials, consistent with a finding that GPT-5.4 family essentially never
produces flat refusals on this task. v3 also produces zero REFUSEs on the
sample.

κ ≈ 0.74 falls in the "substantial agreement" band (Landis & Koch 1977).

## Sample design

200 trials, drawn from 2100 with seed `20260503`:
- 20 per condition × 10 conditions
- 67/67/66 across `gpt-5.4`/`gpt-5.4-mini`/`gpt-5.4-nano`
- All 35 words appear at least once
- 14 trials on BC1-flagged real-referent words (*framelet*, *skeldar*)

## Files

- `heuristic_v3.py` — the coder. Self-contained; running it as `__main__`
  reproduces the validation table against `scored_200.csv` (not bundled here
  but reconstructable from the sample + hand-codes CSV).
- `stratified_sample_200.json` — the 200 sampled trials (input to the
  hand-coding tool).
- `validation_200_with_v3.csv` — the 200 trials with both `my_code`
  (hand) and `heuristic_v3` columns, plus an `agree_v3` flag.

## Key design choices and known limitations

1. **Quote normalization**: GPT outputs use curly quotes (U+2019, U+201C, U+201D)
   throughout. The coder normalizes to ASCII before regex matching. This was
   the single biggest fix between v1 and v3.

2. **Conditional fiction flags**: words like *imaginary*, *imagined*,
   *hypothetical*, *invented* are conditional. They count as fictional flags
   only in flag-likely contexts (e.g. "is an imaginary X", "could be imagined
   as", "imaginary creature"). Mere appearance — e.g. "feels partly invented"
   describing a property — does not count. This handles the case where these
   words appear as descriptive content rather than commitments.

3. **List-of-options exclusion**: when *fantasy*, *fictional*, *made-up*
   appear in alternation contexts (`fantasy/scientific/poetic way`,
   `fictional vs realistic`, `Example (made-up)`), they are treated as the
   model offering style alternatives rather than flagging the description.

4. **DEFLECT-vs-SUBSTITUTE disambiguation**: when non-recognition + offer
   signals are strong, SUBSTITUTE-type signals are interpreted as part of a
   speculation list, not a commitment to a substituted referent.

5. **Remaining errors (n=36 of 200)**: roughly half are genuinely borderline
   cases (e.g. "is a fictional, undefined object type — here's a description")
   where the codebook itself is ambiguous. Per the hand-coder, ~6 are
   inconsistencies in the hand-coding itself rather than heuristic errors.
   Pushing κ further would risk overfitting to noise in the gold standard.

## Headline implication

The single most important finding from validation: the v1-style "naive"
heuristic would have inflated the `real_*` DESCRIBE rate (the v5 paper's
"hard confabulation" metric) **3.4×** — 61.7% vs the hand-coded 18.3%. The
inflation came from conflating SUBSTITUTE responses ("X is the French word
for Y; Y is a hoofed mammal that...") with DESCRIBE responses (model
confabulating an original referent for the nonce). v3 closes that gap to
within 2 points.

The headline metric matters because BC2 found a ~13× gap between original
Chesterton words and analyst-Claude-generated words on real-status confab
rate. Reproducing or breaking that gap on the new GPT-on-GPT-words run
requires a coder that distinguishes substitution from confabulation cleanly.
