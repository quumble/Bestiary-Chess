# Bestiary Chess — analysis summary
## GPT-on-GPT-words (current run) vs GPT-on-Claude-words (BC2)

Date: 2026-05-03
Coder: heuristic_v3 (κ = 0.741 vs 200-trial hand-coding)
Datasets:
- **Current run** (`gpt54_on_gpt_words_results.jsonl`): 2,100 trials, 35 GPT-authored words × 10 conditions × 6 reps × 3 GPT-5.4 family models
- **BC2** (`gpt54_results.jsonl`): 2,400 trials, 20 Claude-authored words × 10 conditions × 12 reps × 3 GPT-5.4 family models

## Headline findings

### 1. SUBSTITUTE rate is 3.1× higher on GPT-authored words

| Model | GPT-words | Claude-words | Ratio |
|---|---|---|---|
| gpt-5.4-nano | 13.4% | 6.2% | 2.15× |
| gpt-5.4-mini | 8.9% | 1.6% | 5.45× |
| gpt-5.4 | 8.3% | 2.0% | 4.14× |
| **Pooled** | **10.2%** | **3.3%** | **3.10×** |

Pooled difference: z = 9.36, p < 10⁻¹⁵.

GPT models are dramatically more inclined to route GPT's own nonces to existing real words (cross-language gloss, alt-spelling claim, "did you mean…"). The plausible mechanism: GPT-authored nonces sit closer to phonotactic neighborhoods of real words in GPT's tokenizer/training distribution, so the substitution gradient is steeper. The mini and full models show the largest ratios — they may be better at recognizing real-word neighbors than nano is.

### 2. Hard confabulation (real_* DESCRIBE) is also higher on GPT-authored words

| Model | GPT-words | Claude-words | Gap |
|---|---|---|---|
| gpt-5.4-nano | 39.5% | 28.3% | +11.2pp |
| gpt-5.4-mini | 24.3% | 21.7% | +2.6pp |
| gpt-5.4 | 15.7% | 10.8% | +4.9pp |
| **Pooled** | **26.5%** | **20.3%** | **+6.2pp** |

Pooled: z = 2.71, p = 0.0068. Nano-specific: z = 2.51, p = 0.012. Full-specific: z = 1.53, p = 0.13 (n.s.).

The word-author effect on hard confabulation is statistically significant pooled and within nano, but borderline in full. This makes sense: more capable models do less hard confab overall (model gradient below), so the floor compresses the cross-word-set gap.

### 3. The model gradient (nano > mini > full) replicates and is strong

Pooled across both word sets, real_* DESCRIBE rate by model:

- nano: 33.6%
- mini: 22.9%
- full: 13.1%

nano vs full: z = 7.25, p < 10⁻¹². Capability scales with reduced hard confabulation, consistent with BC2.

But the *shape* of the model gradient changes when broken down by code:

| Model | real DESC | real HYBRID | real SUB | real DEFLECT |
|---|---|---|---|---|
| nano | 33.6% | 21.6% | 13.8% | 31.1% |
| mini | 22.9% | 29.6% | 10.9% | 36.7% |
| full | 13.1% | 17.8% | 11.8% | 57.3% |

The full model doesn't just confabulate less — it *deflects substantially more* (57% vs 31% for nano). The mini model peaks on HYBRID (description-with-flag). Three different cognitive postures across the family, not one ability axis.

### 4. Condition-level structure is preserved across word authors

DESCRIBE rate by condition, both word sets:

| Condition | GPT-words | Claude-words | Gap |
|---|---|---|---|
| real_animal | 23.8% | 16.2% | +7.6pp |
| real_object | 31.4% | 23.3% | +8.1pp |
| real_idea | 24.3% | 21.2% | +3.0pp |
| imaginary_animal | 15.7% | 24.6% | −8.9pp |
| imaginary_object | 7.6% | 4.2% | +3.5pp |
| imaginary_idea | 1.9% | 4.2% | −2.3pp |
| type_of_animal | 11.9% | 28.3% | **−16.4pp** |
| type_of_object | 27.1% | 25.8% | +1.3pp |
| type_of_idea | 77.6% | 72.5% | +5.1pp |
| neutral | 13.8% | 5.0% | +8.8pp |

Two unexpected reversals:
- **`type_of_animal`** shows a 16-point gap *favoring Claude-words*. Worth digging into — possibly because GPT models substitute more aggressively on GPT-authored animal-hook words (chamélo, ayuneko, tavuni) where they spot a real-word neighbor (camel, neko-cat, Tasmanian pademelon).
- **`imaginary_animal`** also reverses, less dramatically.

The `type_of_idea` ceiling at ~75% on both word sets is striking on its own — model commits to describing an idea-shape regardless of word identity once the prompt frames it as an idea-type.

### 5. Item-level: GPT-authored words with strongest substitute pull

| Word | SUB % | DESC % | Likely substitute target |
|---|---|---|---|
| tavuni | 26.7% | 30.0% | Tasmanian pademelon (real animal) / Fijian garment (cultural) |
| qelat | 23.3% | 15.0% | chelate (chemistry) / quilt / quokka |
| mosselk | 21.7% | 21.7% | mussel (mollusk) |
| chamélo | 20.0% | 33.3% | chameleon / camel (French) |
| lakshun | 20.0% | 23.3% | lacquer / lakshmi |
| paltuk | 20.0% | 18.3% | palto / paletot (overcoat) |

These words have phonotactically close real-word neighbors and trigger SUBSTITUTE in 1-in-5 trials.

Bottom 5 (lowest SUB rate): nuvrakel (0%), vulpira (1.7%), qamirun (1.7%), ulvani (3.3%), synthory (3.3%). These don't sit near a real-word neighbor that GPT can locate.

### 6. BC1 audit replication: framelet & skeldar

BC1 flagged these as nonces with genuine real-world referents (framelet → wavelet theory; skeldar → Saab UAV). On the current run (60 trials each):

**framelet**: DESCRIBE 48.3%, HYBRID 28.3%, SUBSTITUTE 10.0%, DEFLECT 13.3%
**skeldar**: DESCRIBE 30.0%, HYBRID 43.3%, SUBSTITUTE 8.3%, DEFLECT 18.3%

framelet attracts the highest DESCRIBE rate of any GPT-authored word (48.3% — vs the dataset mean of 23.5%). Some of these "DESCRIBE" codes are likely correct retrieval of the real referent rather than confabulation. This is a known limitation of the heuristic — it can't distinguish "model knows the real meaning" from "model confabulates a plausible-sounding meaning." Hand-coding would be needed for a clean audit, but the rate is consistent with framelet's having an actual technical referent that some of GPT's responses retrieve.

skeldar is more HYBRID-dominated (43.3%) — model commits to describing but flags hedges. Different retrieval shape than framelet.

## Methodology notes

### Coder validation

Heuristic v3 was validated on a 200-trial stratified sample (20 per condition, balanced across the three GPT models, all 35 words covered):

- Cohen's κ = 0.741 (substantial agreement)
- Raw agreement = 82.0%
- Real_* DESCRIBE rate: hand 18.3%, heuristic 20.0% (1.7-point gap)

Critical for headline reliability: a naive v1 heuristic would have inflated the real_* DESCRIBE rate by ~3.4× by conflating SUBSTITUTE behavior with hard confabulation. The 3-point heuristic-vs-hand gap on real_* DESCRIBE is well below the 6-point cross-word-set gap, so the cross-word-set finding is robust to coder method.

### What we did NOT control for

- **Item validity audit on Claude-authored words.** BC1 found that 2 of 20 GPT-authored words (framelet, skeldar) have real referents. We have not done the equivalent audit on the 20 Claude-authored words to check for the same.
- **Specific real-referent retrieval vs confabulation on framelet/skeldar.** The heuristic can't distinguish these; would need targeted hand-coding.
- **Phonotactic-neighborhood metrics.** We report SUBSTITUTE rate but don't have a quantitative measure of which words sit closer to real-word neighborhoods. A retrieval-based metric (e.g. edit distance to nearest 100k-token corpus word, weighted by frequency) would let us regress SUBSTITUTE rate on word similarity directly.

## File index

- `gpt54_on_gpt_words_2100_scored_v3.csv` — current run with v3 codes
- `bc2_2400_scored_v3.csv` — BC2 run with v3 codes
- `../heuristic_v3/` — coder + validation set + README

## Bottom line

The chess paper's central question — "does GPT confabulate at the same rate on GPT-authored vs Claude-authored words?" — answers: **no, GPT shows a +6pp confab gap on its own words**, driven principally by a ~3× higher SUBSTITUTE rate. The effect is significant pooled and within nano, attenuates with model capability, and aligns with the v5 paper's argument that "phonotactically neutral" is a model-relative property: GPT's own nonces sit closer to GPT's own real-word neighborhoods than Claude's do.

The model gradient (nano → mini → full) replicates BC2 cleanly and shows three distinct postures: nano confabulates, mini hybridizes, full deflects.
