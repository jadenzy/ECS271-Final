# Training memory reuse audit

28,546 positive occurrences correspond to 28,532 unique stable memory IDs. Only 14 memory IDs are positive for more than one question; the maximum is 2 questions per memory.

4,944 positive occurrences with cosine rank >50 correspond to 4,939 unique IDs.

Stable IDs distinguish source occurrences. Exact role-tagged text is also compared within each persona/history; shared text across unrelated personas is not treated as deployable within-persona reuse.

| Question type | Positive occurrences | Unique IDs | Duplicate excess within type |
| --- | ---: | ---: | ---: |
| anti_stereotypical_pref | 5,148 | 5,148 | 0 |
| ask_to_forget | 7,964 | 7,964 | 0 |
| health_and_medical_conditions | 3,443 | 3,443 | 0 |
| neutral_preferences | 5,063 | 5,061 | 2 |
| stereotypical_pref | 3,073 | 3,073 | 0 |
| therapy_background | 3,855 | 3,853 | 2 |

## Candidate exposure versus positive reuse

The same full history is reused as a candidate pool for each persona's questions. There are 92,387 unique training candidate memories across 2,144,257 question-memory examples. A memory recurring as a candidate is different from being reference-positive for multiple questions.

## Benchmark versus deployment choice

- primary single query deployment: Use the current query and visible history only; cross-question aggregates disabled.
- optional benchmark batch experiment: Separate experiment only when the full unlabeled per-persona query set is declared available. Exclude the current query; use no relevance labels or answers.
- online deployment: Use previously observed queries only if genuinely logged and available. Respect temporal availability and current memory visibility; do not invent benchmark query order as chronology.
- split isolation: Build aggregates separately within allowed histories; never mix personas or train/validation/test partitions.
- positive frequency: Gold reuse frequency is an audit target, never an inference feature. A learned label-based prior requires out-of-fold training by persona.

Cross-question ranking popularity remains a hypothesis despite sparse positive reuse. It must be tested in its declared availability setting; it is not part of the primary feature set.
