# Positive memories with cosine rank above 50

4,944 of 28,546 positive memory occurrences (17.32%), across 4,153 of 18,532 training questions (22.41%).

All results count question-memory occurrences, not unique memories. Above-50 rates use all positive occurrences in the indicated group as the denominator. These are associations, not causal effects.

Stored role-tagged memory text, Qwen tokenizer with add_special_tokens=False; no wrappers or question/options.

## By Memory Role

| Group | All positives | Within 50 | Above 50 | Above-50 rate | Share of tail |
| --- | ---: | ---: | ---: | ---: | ---: |
| assistant | 2 | 2 | 0 | 0.00% | 0.00% |
| user | 3 | 2 | 1 | 33.33% | 0.02% |
| user_assistant | 28,541 | 23,598 | 4,943 | 17.32% | 99.98% |

## By Reference Roles

| Group | All positives | Within 50 | Above 50 | Above-50 rate | Share of tail |
| --- | ---: | ---: | ---: | ---: | ---: |
| assistant | 14 | 5 | 9 | 64.29% | 0.18% |
| user | 9 | 5 | 4 | 44.44% | 0.08% |
| user+assistant | 28,523 | 23,592 | 4,931 | 17.29% | 99.74% |

## By Length

| Group | All positives | Within 50 | Above 50 | Above-50 rate | Share of tail |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1-128 | 9,252 | 7,965 | 1,287 | 13.91% | 26.03% |
| 129-256 | 7,216 | 5,915 | 1,301 | 18.03% | 26.31% |
| 257-512 | 10,255 | 8,294 | 1,961 | 19.12% | 39.66% |
| 513-1024 | 1,822 | 1,427 | 395 | 21.68% | 7.99% |
| 1025+ | 1 | 1 | 0 | 0.00% | 0.00% |

## By Question Type

| Group | All positives | Within 50 | Above 50 | Above-50 rate | Share of tail |
| --- | ---: | ---: | ---: | ---: | ---: |
| anti_stereotypical_pref | 5,148 | 4,122 | 1,026 | 19.93% | 20.75% |
| ask_to_forget | 7,964 | 7,946 | 18 | 0.23% | 0.36% |
| health_and_medical_conditions | 3,443 | 2,573 | 870 | 25.27% | 17.60% |
| neutral_preferences | 5,063 | 3,648 | 1,415 | 27.95% | 28.62% |
| stereotypical_pref | 3,073 | 2,389 | 684 | 22.26% | 13.83% |
| therapy_background | 3,855 | 2,924 | 931 | 24.15% | 18.83% |

## By Length Excluding Forget

| Group | All positives | Within 50 | Above 50 | Above-50 rate | Share of tail |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1-128 | 3,116 | 1,841 | 1,275 | 40.92% | 25.88% |
| 129-256 | 6,860 | 5,562 | 1,298 | 18.92% | 26.35% |
| 257-512 | 8,943 | 6,984 | 1,959 | 21.91% | 39.77% |
| 513-1024 | 1,663 | 1,269 | 394 | 23.69% | 8.00% |
| 1025+ | 0 | 0 | 0 | N/A | 0.00% |

## Length summary

| Cohort | Median Qwen tokens | 90th percentile | Median characters |
| --- | ---: | ---: | ---: |
| all | 220 | 467 | 1050 |
| within_50 | 216 | 461 | 1023 |
| above_50 | 246 | 491.7 | 1204 |
| excluding_forget_within_50 | 267 | 490 | 1290.5 |
| excluding_forget_above_50 | 247 | 492 | 1205.5 |

The pooled length pattern is affected by question-type composition. Compare the question-type/length cross-table and the separate length table excluding ask_to_forget. These remain descriptive associations, not causal estimates.

The standalone-role groups and single-reference-role groups have very few examples; their rates should not be interpreted as reliable general role effects.
