# BM25 rescue of cosine-tail positives

All inputs are the current training data. Tail means cosine rank >50. Rescue means BM25 rank <=k. Counts are positive question-memory occurrences.

| BM25 cutoff | All tail: rescued / 4,944 | Rate | Short non-forget: rescued / 1,275 | Rate |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 201 | 4.07% | 1 | 0.08% |
| 10 | 376 | 7.61% | 8 | 0.63% |
| 20 | 746 | 15.09% | 30 | 2.35% |
| 30 | 1,145 | 23.16% | 63 | 4.94% |
| 50 | 1,839 | 37.20% | 122 | 9.57% |

| Cohort | Tail positives | BM25 median rank | BM25 <=50 count | Rescue rate |
| --- | ---: | ---: | ---: | ---: |
| all_cosine_tail | 4,944 | 67 | 1,839 | 37.20% |
| nonforget_tail | 4,926 | 67 | 1,833 | 37.21% |
| short_nonforget_tail | 1,275 | 99 | 122 | 9.57% |
| neutral_preferences_tail | 1,415 | 69 | 504 | 35.62% |
| short_neutral_preferences_tail | 340 | 99 | 23 | 6.76% |

A top50+top50 union may contain up to 100 candidates; it is not a fixed-50 or token-budget-matched result.

## Inspected examples

Six curated examples from inspected buckets; illustrative, not a random prevalence sample.

The question and selected positive memory below are verbatim, with no answer choices or reference answers. Inspection notes are hypotheses about usefulness, not relabeling decisions.

### 1. Generic editing opener

Question ID: `v2:train:11817`; memory: `v2:32k:persona682:pair:71:72`; type: `neutral_preferences`; Qwen tokens: 31. BM25 / cosine / RRF ranks: 125 / 60 / 91.

**Question**

What are some fun and mentally stimulating activities I could try during a quiet weekend at home?

**Positive memory (full stored text)**

```text
User: Can you help me make this sound clearer and flow better?
Assistant: Sure, please share the text you'd like me to work on.
```

**Inspection note:** By inspection, this unit contains no explicit hobby/preference. Its companion positive mentions a seven-letter word puzzle and has cosine rank 29.

Companion reference-positive memories: `v2:32k:persona682:pair:73:74` (BM25 77, cosine 29, RRF 50).

### 2. Email-understanding opener

Question ID: `v2:train:07224`; memory: `v2:32k:persona828:pair:41:42`; type: `neutral_preferences`; Qwen tokens: 51. BM25 / cosine / RRF ranks: 64 / 71 / 77.

**Question**

What are some interesting ways to spend a week off that would combine relaxation with exploring new places?

**Positive memory (full stored text)**

```text
User: Hi, I got this email from someone named Layla Hassan and I’d like your help understanding what exactly she’s saying and what her main points are.
Assistant: Sure, please share the email and I’ll go through it for you.
```

**Inspection note:** The opener does not contain the travel detail. The companion email/scenic-trip unit is already at cosine rank 12.

Companion reference-positive memories: `v2:32k:persona828:pair:43:44` (BM25 94, cosine 12, RRF 35), `v2:32k:persona828:pair:45:46` (BM25 91, cosine 42, RRF 68).

### 3. Short dojo clue rescued by BM25

Question ID: `v2:train:06440`; memory: `v2:32k:persona634:pair:196:197`; type: `neutral_preferences`; Qwen tokens: 55. BM25 / cosine / RRF ranks: 28 / 69 / 47.

**Question**

I’m looking for fun evening activities to do in the city this weekend—do you have any suggestions that feel a bit more unique than going to a bar or the movies?

**Positive memory (full stored text)**

```text
User: Could you help me refine this short piece I’m preparing for the dojo newsletter? I want the language to be clear, polished, and engaging for both students and their parents.
Assistant: Of course, please share what you have and I will help refine it.
```

**Inspection note:** The dojo newsletter supplies a possible activity/context clue. BM25 ranks it 28 and RRF 47; the longer companion is at cosine rank 40.

Companion reference-positive memories: `v2:32k:persona634:pair:198:199` (BM25 34, cosine 40, RRF 33).

### 4. Short rewrite omits the activity detail

Question ID: `v2:train:01042`; memory: `v2:32k:persona391:pair:141:142`; type: `neutral_preferences`; Qwen tokens: 104. BM25 / cosine / RRF ranks: 96 / 79 / 95.

**Question**

What are some relaxing activities I could try after work to help me unwind and clear my mind?

**Positive memory (full stored text)**

```text
User: I like this, but could we make the opening sentence sound a bit warmer without losing professionalism?
Assistant: Sure! Here’s a warmer yet still professional tweak to the opening:

"I’m a freelance content writer and cultural critic with a deep passion for exploring the nuanced world of Marathi television dramas, uncovering intricate plotlines and rich cultural themes." 

This small change begins with a more personal "I’m" and adds warmth through words like "deep passion," while preserving the credibility of your role.
```

**Inspection note:** This rewrite names a cultural-interest/professional topic. The companion unit at cosine rank 25 contains the more direct early-evening/outdoors detail.

Companion reference-positive memories: `v2:32k:persona391:pair:137:138` (BM25 31, cosine 81, RRF 54), `v2:32k:persona391:pair:139:140` (BM25 83, cosine 25, RRF 45).

### 5. Longer meal clue rescued by BM25

Question ID: `v2:train:16029`; memory: `v2:32k:persona193:pair:186:187`; type: `neutral_preferences`; Qwen tokens: 247. BM25 / cosine / RRF ranks: 3 / 79 / 20.

**Question**

What are some hearty, crowd-pleasing meal ideas I could make at home to bring everyone together?

**Positive memory (full stored text)**

```text
User: Hey, can you help me tidy up the language in this email so it sounds more professional?

Hi Coach Daniels,

Thanks for confirming the next masterclass session on Thursday. I’m super keen to join, but I might need to head off right after training as I promised my family I’d cook dinner that night. I’ve been planning a big pasta meal and want to get it right before mum and dad get home.

Also, just wanted to check if there’s any extra gear we’ll need to bring apart from boots and shin pads? Looking forward to another great session.

Cheers,
Liam
Assistant: Sure! Here's a more professional yet friendly version of your email:

---
Subject: Thursday Masterclass Session

Hi Coach Daniels,

Thank you for confirming the next masterclass session on Thursday. I’m really looking forward to attending. I may need to leave promptly after training, as I’ve promised my family I would prepare dinner for them that evening.

Could you please let me know if we’ll need to bring any additional gear apart from boots and shin pads?

Thanks again, and I’m excited for another great session.

Kind regards,
Liam
---
```

**Inspection note:** The user email explicitly mentions cooking a big pasta meal for family. BM25 retrieves it at rank 3 although cosine ranks it 79.

Companion reference-positive memories: none.

### 6. Terrarium clue embedded in professional prose

Question ID: `v2:train:05531`; memory: `v2:32k:persona757:pair:29:30`; type: `anti_stereotypical_pref`; Qwen tokens: 248. BM25 / cosine / RRF ranks: 109 / 75 / 110.

**Question**

What are some creative ideas for setting up an engaging and visually appealing habitat at home for an unusual animal?

**Positive memory (full stored text)**

```text
User: In my years on the appellate bench, I have learned that maintaining balance in the oversight of emerging technologies is not unlike the attentive regulation of a fragile ecosystem. Just as I periodically calibrate the temperature and humidity in my study's terrarium to ensure the inhabitants remain healthy, so too must our courts adjust the parameters by which we assess novel legal questions. Precision, patience, and steady observation are indispensable; reactionary measures or neglect will destabilize the system. In both contexts, conditions evolve subtly, and only a measured, well-informed response will preserve integrity over time.
Assistant: Here is a refined version of your note:

"During my tenure on the appellate bench, I have observed that overseeing the integration of emerging technologies into legal frameworks requires the same measured care as tending to a delicate ecosystem. Just as I carefully regulate the temperature and humidity in a terrarium to maintain the well-being of its inhabitants, so too must our courts thoughtfully calibrate the standards by which we evaluate novel legal issues. Success in both realms depends upon precision, patience, and sustained observation; hasty action or inattention risks destabilization. In each, subtle shifts demand informed, deliberate responses to safeguard stability and integrity over the long term."
```

**Inspection note:** The user text explicitly mentions regulating terrarium temperature/humidity. Both BM25 and cosine place this plausible habitat evidence outside their top 50.

Companion reference-positive memories: `v2:32k:persona757:pair:27:28` (BM25 118, cosine 107, RRF 130).

Full companion memory texts are preserved in examples.json. Exact labels and datasets are unchanged.
