# Ranker55 V2 validation tuning

Selected trial: 1. Model: C:\Users\jaden\ECS271-Final\models\qwen_graded_all_current_setting\ranker55.txt

Training accepts supplied relevance grades 0/1/2 with linear gains [0,1,2]. Validation metrics use original reference membership; test was not used.
ask_to_forget training questions have weight 0.5; every other question has weight 1.
Validation: 1002 questions, 905 with complete visible references.

| Trial | Min child | Leaves/depth | L2 | Non-forget macro Hit@10 | NDCG@10 | Gain (pp) |
|---|---:|---|---:|---:|---:|---:|
| 0 | 100 | 31/5 | 5.0 | 51.48% | 27.69% | +0.00 |
| 1 | 100 | 63/6 | 10.0 | 52.80% | 28.75% | +1.32 |
