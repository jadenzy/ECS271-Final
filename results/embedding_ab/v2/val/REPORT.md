# Persona-disjoint embedding/ranker comparison

Fine-tune 9,000 questions from A; both 57-feature rankers train on B; evaluate on 909 validation questions. No test run.
Both rankers include learned question/message similarity and generic-message correction, with five persona-held-out folds inside B. Redundancy-aware selection is deferred.

| Method | Hit@10 | NDCG@10 | Recall@15 | Every visible reference@15 | Non-forget type-macro Hit@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| original_cosine | 45.21% | 25.35% | 37.19% | 22.33% | 30.65% |
| original_ranker | 79.98% | 51.82% | 69.03% | 52.70% | 76.10% |
| finetuned_cosine | 92.74% | 65.73% | 80.87% | 61.83% | 91.45% |
| finetuned_ranker | 93.73% | 77.70% | 93.28% | 87.90% | 92.29% |
