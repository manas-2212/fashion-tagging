| split | seed | model_a | model_b | n | both_correct | only_a_correct | only_b_correct | both_wrong | accuracy_a | accuracy_b | oracle_accuracy | share_of_a_errors_fixed_by_b |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 42 | final_dl | h0_hog_svm | 10,000 | 8,867 | 623 | 181 | 329 | 0.9490 | 0.9048 | 0.9671 | 0.3549 |
| test | 42 | final_dl | h2_fusion_svm | 10,000 | 9,465 | 25 | 36 | 474 | 0.9490 | 0.9501 | 0.9526 | 0.0706 |
| test | 42 | h1_emb_svm | h2_fusion_svm | 10,000 | 9,468 | 31 | 33 | 468 | 0.9499 | 0.9501 | 0.9532 | 0.0659 |
