| model_a | model_b | accuracy_a | accuracy_b | a_right_b_wrong | a_wrong_b_right | p_value | p_bonferroni | method | significant_0.05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final_dl | p1_cnn (re-trained) | 0.949 | 0.9281 | 367 | 158 | 1.107e-19 | 7.751e-19 | chi-square with continuity correction | yes |
| h1_emb_svm | final_dl | 0.9499 | 0.949 | 27 | 18 | 0.233 | 1 | chi-square with continuity correction | no |
| h2_fusion_svm | final_dl | 0.9501 | 0.949 | 36 | 25 | 0.2004 | 1 | chi-square with continuity correction | no |
| h3_cascade | final_dl | 0.9501 | 0.949 | 36 | 25 | 0.2004 | 1 | chi-square with continuity correction | no |
| h2_fusion_svm | h1_emb_svm | 0.9501 | 0.9499 | 33 | 31 | 0.9005 | 1 | chi-square with continuity correction | no |
| final_dl | Phase 1 CNN (original checkpoint) | 0.949 | 0.9274 | 377 | 161 | 1.874e-20 | 1.312e-19 | chi-square with continuity correction | yes |
| final_dl | Phase 1 RBF-SVM on pixels | 0.949 | 0.8953 | 718 | 181 | 1.794e-71 | 1.256e-70 | chi-square with continuity correction | yes |
