| model | label | n_seeds | accuracy_mean | accuracy_std | accuracy_min | accuracy_max | macro_f1_mean | macro_f1_std | primary_seed_accuracy | ci95_low | ci95_high | shirt_recall_mean | in_group_confusion_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h0_hog_svm | H0: PCA-HOG + SVM | 1 | 0.9048 | 0.0000 | 0.9048 | 0.9048 | 0.9047 | 0.0000 | 0.9048 | 0.8990 | 0.9102 | 0.7080 | 0.1532 |
| p1_cnn | Phase 1 CNN (re-trained) | 3 | 0.9252 | 0.0034 | 0.9215 | 0.9281 | 0.9250 | 0.0030 | 0.9281 | 0.9231 | 0.9332 | 0.7570 | 0.1225 |
| final_dl | Final deep model | 3 | 0.9488 | 0.0015 | 0.9472 | 0.9501 | 0.9487 | 0.0014 | 0.9490 | 0.9445 | 0.9532 | 0.8343 | 0.0847 |
| h1_emb_svm | H1: CNN embedding + SVM | 3 | 0.9495 | 0.0009 | 0.9485 | 0.9502 | 0.9494 | 0.0009 | 0.9499 | 0.9454 | 0.9540 | 0.8327 | 0.0838 |
| h2_fusion_svm | H2: embedding + HOG fusion SVM | 3 | 0.9502 | 0.0020 | 0.9483 | 0.9523 | 0.9501 | 0.0020 | 0.9501 | 0.9455 | 0.9539 | 0.8387 | 0.0832 |
| h3_cascade | H3: confidence-gated cascade | 3 | 0.9502 | 0.0021 | 0.9482 | 0.9523 | 0.9501 | 0.0021 | 0.9501 | 0.9455 | 0.9539 | 0.8387 | 0.0832 |
