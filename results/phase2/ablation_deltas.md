| name | group | change | n_pairs | delta_mean | delta_std | delta_min | delta_max | consistent_sign | base_mean | variant_mean | delta_accuracy_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arch_deeper | architecture | blocks_per_stage=2 (13 conv layers instead of 7) | 3 | 0.0034 | 0.0010 | 0.0024 | 0.0045 | yes | 0.9523 | 0.9557 | 0.0034 |
| arch_narrow | architecture | widths=(16, 32, 64) (about 4x fewer weights) | 3 | -0.0110 | 0.0011 | -0.0119 | -0.0097 | yes | 0.9523 | 0.9413 | -0.0109 |
| arch_no_bn | architecture | use_bn=False | 3 | -0.0070 | 0.0050 | -0.0127 | -0.0030 | yes | 0.9523 | 0.9453 | -0.0070 |
| arch_plain | architecture | residual=False | 3 | -0.0013 | 0.0024 | -0.0026 | 0.0015 | no | 0.9523 | 0.9510 | -0.0013 |
| aug_no_erase | augmentation | aug='crop_flip' | 3 | -0.0016 | 0.0008 | -0.0024 | -0.0009 | yes | 0.9523 | 0.9507 | -0.0016 |
| aug_no_flip | augmentation | aug='crop_erase' | 3 | 0.0000 | 0.0003 | -0.0003 | 0.0003 | no | 0.9523 | 0.9523 | -0.0000 |
| aug_none | augmentation | aug='none' | 3 | -0.0072 | 0.0039 | -0.0104 | -0.0028 | yes | 0.9523 | 0.9451 | -0.0069 |
| opt_constant_lr | optimisation | schedule='constant', warmup_epochs=0 | 3 | -0.0068 | 0.0024 | -0.0094 | -0.0046 | yes | 0.9523 | 0.9455 | -0.0068 |
| opt_sgd | optimisation | SGD + Nesterov momentum 0.9, lr=0.05, wd=0.0005 | 3 | -0.0009 | 0.0015 | -0.0025 | 0.0005 | no | 0.9523 | 0.9514 | -0.0009 |
| p1_cnn_p1_recipe | roadmap | p1_cnn_p1_recipe | 3 | -0.0180 | 0.0020 | -0.0203 | -0.0166 | yes | 0.9523 | 0.9343 | -0.0181 |
| p1_cnn_p2_recipe | roadmap | p1_cnn_p2_recipe | 3 | -0.0209 | 0.0008 | -0.0218 | -0.0203 | yes | 0.9523 | 0.9314 | -0.0206 |
| reg_dropout05 | regularisation | dropout=0.5 | 3 | -0.0018 | 0.0013 | -0.0029 | -0.0003 | yes | 0.9523 | 0.9504 | -0.0019 |
| reg_dropout0 | regularisation | dropout=0.0 | 3 | 0.0002 | 0.0023 | -0.0020 | 0.0026 | no | 0.9523 | 0.9525 | 0.0002 |
| reg_no_ls | regularisation | label_smoothing=0.0 | 3 | -0.0029 | 0.0015 | -0.0046 | -0.0017 | yes | 0.9523 | 0.9494 | -0.0030 |
| reg_no_wd | regularisation | weight_decay=0.0 | 3 | -0.0010 | 0.0016 | -0.0028 | 0.0003 | no | 0.9523 | 0.9513 | -0.0011 |
