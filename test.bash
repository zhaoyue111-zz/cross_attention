CUDA_VISIBLE_DEVICES=3 python test.py    \
                            --data_dir  datasets/Fundus512/    \
                            --test_bs   1   \
                            --lr        1e-3    \
                            --size      swinv2_small_patch4_window16_256    \
                            --save_result  Fundus256/result_1e-3_9e-1_small16/01/output    \
                            --checkpoint_dir   Fundus256/result_1e-3_9e-1_small16/01/checkpoints    \
                            --num_classes  2    \
                            --train_domains 02,03,04,05    \
                            --test_domains 01    \


#CUDA_VISIBLE_DEVICES=4 python test.py    \
#                            --data_dir  datasets/Fundus512/    \
#                            --test_bs   1   \
#                            --lr        1e-3    \
#                            --size      swinv2_small_patch4_window16_256    \
#                            --save_result  Fundus256/result_1e-3_9e-1_small16/02/output    \
#                            --checkpoint_dir   Fundus256/result_1e-3_9e-1_small16/02/checkpoints    \
#                            --num_classes  2    \
#                            --train_domains 01,03,04,05    \
#                            --test_domains 02    \
#
#
#CUDA_VISIBLE_DEVICES=4 python test.py    \
#                            --data_dir  datasets/Fundus512/    \
#                            --test_bs   1   \
#                            --lr        1e-3    \
#                            --size      swinv2_small_patch4_window16_256    \
#                            --save_result  Fundus256/result_1e-3_9e-1_small16/03/output    \
#                            --checkpoint_dir   Fundus256/result_1e-3_9e-1_small16/03/checkpoints    \
#                            --num_classes  2    \
#                            --train_domains 01,02,04,05    \
#                            --test_domains 03    \
#
#
#CUDA_VISIBLE_DEVICES=4 python test.py    \
#                            --data_dir  datasets/Fundus512/    \
#                            --test_bs   1   \
#                            --lr        1e-3    \
#                            --size      swinv2_small_patch4_window16_256    \
#                            --save_result  Fundus256/result_1e-3_9e-1_small16/04/output    \
#                            --checkpoint_dir   Fundus256/result_1e-3_9e-1_small16/04/checkpoints    \
#                            --num_classes  2    \
#                            --train_domains 01,02,03,05    \
#                            --test_domains 04    \
#
#
#CUDA_VISIBLE_DEVICES=4 python test.py    \
#                            --data_dir  datasets/Fundus512/    \
#                            --test_bs   1   \
#                            --lr        1e-3    \
#                            --size      swinv2_small_patch4_window16_256    \
#                            --save_result  Fundus256/result_1e-3_9e-1_small16/05/output    \
#                            --checkpoint_dir   Fundus256/result_1e-3_9e-1_small16/05/checkpoints    \
#                            --num_classes  2    \
#                            --train_domains 01,02,03,04    \
#                            --test_domains 05    \