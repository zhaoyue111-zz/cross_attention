# small16
# CUDA_VISIBLE_DEVICES=2 python test.py    \
#     --data_dir  datasets/data/OD/    \
#     --test_bs   1   \
#     --size     swinv2_small_patch4_window16_256    \
#     --checkpoint_dir   ./results_1e-3_9e-1_small16/01/checkpoints    \
#     --save_result  ./results_1e-3_9e-1_small16/01/output    \
#     --train_domains 02,03,04    \
#     --test_domains 01
#
# CUDA_VISIBLE_DEVICES=2 python test.py    \
#     --data_dir  datasets/data/OD/    \
#     --test_bs   1   \
#     --size     swinv2_small_patch4_window16_256    \
#     --checkpoint_dir   ./results_1e-3_9e-1_small16/02/checkpoints    \
#     --save_result  ./results_1e-3_9e-1_small16/02/output    \
#     --train_domains 01,03,04    \
#     --test_domains 02