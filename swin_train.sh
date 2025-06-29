# Crescent
# CUDA_VISIBLE_DEVICES=1 python train_gpu.py    \
#     --data_dir  ./datasets/data/Crescent/    \
#     --train_bs  8    \
#     --valid_bs  8    \
#     --img_size  512    \
#     --lr        1e-3    \
#     --n_iter    300    \
#     --size      swinv2_small_patch4_window16_256    \
#     --log_dir   ./Crescent/result_1e-3_9e-1_small16/logs    \
#     --result_dir  ./Crescent/result_1e-3_9e-1_small16   \
#     --checkpoint_dir ./Crescent/result_1e-3_9e-1_small16/checkpoints    \
#     --train_domains 01,02,03    \
#     --test_domains 04    \
#     --alpha     9e-1  \
#     --image_subdir  images \
#     --mask_subdir  masks_oc



# # # OCD
# # OC
#CUDA_VISIBLE_DEVICES=0 python train_gpu.py    \
#    --data_dir  ./datasets/data/OCD    \
#    --train_bs  8    \
#    --valid_bs  8    \
#    --img_size  512    \
#    --lr        1e-3    \
#    --n_iter    300    \
#    --size      swinv2_small_patch4_window16_256    \
#    --log_dir   ./OCD/result_1e-3_9e-1_small16/OC/01/logs    \
#    --result_dir  ./OCD/result_1e-3_9e-1_small16/OC/01   \
#    --checkpoint_dir ./OCD/result_1e-3_9e-1_small16/OC/01/checkpoints    \
#    --train_domains 02,03,04    \
#    --test_domains 01    \
#    --alpha     9e-1  \
#    --image_subdir  images \
#    --mask_subdir  masks_oc
#
#CUDA_VISIBLE_DEVICES=0 python train_gpu.py    \
#    --data_dir  ./datasets/data/OCD    \
#    --train_bs  8    \
#    --valid_bs  8    \
#    --img_size  512    \
#    --lr        1e-3    \
#    --n_iter    300    \
#    --size      swinv2_small_patch4_window16_256    \
#    --log_dir   ./OCD/result_1e-3_9e-1_small16/OC/02/logs    \
#    --result_dir  ./OCD/result_1e-3_9e-1_small16/OC/02   \
#    --checkpoint_dir ./OCD/result_1e-3_9e-1_small16/OC/02/checkpoints    \
#    --train_domains 01,03,04    \
#    --test_domains 02    \
#    --alpha     9e-1    \
#    --image_subdir  images \
#    --mask_subdir  masks_oc
#
#CUDA_VISIBLE_DEVICES=2 python train_gpu.py    \
#    --data_dir  ./datasets/data/OCD    \
#    --train_bs  8    \
#    --valid_bs  8    \
#    --img_size  512    \
#    --lr        1e-3    \
#    --n_iter    300    \
#    --size      swinv2_small_patch4_window16_256    \
#    --log_dir   ./OCD/result_1e-3_9e-1_small16/OC/03/logs    \
#    --result_dir  ./OCD/result_1e-3_9e-1_small16/OC/03   \
#    --checkpoint_dir ./OCD/result_1e-3_9e-1_small16/OC/03/checkpoints    \
#    --train_domains 01,02,04    \
#    --test_domains 03    \
#    --alpha     9e-1    \
#    --image_subdir  images \
#    --mask_subdir  masks_oc
#
#CUDA_VISIBLE_DEVICES=3 python train_gpu.py    \
#    --data_dir  ./datasets/data/OCD    \
#    --train_bs  8    \
#    --valid_bs  8    \
#    --img_size  512    \
#    --lr        1e-3    \
#    --n_iter    300    \
#    --size      swinv2_small_patch4_window16_256    \
#    --log_dir   ./OCD/result_1e-3_9e-1_small16/OC/04/logs    \
#    --result_dir  ./OCD/result_1e-3_9e-1_small16/OC/04   \
#    --checkpoint_dir ./OCD/result_1e-3_9e-1_small16/OC/04/checkpoints    \
#    --train_domains 01,02,03    \
#    --test_domains 04    \
#    --alpha     9e-1    \
#    --image_subdir  images \
#    --mask_subdir  masks_oc
#
## # OD
#CUDA_VISIBLE_DEVICES=0 python train_gpu.py    \
#    --data_dir  ./datasets/data/OCD    \
#    --train_bs  8    \
#    --valid_bs  8    \
#    --img_size  512    \
#    --lr        1e-3    \
#    --n_iter    300    \
#    --size      swinv2_small_patch4_window16_256    \
#    --log_dir   ./OCD/result_1e-3_9e-1_small16/OD/01/logs    \
#    --result_dir  ./OCD/result_1e-3_9e-1_small16/OD/01   \
#    --checkpoint_dir ./OCD/result_1e-3_9e-1_small16/OD/01/checkpoints    \
#    --train_domains 02,03,04    \
#    --test_domains 01    \
#    --alpha     9e-1  \
#    --image_subdir  images \
#    --mask_subdir  masks_od
#
#CUDA_VISIBLE_DEVICES=0 python train_gpu.py    \
#    --data_dir  ./datasets/data/OCD    \
#    --train_bs  8    \
#    --valid_bs  8    \
#    --img_size  512    \
#    --lr        1e-3    \
#    --n_iter    300    \
#    --size      swinv2_small_patch4_window16_256    \
#    --log_dir   ./OCD/result_1e-3_9e-1_small16/OD/02/logs    \
#    --result_dir  ./OCD/result_1e-3_9e-1_small16/OD/02   \
#    --checkpoint_dir ./OCD/result_1e-3_9e-1_small16/OD/02/checkpoints    \
#    --train_domains 01,03,04    \
#    --test_domains 02    \
#    --alpha     9e-1    \
#    --image_subdir  images \
#    --mask_subdir  masks_od
#
#CUDA_VISIBLE_DEVICES=0 python train_gpu.py    \
#    --data_dir  ./datasets/data/OCD    \
#    --train_bs  8    \
#    --valid_bs  8    \
#    --img_size  512    \
#    --lr        1e-3    \
#    --n_iter    300    \
#    --size      swinv2_small_patch4_window16_256    \
#    --log_dir   ./OCD/result_1e-3_9e-1_small16/OD/03/logs    \
#    --result_dir  ./OCD/result_1e-3_9e-1_small16/OD/03   \
#    --checkpoint_dir ./OCD/result_1e-3_9e-1_small16/OD/03/checkpoints    \
#    --train_domains 01,02,04    \
#    --test_domains 03    \
#    --alpha     9e-1    \
#    --image_subdir  images \
#    --mask_subdir  masks_od
#
#CUDA_VISIBLE_DEVICES=0 python train_gpu.py    \
#    --data_dir  ./datasets/data/OCD    \
#    --train_bs  8    \
#    --valid_bs  8    \
#    --img_size  512    \
#    --lr        1e-3    \
#    --n_iter    300    \
#    --size      swinv2_small_patch4_window16_256    \
#    --log_dir   ./OCD/result_1e-3_9e-1_small16/OD/04/logs    \
#    --result_dir  ./OCD/result_1e-3_9e-1_small16/OD/04   \
#    --checkpoint_dir ./OCD/result_1e-3_9e-1_small16/OD/04/checkpoints    \
#    --train_domains 01,02,03    \
#    --test_domains 04    \
#    --alpha     9e-1    \
#    --image_subdir  images \
#    --mask_subdir  masks_od

# crescent
# CUDA_VISIBLE_DEVICES=2 python train_gpu.py    \
#     --data_dir  ./datasets/Fundus512   \
#     --train_bs  4    \
#     --valid_bs  4    \
#     --img_size  256    \
#     --lr        1e-3    \
#     --n_iter    300    \
#     --size      swinv2_small_patch4_window16_256    \
#     --log_dir   ./Fundus256/result_1e-3_9e-1_small16_bank1_2/04/logs    \
#     --result_dir  ./Fundus256/result_1e-3_9e-1_small16_bank1_2/04   \
#     --checkpoint_dir ./Fundus256/result_1e-3_9e-1_small16_bank1_2/04/checkpoints    \
#     --train_domains 01,02,03,05    \
#     --test_domains 04    \
#     --alpha     9e-1    \
#     --image_subdir  image \
#     --mask_subdir  masks \
#     --num_classes 2

 CUDA_VISIBLE_DEVICES=2 NO_ALBUMENTATIONS_UPDATE=1 python train_gpu.py    \
     --data_dir  ./datasets/Fundus512   \
     --train_bs  4    \
     --valid_bs  4    \
     --img_size  256    \
     --lr        1e-3    \
     --n_iter    300    \
     --size      swinv2_small_patch4_window16_256    \
     --log_dir   ./Fundus256/result_1e-3_9e-1_small16_bank1_mainLoss/01/logs    \
     --result_dir  ./Fundus256/result_1e-3_9e-1_small16_bank1_mainLoss/01   \
     --checkpoint_dir ./Fundus256/result_1e-3_9e-1_small16_bank1_mainLoss/01/checkpoints    \
     --train_domains 02,03,04,05    \
     --test_domains 01    \
     --alpha     9e-1    \
     --image_subdir  image \
     --mask_subdir  masks \
     --num_classes 2


# CUDA_VISIBLE_DEVICES=2,3 python train_gpu.py    \
#     --data_dir  ./datasets/Fundus512   \
#     --train_bs  4    \
#     --valid_bs  4    \
#     --img_size  256    \
#     --lr        1e-3    \
#     --n_iter    1    \
#     --size      swinv2_small_patch4_window16_256    \
#     --log_dir   ./Fundus256/result_1e-3_9e-1_small16_bank1/02/logs    \
#     --result_dir  ./Fundus256/result_1e-3_9e-1_small16_bank1/02   \
#     --checkpoint_dir ./Fundus256/result_1e-3_9e-1_small16_bank1/02/checkpoints    \
#     --train_domains 01,03,04,05    \
#     --test_domains 02    \
#     --alpha     9e-1    \
#     --image_subdir  image \
#     --mask_subdir  masks \
#     --num_classes 2
#
# CUDA_VISIBLE_DEVICES=2 python train_gpu.py    \
#     --data_dir  ./datasets/Fundus512   \
#     --train_bs  4    \
#     --valid_bs  4    \
#     --img_size  256    \
#     --lr        1e-3    \
#     --n_iter    300    \
#     --size      swinv2_small_patch4_window16_256    \
#     --log_dir   ./Fundus256/result_1e-3_9e-1_small16_bank1/03/logs    \
#     --result_dir  ./Fundus256/result_1e-3_9e-1_small16_bank1/03   \
#     --checkpoint_dir ./Fundus256/result_1e-3_9e-1_small16_bank1/03/checkpoints    \
#     --train_domains 01,02,04,05    \
#     --test_domains 03    \
#     --alpha     9e-1    \
#     --image_subdir  image \
#     --mask_subdir  masks \
#     --num_classes 2
#
# CUDA_VISIBLE_DEVICES=2,3 python train_gpu.py    \
#     --data_dir  ./datasets/Fundus512   \
#     --train_bs  4    \
#     --valid_bs  4    \
#     --img_size  256    \
#     --lr        1e-3    \
#     --n_iter    300    \
#     --size      swinv2_small_patch4_window16_256    \
#     --log_dir   ./Fundus256/result_1e-3_9e-1_small16_bank1/05/logs    \
#     --result_dir  ./Fundus256/result_1e-3_9e-1_small16_bank1/05   \
#     --checkpoint_dir ./Fundus256/result_1e-3_9e-1_small16_bank1/05/checkpoints    \
#     --train_domains 01,02,03,04    \
#     --test_domains 05    \
#     --alpha     9e-1    \
#     --image_subdir  image \
#     --mask_subdir  masks \
#     --num_classes 2

# bank=5
# CUDA_VISIBLE_DEVICES=4 python train_gpu.py    \
#     --data_dir  ./datasets/Fundus512   \
#     --train_bs  4    \
#     --valid_bs  4    \
#     --img_size  256    \
#     --lr        1e-3    \
#     --n_iter 300    \
#     --size      swinv2_small_patch4_window16_256    \
#     --log_dir   ./Fundus256/result_1e-3_9e-1_small16_bank5/04/logs    \
#     --result_dir  ./Fundus256/result_1e-3_9e-1_small16_bank5/04   \
#     --checkpoint_dir ./Fundus256/result_1e-3_9e-1_small16_bank5_bank1/04/checkpoints    \
#     --train_domains 01,02,03,05    \
#     --test_domains 04    \
#     --alpha     9e-1    \
#     --image_subdir  image \
#     --mask_subdir  masks \
#     --num_classes 2
#
# CUDA_VISIBLE_DEVICES=4 python train_gpu.py    \
#     --data_dir  ./datasets/Fundus512   \
#     --train_bs  4    \
#     --valid_bs  4    \
#     --img_size  256    \
#     --lr        1e-3    \
#     --n_iter  300    \
#     --size      swinv2_small_patch4_window16_256    \
#     --log_dir   ./Fundus256/result_1e-3_9e-1_small16_bank5/05/logs    \
#     --result_dir  ./Fundus256/result_1e-3_9e-1_small16_bank5/05   \
#     --checkpoint_dir ./Fundus256/result_1e-3_9e-1_small16_bank5/05/checkpoints    \
#     --train_domains 01,02,03,04    \
#     --test_domains 05    \
#     --alpha     9e-1    \
#     --image_subdir  image \
#     --mask_subdir  masks \
#     --num_classes 2