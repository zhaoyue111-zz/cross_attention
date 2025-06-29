import argparse

# 可以用来复制文件名
# MODEL_URLS = {
#     'swinv2_tiny_patch4_window16_256': 'configs/weights/swinv2_tiny_patch4_window16_256.pth',
#     'swinv2_small_patch4_window8_256': 'configs/weights/swinv2_small_patch4_window8_256.pth',
#     'swinv2_small_patch4_window16_256': 'configs/weights/swinv2_small_patch4_window16_256.pth',
#     'swinv2_base_patch4_window8_256': 'configs/weights/swinv2_base_patch4_window8_256.pth',
#     'swinv2_base_patch4_window16_256': 'configs/weights/swinv2_base_patch4_window16_256.pth',
#     'swinv2_base_patch4_window12_192_22k': 'configs/weights/swinv2_base_patch4_window12_192_22k.pth',
#     'swinv2_large_patch4_window12_192_22k': 'configs/weights/swinv2_large_patch4_window12_192_22k.pth',
# }

def get_args():
    parser = argparse.ArgumentParser(description='SwinTV2UNet Training')
    # 数据和模型参数
    parser.add_argument('--data_dir', type=str, default='datasets/Fundus512', help='data directory')
    # parser.add_argument('--train_dir', type=str, default='datasets/data/OD/', help='data directory')
    # parser.add_argument('--test_dir', type=str, default='datasets/data/OD/', help='data directory')

    parser.add_argument('--train_bs', type=int, default=8, help='train batch size')
    parser.add_argument('--valid_bs', type=int, default=8, help='validation batch size')
    parser.add_argument('--test_bs', type=int, default=1, help='test batch size')
    # 训练参数
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loading')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')  # TODO 调整学习率
    parser.add_argument('--n_iter', type=int, default=300, help='number of iterations')
    # 模型和设备设置
    parser.add_argument('--size', type=str, default='swinv2_small_patch4_window16_256', help='model size')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
    # 模式切换
    parser.add_argument('--debug', action='store_true', help='debug mode')
    # 路径参数
    parser.add_argument('--model_path', type=str, default='results_1e-3_9e-1_small16/checkpoints/model_best.pth', help='best model directory')
    parser.add_argument('--model_dir', type=str, default='./results_1e-3_9e-1_small16/models', help='model directory')
    parser.add_argument('--log_dir', type=str, default='./results_1e-3_9e-1_small16/logs', help='log directory')
    parser.add_argument('--result_dir', type=str, default='./results_1e-3_9e-1_small16', help='result directory')
    # 其他参数
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha for loss function')
    parser.add_argument('--resume', type=str, default='', 
                       help='path to checkpoint for resuming training (default: none)')
    parser.add_argument('--start_epoch', type=int, default=1, 
                       help='manual epoch number (useful on restarts)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/',
                       help='directory to save checkpoints')
    # Checkpoint管理
    parser.add_argument('--save_freq', type=int, default=10,
                       help='checkpoint save frequency (epochs)')
    parser.add_argument('--max_save', type=int, default=5,
                       help='maximum number of checkpoints to keep')
    # 结果保存
    parser.add_argument('--save_result', type=str, default='',
                       help='result image path')
    # 自定义域参数
    parser.add_argument('--train_domains', type=str, default='02,03,04,05', help='Comma-separated list of training domain names')
    parser.add_argument('--test_domains', type=str, default='01', help='Comma-separated list of testing domain names')
    # 自定义子目录参数
    parser.add_argument('--image_subdir', type=str, default='images', help='Subdirectory for images')
    parser.add_argument('--mask_subdir', type=str, default='masks_oc', help='Subdirectory for masks')

    parser.add_argument('--num_classes', type=int, default='1',help='number of classes')

    args = parser.parse_args()

    return args
