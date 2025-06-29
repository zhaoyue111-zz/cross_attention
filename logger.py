# nnunet_logger.py
import os
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

matplotlib.use('agg')

class nnUNetLogger(object):
    """
    这个日志记录器用于记录训练和验证过程中关键指标。
    """
    def __init__(self, verbose: bool = False):
        self.my_fantastic_logging = {
            'train_losses': list(),  # main loss
            'single_losses': list(),  # 单独分割损失
            'consistency_losses': list(),  # 一致性损失
            'memory_losses': list(),  # 对比损失
            'val_dice': list(),
            'val_dice_std': list(),
            'val_memory_losses': list(),
            'learning_rates': list(),
        }
        self.verbose = verbose

    def log(self, key, value, epoch: int):
        assert key in self.my_fantastic_logging.keys() and isinstance(self.my_fantastic_logging[key], list), \
            'This function is only intended to log stuff to lists and to have one entry per epoch'

        if self.verbose:
            print(f'logging {key}: {value} for epoch {epoch}')

        if len(self.my_fantastic_logging[key]) < epoch + 1:
            self.my_fantastic_logging[key].append(value)
        else:
            assert len(self.my_fantastic_logging[key]) == epoch + 1, 'something went horribly wrong. My logging ' \
                                                                      'lists length is off by more than 1'
            print(f'maybe some logging issue!? logging {key} and {value}')
            self.my_fantastic_logging[key][epoch] = value

    def plot_progress_png(self, output_folder):
        epoch = min([len(i) for i in self.my_fantastic_logging.values()]) - 1
        sns.set(font_scale=2.5)
        fig, ax_all = plt.subplots(5, 1, figsize=(30, 90))  # 增加一个子图

        # 绘制训练损失和验证Dice
        ax = ax_all[0]
        x_values = list(range(epoch + 1))
        ax.plot(x_values, self.my_fantastic_logging['train_losses'][:epoch + 1], color='b', label="Main Loss", linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['single_losses'][:epoch + 1], color='g', label="Single Loss", linewidth=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(loc='best')

        # 绘制验证Dice
        ax = ax_all[1]
        ax.plot(x_values, self.my_fantastic_logging['val_dice'][:epoch + 1], color='r', label="Validation Dice", linewidth=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Dice")
        ax.legend(loc='best')

        # 绘制学习率
        ax = ax_all[2]
        ax.plot(x_values, self.my_fantastic_logging['learning_rates'][:epoch + 1], color='g', label="Learning Rate", linewidth=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.legend(loc='best')

        # 绘制Memory Loss
        ax = ax_all[3]
        ax.plot(x_values, self.my_fantastic_logging['memory_losses'][:epoch + 1], color='m', 
                label="Memory Loss", linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['val_memory_losses'][:epoch + 1], color='c', 
                label="Val Memory Loss", linewidth=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Memory Loss")
        ax.legend(loc='best')

        # 绘制Consistency Loss
        ax = ax_all[4]
        ax.plot(x_values, self.my_fantastic_logging['consistency_losses'][:epoch + 1], color='y', 
                label="Consistency Loss", linewidth=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Consistency Loss")
        ax.legend(loc='best')

        plt.tight_layout()
        fig.savefig(os.path.join(output_folder, "progress.png"))
        plt.close()

    def get_checkpoint(self):
        return self.my_fantastic_logging

    def load_checkpoint(self, checkpoint: dict):
        self.my_fantastic_logging = checkpoint
