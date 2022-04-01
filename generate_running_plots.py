import os
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np


def compute_moving_average(result_all, average_num=100):
    result_moving_average_all = []
    moving_values = deque([], maxlen=average_num)
    for result in result_all:
        moving_values.append(result)
        result_moving_average_all.append(np.mean(moving_values))
    return result_moving_average_all


def plot_curve(draw_keys, x_dict, y_dict, plot_name,
               linewidth=3, xlabel=None, ylabel=None,
               title=None,
               apply_rainbow=False,
               img_size=(8, 5), axis_size=15, legend_size=15):
    import matplotlib as mpl
    mpl.rcParams['xtick.labelsize'] = axis_size
    mpl.rcParams['ytick.labelsize'] = axis_size
    fig = plt.figure(figsize=img_size)
    ax = fig.add_subplot(1, 1, 1)
    from matplotlib.pyplot import cm
    if apply_rainbow:
        color = cm.rainbow(np.linspace(0, 1, len(draw_keys)))
        for key, c in zip(draw_keys, color):
            plt.plot(x_dict[key], y_dict[key], label=key, linewidth=linewidth, c=c)
    else:
        for key in draw_keys:
            plt.plot(x_dict[key], y_dict[key], label=key, linewidth=linewidth)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%01.2lf'))
    if legend_size is not None:
        plt.legend(fontsize=legend_size, loc='upper right')
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=axis_size)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=axis_size)
    if title is not None:
        plt.title(title, fontsize=axis_size)
    if not plot_name:
        plt.show()
    else:
        plt.savefig('{0}.png'.format(plot_name))
    plt.close()


def read_running_logs(log_path, read_keys):
    read_running_logs = {}

    with open(log_path, 'r') as file:
        running_logs = file.readlines()
    old_results = None

    key_indices = {}
    record_keys = running_logs[1].replace('\n', '').split(',')
    for key in read_keys:
        key_idx = record_keys.index(key)
        key_indices.update({key: key_idx})
        read_running_logs.update({key: []})

    for running_performance in running_logs[2:]:
        log_items = running_performance.split(',')
        if len(log_items) != len(record_keys):
            # continue
            results = old_results
        else:
            try:
                results = [item.replace("\n", "") for item in log_items]
            except:
                results = old_results
                # continue
        if results is None:
            continue
        for key in read_keys:
            read_running_logs[key].append(float(results[key_indices[key]]))
    return read_running_logs


def plot_results(results_moving_average, label='Rewards', save_label=''):
    plot_x_dict = {'PPO': [i for i in range(len(results_moving_average))]}
    plot_y_dict = {'PPO': results_moving_average}
    plot_curve(draw_keys=['PPO'],
               x_dict=plot_x_dict,
               y_dict=plot_y_dict,
               xlabel='Episode',
               ylabel=label,
               title='{0}'.format(save_label),
               plot_name='./plot_results/{0}'.format(save_label))


def generate_plots():
    modes = ['train']

    for mode in modes:
        plot_key = ['r', 'l', 't']
        log_path = './icrl/save_models/HCWithPos-v0_HCWithPosTest-v0_aclr_0.9_cl_20_clr_0.05_crc_0.5_ctkno_2.5_d_cuda_ep_HCWithPos-New_er_10_ft_200000_nt_1_psis_True_tk_0.01_s_11_sid_-1/'

        if mode == 'train':
            log_path += 'monitor.csv'
        else:
            log_path += 'test/test.monitor.csv'

        # rewards, is_collision, is_off_road, is_goal_reached, is_time_out = read_running_logs(log_path=log_path)
        results = read_running_logs(log_path=log_path, read_keys=plot_key)

        if not os.path.exists('./plot_results/' + log_path.split('/')[3]):
            os.mkdir('./plot_results/' + log_path.split('/')[3])

        for idx in range(len(plot_key)):
            results_moving_average = compute_moving_average(result_all=results[plot_key[idx]], average_num=100)
            plot_results(results_moving_average,
                         label=plot_key[idx],
                         save_label=log_path.split('/')[3] + '/' + plot_key[idx] + '_' + mode)


if __name__ == "__main__":
    generate_plots()
