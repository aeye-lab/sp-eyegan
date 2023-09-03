from __future__ import annotations

import argparse
import subprocess

import matplotlib.pyplot as plt
import numpy as np


def system_call(command):
    p = subprocess.Popen([command], stdout=subprocess.PIPE, shell=True)
    return p.stdout.read()


def get_sep_lines(nvidia_out):
    lines = str(nvidia_out).split('\\n')
    out_ids = []
    for i in range(len(lines)):
        if '===' in lines[i]:
            out_ids.append(i)
    return out_ids


def get_potential_gpu_lines(nvidia_out, nvidia_lines=3, skip_lines=1):
    lines = str(nvidia_out).split('\\n')
    start_line = get_sep_lines(nvidia_out)[0]
    out_gpu_lines_list = []
    gpu_lines = []
    counter_n = 0
    for i in np.arange(start_line + 1, len(lines), 1):
        if counter_n < nvidia_lines:
            gpu_lines.append(lines[i])
            counter_n += 1
        else:
            if len(gpu_lines[0].split('|')) == 5:
                out_gpu_lines_list.append(gpu_lines)
            counter_n = 0
            gpu_lines = []
    return out_gpu_lines_list


def get_gpu_numbers_mib(nvidia_out):
    gpu_lines = get_potential_gpu_lines(nvidia_out)
    gpu_dict = dict()
    for i in range(len(gpu_lines)):
        gpu_number = int(gpu_lines[i][0].split('|')[1].split()[0])
        ram_string = gpu_lines[i][1].split('|')[2].strip()
        occ   = int(ram_string.split('/')[0].replace('MiB', ''))
        avail = int(ram_string.split('/')[1].replace('MiB', ''))
        free  = avail - occ
        gpu_dict[gpu_number] = {
            'free': free,
            'occ': occ,
            'avail': avail,
        }
    return gpu_dict


def get_gpu_dict():
    nvidia_out = system_call('nvidia-smi')
    return get_gpu_numbers_mib(nvidia_out)


def select_gpu(min_free):
    nvidia_out = system_call('nvidia-smi')
    gpu_dict = get_gpu_numbers_mib(nvidia_out)
    free_list = []
    gpu_number_list = []
    for gpu_number in gpu_dict:
        gpu_number_list.append(gpu_number)
        free_list.append(gpu_dict[gpu_number]['free'])
    sort_ids = np.argsort(free_list)[::-1]
    best_gpu = sort_ids[0]
    if free_list[best_gpu] >= min_free:
        return gpu_number_list[best_gpu]
    else:
        return -1


def get_command_for_process_id(process_id):
    command = system_call('ps -p ' + str(process_id) + ' -o args').strip().decode('utf-8').replace('COMMAND', '').replace('\n', '')
    return command


def get_user_for_process_id(process_id):
    return(system_call('ps -o user= -p ' + str(process_id)).strip().decode('utf-8'))


def get_starttime_for_process_id(process_id):
    return(system_call('ps -p ' + str(process_id) + ' -o lstart').strip().decode('utf-8').replace('STARTED', '').replace('\n', ''))


def get_process_dict():
    nvidia_out = system_call('nvidia-smi')
    lines = str(nvidia_out).split('\\n')
    process_lines = get_sep_lines(nvidia_out)[1]
    process_dict = dict()
    for i in np.arange(process_lines, len(lines)):
        if '===' in lines[i] or '---' in lines[i]:
            continue
        split_line = lines[i].split()
        if len(split_line) != 9:
            continue
        # print(split_line)
        gpu_number      = int(split_line[1])
        process_number  = int(split_line[4])
        process_user    = get_user_for_process_id(process_number)
        process_command = get_command_for_process_id(process_number)
        starttime       = get_starttime_for_process_id(process_number)
        usage           = int(split_line[7].replace('MiB', ''))
        process_dict[process_number] = {
            'gpu': gpu_number,
            'usage': usage,
            'user': process_user,
            'command': process_command,
            'starttime': starttime,
        }
    return process_dict


def get_gpu_usage(gpu_dict, process_dict, gpu_name):
    usage_dict = dict()
    process_list_dict = dict()
    free = gpu_dict[gpu_name]['free']
    usage_dict['free'] = free
    process_list_dict['free'] = [-1]
    for process_id in process_dict.keys():
        cur_p_dict = process_dict[process_id]
        if cur_p_dict['gpu'] == gpu_name:
            cur_name = cur_p_dict['user']
            if cur_name not in usage_dict:
                usage_dict[cur_name] = 0
                process_list_dict[cur_name] = []
            usage_dict[cur_name] += cur_p_dict['usage']
            process_list_dict[cur_name].append(process_id)
    return usage_dict, process_list_dict


def plot_gpu_usage():
    process_dict = get_process_dict()
    gpu_dict = get_gpu_dict()

    num_rows = 2
    num_cols = int(np.ceil(len(gpu_dict.keys()) / num_rows))
    fig, axs = plt.subplots(num_cols, num_rows, figsize=(num_rows*5, num_cols*5))
    gpu_list = list(gpu_dict.keys())
    counter = 0
    for n_r in range(num_rows):
        for n_c in range(num_cols):
            cur_gpu = gpu_list[counter]
            usage_dict, process_list_dict = get_gpu_usage(gpu_dict, process_dict, cur_gpu)

            users = list(usage_dict.keys())
            usages = list(usage_dict.values())
            max_id = np.argmax(usages)
            explodes = np.zeros([len(users)])
            explodes[max_id] = 0.05

            axs[n_c, n_r].pie(
                x=usages, autopct='%.1f%%', explode=explodes, labels=users, pctdistance=0.5,
            )
            axs[n_c, n_r].set_title('GPU ' + str(cur_gpu))
            counter += 1
    plt.show()


def print_gpu_usage():
    process_dict = get_process_dict()
    gpu_dict = get_gpu_dict()

    gpu_list = list(gpu_dict.keys())
    for cur_gpu in gpu_list:
        print('GPU: ' + str(cur_gpu))
        usage_dict, process_list_dict = get_gpu_usage(gpu_dict, process_dict, cur_gpu)

        users = list(usage_dict.keys())
        usages = list(usage_dict.values())
        sort_ids = np.argsort(usages)[::-1]
        for i in range(len(sort_ids)):
            cur_usage = usages[sort_ids[i]]
            print(
                '    ' + users[sort_ids[i]] + ': ' + str(cur_usage) +
                'MiB [' + str(np.round(cur_usage / 1000., decimals=2)) + ' GiB]',
            )
            print('        Processes: ' + str(process_list_dict[users[sort_ids[i]]]))


def print_commands_for_user(username):
    process_dict = get_process_dict()
    user_process_dict = dict()
    for process_id in process_dict:
        cur_proc_dict = process_dict[process_id]
        if cur_proc_dict['user'] == username:
            user_process_dict[process_id] = process_dict[process_id]
    print('processes for ' + str(username))
    for proc in user_process_dict:
        print(
            '    ' + str(proc) + ': ' + str(user_process_dict[proc]['command']) +
            ' [GPU:' + str(user_process_dict[proc]['gpu']) +
            ', Usage:' + str(user_process_dict[proc]['usage']) + ' MiB, ' +
            'start-time: ' + str(user_process_dict[proc]['starttime']) + ']',
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--user', type=str)
    args = parser.parse_args()
    user = args.user

    print_gpu_usage()
    print_commands_for_user(user)


if __name__ == '__main__':
    raise SystemExit(main())
