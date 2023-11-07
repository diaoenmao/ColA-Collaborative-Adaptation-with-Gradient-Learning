import argparse
import os
import itertools

parser = argparse.ArgumentParser(description='config')
parser.add_argument('--run', default='train', type=str)
parser.add_argument('--init_gpu', default=0, type=int)
parser.add_argument('--num_gpu', default=4, type=int)
parser.add_argument('--world_size', default=1, type=int)
parser.add_argument('--init_seed', default=0, type=int)
parser.add_argument('--round', default=4, type=int)
parser.add_argument('--experiment_step', default=1, type=int)
parser.add_argument('--num_experiment', default=1, type=int)
parser.add_argument('--resume_mode', default=0, type=int)
parser.add_argument('--mode', default=None, type=str)
parser.add_argument('--split_round', default=65535, type=int)
parser.add_argument('--task_name', default=None, type=str)
args = vars(parser.parse_args())


def make_controls(script_name, init_seeds, world_size, num_experiment, resume_mode, control_name):
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
    control_names = [control_names]
    controls = script_name + init_seeds + world_size + num_experiment + resume_mode + control_names
    controls = list(itertools.product(*controls))
    return controls


def main():
    run = args['run']
    init_gpu = args['init_gpu']
    num_gpu = args['num_gpu']
    world_size = args['world_size']
    round = args['round']
    experiment_step = args['experiment_step']
    init_seed = args['init_seed']
    num_experiment = args['num_experiment']
    resume_mode = args['resume_mode']
    mode = args['mode']
    split_round = args['split_round']
    task_name = args['task_name']
    gpu_ids = [','.join(str(i) for i in list(range(x, x + world_size))) for x in
               list(range(init_gpu, init_gpu + num_gpu, world_size))]
    init_seeds = [list(range(init_seed, init_seed + num_experiment, experiment_step))]
    world_size = [[world_size]]
    num_experiment = [[experiment_step]]
    resume_mode = [[resume_mode]]
    filename = '{}_{}_{}'.format(run, mode, task_name)
    if task_name == 's2s':
        data_names = ['fpb-sa', 'wikisql', 'samsum', 'e2enlg', 'webnlg-2017', 'dart']
        model_names = ['bart-base']
    elif task_name == 'clm':
        data_names = ['dolly-15k']
        # model_names = ['llama-2']
        model_names = ['gpt2']
    elif task_name == 'sc':
        data_names = ['glue-cola', 'glue-mnli', 'glue-mrpc', 'glue-qnli', 'glue-qqp', 'glue-rte', 'glue-sst2',
                      'glue-stsb']
        model_names = ['roberta-base']
    elif task_name == 'ic':
        data_names = ['MNIST', 'CIFAR10']
        model_names = ['linear', 'mlp', 'cnn']
    else:
        raise ValueError('Not valid task name')
    if mode == 'full':
        script_name = [['{}_model.py'.format(run)]]
        if task_name == 'ic':
            batch_size = ['256']
        else:
            batch_size = ['32']
        control_name = [[data_names, model_names, [task_name], ['full'], batch_size]]
        controls = make_controls(script_name, init_seeds, world_size, num_experiment, resume_mode, control_name)
    elif mode == 'peft':
        if task_name == 'ic':
            ft_name = ['lora']
            batch_size = ['256']
        else:
            ft_name = ['lora', 'adalora', 'ia3', 'promptune', 'prefixtune', 'ptune']
            if model_names[0] == 'llama-2':
                batch_size = ['8']
            else:
                batch_size = ['32']
        script_name = [['{}_peft.py'.format(run)]]
        control_name = [[data_names, model_names, [task_name], ft_name, batch_size]]
        controls = make_controls(script_name, init_seeds, world_size, num_experiment, resume_mode, control_name)
    elif mode == 'cola':
        ft_name = ['cola-lowrank-1', 'cola-linear-1', 'cola-mlp-1']
        if task_name == 'ic':
            batch_size = ['256']
        else:
            if model_names[0] == 'llama-2':
                batch_size = ['8']
            else:
                batch_size = ['32']
        script_name = [['{}_cola.py'.format(run)]]
        control_name = [[data_names, model_names, [task_name], ft_name, batch_size]]
        controls = make_controls(script_name, init_seeds, world_size, num_experiment, resume_mode, control_name)
    elif mode == 'cola_step':
        ft_name = ['cola-lowrank-1', 'cola-lowrank-2', 'cola-lowrank-4', 'cola-lowrank-8']
        if task_name == 'ic':
            batch_size = ['64']
        else:
            batch_size = ['8']
        script_name = [['{}_cola.py'.format(run)]]
        control_name = [[data_names, model_names, [task_name], ft_name, batch_size]]
        controls = make_controls(script_name, init_seeds, world_size, num_experiment, resume_mode, control_name)
    elif mode == 'cola_dist':
        data_names = ['dolly-15k']
        ft_name = ['cola-lowrank-1', 'cola-lowrank~linear-1', 'cola-lowrank~mlp-1']
        batch_size = ['32']
        dist_mode = ['alone', 'col']
        script_name = [['{}_cola_dist.py'.format(run)]]
        control_name = [[data_names, model_names, [task_name], ft_name, batch_size, dist_mode]]
        controls = make_controls(script_name, init_seeds, world_size, num_experiment, resume_mode, control_name)
    else:
        raise ValueError('Not valid mode')
    s = '#!/bin/bash\n'
    j = 1
    k = 1
    for i in range(len(controls)):
        controls[i] = list(controls[i])
        s = s + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --init_seed {} --world_size {} --num_experiment {} ' \
                '--resume_mode {} --control_name {}&\n'.format(gpu_ids[i % len(gpu_ids)], *controls[i])
        if i % round == round - 1:
            s = s[:-2] + '\nwait\n'
            if j % split_round == 0:
                print(s)
                run_file = open(os.path.join('scripts', '{}_{}.sh'.format(filename, k)), 'w')
                run_file.write(s)
                run_file.close()
                s = '#!/bin/bash\n'
                k = k + 1
            j = j + 1
    if s != '#!/bin/bash\n':
        if s[-5:-1] != 'wait':
            s = s + 'wait\n'
        print(s)
        if not os.path.exists('scripts'):
            os.makedirs('scripts')
        run_file = open(os.path.join('scripts', '{}_{}.sh'.format(filename, k)), 'w')
        run_file.write(s)
        run_file.close()
    return


if __name__ == '__main__':
    main()
