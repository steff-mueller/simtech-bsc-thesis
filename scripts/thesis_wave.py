import argparse
import pathlib
import asyncio
import shutil
import io
import os
import numpy as np
import csv

os.environ["PYTHONUNBUFFERED"] = "1"

root_path = pathlib.Path(__file__).parent.absolute().joinpath('..')
experiments_path = root_path.joinpath('experiments')
data_path = root_path.joinpath('scripts/data-wave')
latex_data_path = root_path.joinpath('latex/data-wave')

class Architecture:
    def __init__(self, name, extra_cmd=[]):
        self.name = name
        self.extra_cmd = extra_cmd

class Experiment:
    def __init__(self, name, cmd, activations, architectures, epochs, l, n_x):
        self.name = name
        self.cmd = cmd
        self.activations = activations
        self.architectures = architectures
        self.epochs = epochs
        self.l = l
        self.n_x = n_x

experiments = [
    Experiment('transport', 
        [
            'python', experiments_path.joinpath('wave.py'),
            '--model', 'transport',
            '--dt', '0.01',
            '-l', '1',
            '--nx', '100',
            '--time-total', '10',
            '--time-training', '1',
            '-lr', str(1e-1)
        ],
        n_x = 100,
        l = 1,
        activations=['sigmoid'],
        architectures=[
            Architecture('linear_canonical'),
            Architecture('linear_fd'),
            Architecture('linear_cnn')
        ], 
        epochs=2000
    ),
    Experiment('sine_gordon_sin',
        [
            'python', experiments_path.joinpath('wave.py'),
            '--model', 'sine_gordon',
            '--dt', '0.01',
            '-l', '50',
            '--nx', '2000',
            '--time-total', '10',
            '--time-training', '1',
            '-lr', str(1e-2)
        ], 
        n_x = 2000,
        l = 50,
        activations=['sigmoid'], 
        architectures=[
            Architecture('nonlinear'),
        ], 
        epochs=2000
    ),
    Experiment('sine_gordon', 
        [
            'python', experiments_path.joinpath('wave.py'),
            '--model', 'sine_gordon',
            '--dt', '0.01',
            '-l', '50',
            '--nx', '2000',
            '--time-total', '10',
            '--time-training', '1',
            '-lr', str(1e-2),
            '--log-intermediate'
        ], 
        n_x = 2000,
        l = 50,
        activations=['sigmoid', 'tanh', 'elu'], 
        architectures=[
            Architecture('gradient'),
            Architecture('n1-gradient'), 
            Architecture('n2-gradient'),
            Architecture('cnn')
        ], 
        epochs=2000
    )
]

async def forward_stream(source: asyncio.StreamReader, dest: io.TextIOWrapper):
    while True:
        buffer = await source.read(10)
        if buffer:
            dest.write(buffer)
            dest.flush()
        else:
            break

async def execute(cmd, log_file):
    path = pathlib.Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    with io.open(log_file, 'wb') as log_writer:
        print('Executing ', cmd)
        proc = await asyncio.create_subprocess_exec(*cmd, 
            stdout=asyncio.subprocess.PIPE, 
            stderr=asyncio.subprocess.PIPE)

        await forward_stream(proc.stdout, log_writer)

        _, stderr = await proc.communicate()
        if stderr:
            print('Error while executing', cmd)
            print(stderr)
            log_writer.write(stderr)
        else:
            print('Finished executing', cmd)


async def safe_execute(cmd, log_file, sem: asyncio.Semaphore):
    async with sem:
        return await execute(cmd, log_file)
        

async def run_experiments(args):
    # Clean up
    if data_path.exists():
        shutil.rmtree(data_path)

    sem = asyncio.BoundedSemaphore(args.parallel)

    tasks = []
    for exp in experiments:
        for arch in exp.architectures:
            for activation in exp.activations:
                cmd = exp.cmd + [
                    '--architecture', arch.name,
                    '--activation', activation,
                    '--output-dir', data_path.joinpath(exp.name, arch.name, activation),
                    '--epochs', str(exp.epochs)
                ] + arch.extra_cmd
                log_file = data_path.joinpath(exp.name, arch.name, activation, 'run.log')

                task = asyncio.create_task(safe_execute(cmd, log_file, sem))
                tasks.append(task)

    await asyncio.gather(*tasks)


def save_csv(dest, vec, header):
    if not dest.parent.exists():
        dest.parent.mkdir(parents=True)

    np.savetxt(dest, vec, delimiter=',', header=header, comments='')

def save_dict_as_csv(dest, dict_array):
    with open(dest, 'w', newline='') as csvfile:
        fieldnames = dict_array[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for dict in dict_array:
            writer.writerow(dict)


def subsample(vec: np.ndarray, every_nth: int):
    """Subsamples rows"""

    vec_sample = vec[::every_nth]
    vec_sample[-1,:] = vec[-1,:] # return last
    return vec_sample


def print_q_over_domain(td_x: np.ndarray, t: float, exp: Experiment, dest: str):
    x_t = td_x[np.isclose(td_x[:,0], t)]
    d = int((x_t.shape[1] - 1)/2)
    coord = np.linspace(-exp.l/2, exp.l/2, exp.n_x)
    q = x_t[0,1:d+1]
    save_csv(dest, np.stack((coord,q), axis=1), 'x,q')


def update_csv(args):
    for exp in experiments:
        curr_exp_dir = data_path.joinpath(exp.name)
        curr_destination_dir = latex_data_path.joinpath(exp.name)

        td_x = np.load(curr_exp_dir
            .joinpath(exp.architectures[0].name, exp.activations[0], 'exact_td_x.npy'))
        
        print_q_over_domain(td_x, 9.0, exp, curr_destination_dir.joinpath('exact', 'q_t9.csv'))

        for arch in exp.architectures:
            for activation in exp.activations:
                # Training loss
                losses = np.load(curr_exp_dir.joinpath(arch.name, activation, 'losses.npy'))
                losses = np.stack([np.arange(0, len(losses)), losses], axis=1)
                save_csv(curr_destination_dir
                    .joinpath(arch.name, activation, 'loss.csv'), subsample(losses,5), header='epoch,loss')

                # Test loss
                losses = np.load(curr_exp_dir.joinpath(arch.name, activation, 'test_losses.npy'))
                losses = np.stack([np.arange(0, len(losses)), losses], axis=1)
                save_csv(curr_destination_dir
                    .joinpath(arch.name, activation, 'test_loss.csv'), subsample(losses,5), header='epoch,loss')

                td_x = np.load(curr_exp_dir
                    .joinpath(arch.name, activation, 'epoch{}_td_x.npy'.format(exp.epochs)))

                print_q_over_domain(td_x, 9.0, exp, curr_destination_dir.joinpath(arch.name, activation, 'q_t9.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Orchestrator for simulations and data manipulation.')
    subparsers = parser.add_subparsers()

    parser_run = subparsers.add_parser('run-experiments')
    parser_run.add_argument('--parallel', default=2, type=int)
    parser_run.set_defaults(func=lambda args: asyncio.run(run_experiments(args)))

    parser_run = subparsers.add_parser('update-csv')
    parser_run.set_defaults(func=update_csv)

    args = parser.parse_args()
    args.func(args)