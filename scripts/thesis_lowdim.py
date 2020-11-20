import argparse
import pathlib
import asyncio
import io
import shutil
import numpy as np
import csv

root_path = pathlib.Path(__file__).parent.absolute().joinpath('..')
experiments_path = root_path.joinpath('experiments')
data_path = root_path.joinpath('scripts/data-lowdim')
latex_data_path = root_path.joinpath('latex/data')

class Architecture:
    def __init__(self, name, extra_cmd=[]):
        self.name = name
        self.extra_cmd = extra_cmd

architectures = [
    Architecture('fnn'),
    Architecture('la-sympnet'), 
    Architecture('n1-la-sympnet'),
    Architecture('n2-la-sympnet'),
    Architecture('g-sympnet'), 
    Architecture('n1-g-sympnet'),
    Architecture('n2-g-sympnet')
]

large_architectures = [
    Architecture('large-fnn'),
    Architecture('large-la-sympnet'), 
    Architecture('large-n1-la-sympnet'),
    Architecture('large-n2-la-sympnet'),
    Architecture('large-g-sympnet'), 
    Architecture('large-n1-g-sympnet'),
    Architecture('large-n2-g-sympnet')
]

class Experiment:
    def __init__(self, name, configurations, cmd, activations, architectures, epochs):
        self.name = name
        self.configurations = configurations
        self.cmd = cmd
        self.activations = activations
        self.architectures = architectures
        self.epochs = epochs

experiments = [
    Experiment('harmonic_oscillator', ['harmonic_oscillator'],
        [
            'python', experiments_path.joinpath('lowdim.py'),
            '--model', 'harmonic_oscillator',
            '--qmin', '-2',
            '--qmax', '2',
            '--pmin', '-2',
            '--pmax', '2',
            '-n', '40'
        ], 
        activations=['sigmoid'], 
        architectures=[Architecture('l-sympnet')], 
        epochs=500
    ),
    Experiment('simple_pendulum_swing', ['swinging_case', 'rotating_case'], 
        [
            'python', experiments_path.joinpath('lowdim.py'),
            '--model', 'simple_pendulum',
            '--qmin', str(-np.pi/2),
            '--qmax', str(np.pi/2),
            '--pmin', str(-np.sqrt(2)),
            '--pmax', str(np.sqrt(2)),
            '-n', '40'
        ], 
        activations=['sigmoid', 'tanh', 'elu'], 
        architectures=architectures, 
        epochs=100000
    ),
    Experiment('simple_pendulum_swing_rot', ['swinging_case', 'rotating_case'], 
        [
            'python', experiments_path.joinpath('lowdim.py'),
            '--model', 'simple_pendulum',
            '--qmin', '-20',
            '--qmax', '20',
            '--pmin', '-2.5',
            '--pmax', '2.5',
            '-n', '400'
        ], 
        activations=['sigmoid', 'tanh', 'elu'], 
        architectures=large_architectures, 
        epochs=300000
    ),
    Experiment('simple_pendulum_swing_rot_fast_lr', ['swinging_case', 'rotating_case'], 
        [
            'python', experiments_path.joinpath('lowdim.py'),
            '--model', 'simple_pendulum',
            '--qmin', '-20',
            '--qmax', '20',
            '--pmin', '-2.5',
            '--pmax', '2.5',
            '-n', '400',
            '-lr', str(1e0)
        ], 
        activations=['sigmoid', 'tanh', 'elu'], 
        architectures=large_architectures, 
        epochs=300000
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

def save_phase_plot(dest: pathlib.Path, qpvec):
    save_csv(dest, qpvec, header='q,p')

def subsample(vec: np.ndarray, every_nth: int):
    """Subsamples rows"""

    vec_sample = vec[::every_nth]
    vec_sample[-1,:] = vec[-1,:] # return last
    return vec_sample

def update_csv(args):
    for exp in experiments:
        curr_exp_dir = data_path.joinpath(exp.name)
        curr_destination_dir = latex_data_path.joinpath(exp.name)

        test_loss_summary = []
        for arch in exp.architectures:

            arch_summary = { 'architecture': arch.name }

            for activation in exp.activations:
                n_subsample = 2 if exp.name == 'harmonic_oscillator' else 200

                # Training loss
                losses = np.load(curr_exp_dir.joinpath(arch.name, activation, 'losses.npy'))
                losses = np.stack([np.arange(0, len(losses)), losses], axis=1)
                save_csv(curr_destination_dir
                    .joinpath(arch.name, activation, 'loss.csv'), subsample(losses,n_subsample), header='epoch,loss')

                # Test loss
                losses = np.load(curr_exp_dir.joinpath(arch.name, activation, 'test_losses.npy'))
                losses = np.stack([np.arange(0, len(losses)), losses], axis=1)
                save_csv(curr_destination_dir
                    .joinpath(arch.name, activation, 'test_loss.csv'), subsample(losses,n_subsample), header='epoch,loss')

                arch_summary['test_loss_{}'.format(activation)] = losses[-1,1]
            
            test_loss_summary.append(arch_summary)
        
        save_dict_as_csv(curr_destination_dir.joinpath('test_loss_summary.csv'), test_loss_summary)

        for config in exp.configurations:

            td_x = np.load(curr_exp_dir
                .joinpath(exp.architectures[0].name, exp.activations[0], config, 'exact_td_x.npy'))
            td_Ham = np.load(curr_exp_dir
                .joinpath(exp.architectures[0].name, exp.activations[0], config, 'exact_td_Ham.npy'))

            save_phase_plot(
                curr_destination_dir.joinpath('exact', config, 'phase_plot.csv'), 
                td_x[:,1:]
            )

            save_csv(
                curr_destination_dir.joinpath('exact', config, 'total_energy.csv'),
                td_Ham,
                't,E'
            )

            for arch in exp.architectures:
                for activation in exp.activations:

                    td_x = np.load(curr_exp_dir
                        .joinpath(arch.name, activation, config, 'epoch{}_td_x.npy'.format(exp.epochs)))
                    td_Ham = np.load(curr_exp_dir
                        .joinpath(arch.name, activation, config, 'epoch{}_td_Ham.npy'.format(exp.epochs)), allow_pickle=True)

                    save_phase_plot(
                        curr_destination_dir.joinpath(arch.name, activation, config, 'phase_plot.csv'), 
                        td_x[:,1:]
                    )

                    save_csv(
                        curr_destination_dir.joinpath(arch.name, activation, config, 'total_energy.csv'),
                        td_Ham,
                        't,E'
                    )


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
