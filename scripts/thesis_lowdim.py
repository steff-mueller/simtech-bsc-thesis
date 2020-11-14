import argparse
import pathlib
import subprocess
import sys
import shutil
import numpy as np

root_path = pathlib.Path(__file__).parent.absolute().joinpath('..')
experiments_path = root_path.joinpath('experiments')
data_path = root_path.joinpath('scripts/data-lowdim')
latex_data_path = root_path.joinpath('latex/data')

simple_pendulum_swing_rot_dir = data_path.joinpath('simple_pendulum_swing_rot')
simple_pendulum_swing_rot_latex = latex_data_path.joinpath('simple_pendulum_swing_rot')

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
epochs = 2000

class Experiment:
    def __init__(self, name, configurations, cmd):
        self.name = name
        self.configurations = configurations
        self.cmd = cmd

experiments = [
    Experiment('simple_pendulum_swing', ['swinging_case', 'rotating_case'], [
        'python', experiments_path.joinpath('lowdim.py'),
        '--model', 'simple_pendulum',
        '--qmin', str(-np.pi/2),
        '--qmax', str(np.pi/2),
        '--pmin', str(-np.sqrt(2)),
        '--pmax', str(np.sqrt(2)),
        '-n', '40',
        '--epochs', str(epochs)
    ]),
    Experiment('simple_pendulum_rot', ['swinging_case', 'rotating_case'], [
        'python', experiments_path.joinpath('lowdim.py'),
        '--model', 'simple_pendulum',
        '--qmin', '-20',
        '--qmax', '20',
        '--pmin', '0.5',
        '--pmax', '2.5',
        '-n', '400',
        '--epochs', str(epochs)
    ]),
    Experiment('simple_pendulum_swing_rot', ['swinging_case', 'rotating_case'], [
        'python', experiments_path.joinpath('lowdim.py'),
        '--model', 'simple_pendulum',
        '--qmin', '-20',
        '--qmax', '20',
        '--pmin', '-2.5',
        '--pmax', '2.5',
        '-n', '400',
        '--epochs', str(epochs)
    ])
]

def execute(command):
    subprocess.run(command, check=True, stdout=sys.stdout, stderr=subprocess.STDOUT)

def run_experiments(args):
    # Clean up
    if data_path.exists():
        shutil.rmtree(data_path)

    for exp in experiments:
        for arch in architectures:
            execute(exp.cmd + [
                '--architecture', arch.name,
                '--output-dir', data_path.joinpath(exp.name, arch.name)
            ] + arch.extra_cmd)

def save_csv(dest, vec, header):
    if not dest.parent.exists():
        dest.parent.mkdir(parents=True)

    np.savetxt(dest, vec, delimiter=',', header=header, comments='')

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

        for arch in architectures:
            # Training loss
            losses = np.load(curr_exp_dir.joinpath(arch.name, 'losses.npy'))
            losses = np.stack([
                np.arange(0, len(losses)),
                losses
            ], axis=1)
            save_csv(curr_destination_dir
                .joinpath(arch.name, 'loss.csv'), subsample(losses,10), header='epoch,loss')

            # Test loss
            losses = np.load(curr_exp_dir.joinpath(arch.name, 'test_losses.npy'))
            losses = np.stack([
                np.arange(0, len(losses)),
                losses
            ], axis=1)
            save_csv(curr_destination_dir
                .joinpath(arch.name, 'test_loss.csv'), subsample(losses,10), header='epoch,loss')

        for config in exp.configurations:

            td_x = np.load(curr_exp_dir
                .joinpath(architectures[0].name, config, 'exact_td_x.npy'))
            td_Ham = np.load(curr_exp_dir
                .joinpath(architectures[0].name, config, 'exact_td_Ham.npy'))

            save_phase_plot(
                curr_destination_dir.joinpath('exact', config, 'phase_plot.csv'), 
                td_x[:,1:]
            )

            save_csv(
                curr_destination_dir.joinpath('exact', config, 'total_energy.csv'),
                td_Ham,
                't,E'
            )

            for arch in architectures:

                td_x = np.load(curr_exp_dir
                    .joinpath(arch.name, config, 'epoch{}_td_x.npy'.format(epochs)))
                td_Ham = np.load(curr_exp_dir
                    .joinpath(arch.name, config, 'epoch{}_td_Ham.npy'.format(epochs)), allow_pickle=True)

                save_phase_plot(
                    curr_destination_dir.joinpath(arch.name, config, 'phase_plot.csv'), 
                    td_x[:,1:]
                )

                save_csv(
                    curr_destination_dir.joinpath(arch.name, config, 'total_energy.csv'),
                    td_Ham,
                    't,E'
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Orchestrator for simulations and data manipulation.')
    subparsers = parser.add_subparsers()

    parser_run = subparsers.add_parser('run-experiments')
    parser_run.set_defaults(func=run_experiments)

    parser_run = subparsers.add_parser('update-csv')
    parser_run.set_defaults(func=update_csv)

    args = parser.parse_args()
    args.func(args)
