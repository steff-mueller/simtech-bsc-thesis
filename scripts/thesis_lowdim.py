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

architectures = [
    'la-sympnet', 'normalized-la-sympnet', 
    'g-sympnet', 'normalized-g-sympnet'
]
epochs = 10000

class Experiment:
    def __init__(self, name, configurations, cmd):
        self.name = name
        self.configurations = configurations
        self.cmd = cmd

experiments = [
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
                '--architecture', arch,
                '--output-dir', data_path.joinpath(exp.name, arch)
            ])

def save_csv(dest, vec, header):
    if not dest.parent.exists():
        dest.parent.mkdir(parents=True)

    np.savetxt(dest, vec, delimiter=',', header=header, comments='')

def save_phase_plot(dest: pathlib.Path, qpvec):
    save_csv(dest, qpvec, header='q,p')

def update_csv(args):
    for exp in experiments:
        curr_exp_dir = data_path.joinpath(exp.name)
        curr_destination_dir = latex_data_path.joinpath(exp.name)

        for config in exp.configurations:

            # Exact td x
            exact_td_x = np.load(curr_exp_dir
                .joinpath(architectures[0], config, 'exact_td_x.npy'))
            save_phase_plot(curr_destination_dir
                .joinpath('exact', config, 'phase_plot.csv'), exact_td_x)

            for arch in architectures:
                
                # Predicted td x by architecture
                td_x = np.load(curr_exp_dir
                    .joinpath(arch, config, 'epoch{}_td_x.npy'.format(epochs)))
                save_phase_plot(curr_destination_dir
                    .joinpath(arch, config, 'phase_plot.csv'), td_x)

                # Training loss
                losses = np.load(curr_exp_dir
                    .joinpath(arch, 'losses.npy'))
                losses = np.stack([
                    np.arange(0, len(losses)),
                    losses
                ], axis=1)
                losses = losses[::5]
                save_csv(curr_destination_dir
                    .joinpath(arch, 'loss.csv'), losses, header='epoch,loss')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Orchestrator for simulations and data manipulation.')
    subparsers = parser.add_subparsers()

    parser_run = subparsers.add_parser('run-experiments')
    parser_run.set_defaults(func=run_experiments)

    parser_run = subparsers.add_parser('update-csv')
    parser_run.set_defaults(func=update_csv)

    args = parser.parse_args()
    args.func(args)
