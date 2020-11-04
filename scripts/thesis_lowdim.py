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

def save_phase_plot(dest: pathlib.Path, qpvec):
    if not dest.parent.exists():
        dest.parent.mkdir()

    np.savetxt(dest, qpvec, delimiter=',', header='q,p', comments='')

def execute(command):
    subprocess.run(command, check=True, stdout=sys.stdout, stderr=subprocess.STDOUT)

def run_experiments(args):
    # Clean up
    if data_path.exists():
        shutil.rmtree(data_path)

    simple_pendulum_swing_rot_cmd = [
        'python', experiments_path.joinpath('lowdim.py'),
        '--output-dir', data_path.joinpath('simple_pendulum_swing_rot'),
        '--model', 'simple_pendulum',
        '--qmin', '-20',
        '--qmax', '20',
        '--pmin', '-2.5',
        '--pmax', '2.5',
        '-n', '400',
        '--epochs', '10000'
    ]

    for arch in architectures:
        execute(simple_pendulum_swing_rot_cmd + [
            '--architecture', arch,
            '--output-dir', simple_pendulum_swing_rot_dir.joinpath(arch)
        ])


def update_csv(args):
    exact_td_x_swing_rot = np.load(simple_pendulum_swing_rot_dir
        .joinpath('la-sympnet/rotating_case/exact_td_x.npy'))
    save_phase_plot(simple_pendulum_swing_rot_latex
        .joinpath('exact_phase_plot.csv'), exact_td_x_swing_rot)

    for arch in architectures:
        td_x_swing_rot = np.load(simple_pendulum_swing_rot_dir
            .joinpath(arch, 'rotating_case/epoch10000_td_x.npy'))
        save_phase_plot(simple_pendulum_swing_rot_latex
            .joinpath('{}_phase_plot.csv'.format(arch)), td_x_swing_rot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Orchestrator for simulations and data manipulation.')
    subparsers = parser.add_subparsers()

    parser_run = subparsers.add_parser('run-experiments')
    parser_run.set_defaults(func=run_experiments)

    parser_run = subparsers.add_parser('update-csv')
    parser_run.set_defaults(func=update_csv)

    args = parser.parse_args()
    args.func(args)
