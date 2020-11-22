import argparse
import pathlib
import asyncio
import shutil
import io
import os

os.environ["PYTHONUNBUFFERED"] = "1"

root_path = pathlib.Path(__file__).parent.absolute().joinpath('..')
experiments_path = root_path.joinpath('experiments')
data_path = root_path.joinpath('scripts/data-wave')
latex_data_path = root_path.joinpath('latex/data')

class Architecture:
    def __init__(self, name, extra_cmd=[]):
        self.name = name
        self.extra_cmd = extra_cmd

class Experiment:
    def __init__(self, name, cmd, activations, architectures, epochs):
        self.name = name
        self.cmd = cmd
        self.activations = activations
        self.architectures = architectures
        self.epochs = epochs

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
            '--nx', '4000',
            '--time-total', '10',
            '--time-training', '0.1'
        ], 
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
            '--nx', '4000',
            '--time-total', '10',
            '--time-training', '0.1'
        ], 
        activations=['sigmoid', 'tanh', 'elu'], 
        architectures=[
            Architecture('gradient'),
            Architecture('n1-gradient'), 
            Architecture('n2-gradient'),
            Architecture('cnn')
        ], 
        epochs=100000
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


def update_csv():
    pass


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