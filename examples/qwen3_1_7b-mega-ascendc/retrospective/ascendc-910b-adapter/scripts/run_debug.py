#!/usr/bin/env python3
"""Run moved debug scripts with the flat working directory their children expect."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='check links without importing torch or using an NPU')
    parser.add_argument('script', help='a filename under the project debug/ directory')
    parser.add_argument('args', nargs=argparse.REMAINDER)
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    target = root / 'debug' / options.script
    if Path(options.script).name != options.script or not target.is_file() or target.suffix != '.py':
        parser.error('script must name an existing debug/*.py file')
    with tempfile.TemporaryDirectory(prefix='ascendc-retrospective-') as work:
        stage = Path(work)
        for directory in (root, root / 'debug'):
            for source in directory.glob('*.py'):
                destination = stage / source.name
                if destination.exists():
                    parser.error(f'ambiguous script name: {source.name}')
                destination.symlink_to(source)
        for name in ('kernel', 'profiling', 'acts', 'prepared'):
            source = root / name
            if source.exists():
                (stage / name).symlink_to(source, target_is_directory=source.is_dir())
        for name in ('model_run.py', 'test_mega.py', 'single_decode_check.py', 'isolated_decode_suite.py', 'kernel'):
            if not (stage / name).exists():
                parser.error(f'missing required project artifact: {name}')
        command = [sys.executable, options.script, *options.args]
        print('Project:', root, flush=True)
        print('Command:', command, flush=True)
        print('Output profiling/ links to the project profiling/ directory.', flush=True)
        if options.dry_run:
            assert (stage / options.script).resolve() == target.resolve()
            print('Flat script links and child-script dependencies verified; no NPU work executed.')
            return 0
        environment = dict(os.environ)
        environment['PYTHONPATH'] = os.pathsep.join([str(stage), str(root), environment.get('PYTHONPATH', '')])
        return subprocess.call(command, cwd=stage, env=environment)


if __name__ == '__main__':
    raise SystemExit(main())
