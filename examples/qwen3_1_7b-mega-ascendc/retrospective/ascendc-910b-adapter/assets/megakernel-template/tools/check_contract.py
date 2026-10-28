#!/usr/bin/env python3
"""Check duplicated literal dimensions without importing torch or opening an NPU.

This checks host/device agreement, not the validity of a new placement. Adapt
this checker with the schema if a port replaces literals by generated values.
"""
import ast
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SHARED = ('L', 'H', 'KVH', 'D', 'II', 'VV', 'NB', 'SLT', 'NSL', 'STEP',
          'SPADG', 'NQKV', 'NQKV_PAD', 'NO_PAD', 'NGU', 'ND_PAD', 'SPAN', 'SPANP')


def literals(path):
    result = {}
    for node in ast.parse(path.read_text()).body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError, SyntaxError):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                result[target.id] = value
            elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (tuple, list)):
                for key, item in zip(target.elts, value):
                    if isinstance(key, ast.Name):
                        result[key.id] = item
    return result


def main():
    declarations = '\n'.join((ROOT / name).read_text() for name in (
        'kernel/model/model_config.hpp', 'kernel/backend/ascend910b/layout.hpp'))
    device = {name: int(value) for name, value in re.findall(
        r'constexpr\s+int\s+(\w+)\s*=\s*(\d+)\s*;', declarations)}
    errors = []
    for filename in ('model_run.py', 'test_mega.py'):
        host = literals(ROOT / filename)
        for key in SHARED:
            if key not in device or key not in host:
                errors.append(f'{filename}: {key} must remain a supported literal or update this checker')
            elif device[key] != host[key]:
                errors.append(f'{filename}: {key}={host[key]} != device {device[key]}')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'CONTRACT PASS: {len(SHARED)} shared literal dimensions agree in device/runtime/reference')
    print('This does not validate hard-coded strides, resource sizes, ABI or new-model semantics.')


if __name__ == '__main__':
    main()
