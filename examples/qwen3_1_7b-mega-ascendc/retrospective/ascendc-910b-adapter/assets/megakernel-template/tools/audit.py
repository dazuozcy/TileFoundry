#!/usr/bin/env python3
"""Expand local includes and compare executable C++ tokens with the R30 baseline.

This does not compile or prove correctness. It detects extraction omissions;
real compiler/ELF/device checks are separate. --baseline intentionally fails
once a new model edits the executable source.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def expand(path, stack=()):
    path = path.resolve()
    if path in stack:
        raise ValueError(f'include cycle: {path}')
    if not path.is_relative_to(ROOT):
        raise ValueError(f'include escaped standalone template: {path}')
    result = []
    for line in path.read_text().splitlines(keepends=True):
        match = re.fullmatch(r'\s*#include "([^"]+)"\s*', line)
        if match:
            candidates = [path.parent / match[1], ROOT / 'kernel' / match[1]]
            target = next((p for p in candidates if p.is_file()), None)
            if target is None:
                raise FileNotFoundError((path, match[1]))
            result.append(expand(target, (*stack, path)))
        elif line.strip() != '#pragma once':
            result.append(line)
    return ''.join(result)


def token_hash(source):
    # Keep string/character literals intact while removing comments. Token
    # boundaries are retained, so replacing "a b" with "ab" changes the digest.
    pattern = r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*[\s\S]*?\*/|[A-Za-z_]\w*|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|[^\s]'
    tokens = [m.group() for m in re.finditer(pattern, source)]
    tokens = [t for t in tokens if not t.startswith('//') and not t.startswith('/*')]
    return hashlib.sha256('\0'.join(tokens).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--emit', type=Path, help='write a standalone amalgamated C++ source')
    args = parser.parse_args()
    source = expand(ROOT / 'kernel/mega.cpp')
    digest = token_hash(source)
    provenance = json.loads((ROOT / 'validation/provenance.json').read_text())
    same = digest == provenance['upstream_executable_token_sha256']
    print(json.dumps({'executable_token_sha256': digest, 'matches_r30_baseline': same}, indent=2))
    if args.emit:
        args.emit.write_text(source)
    if args.baseline and not same:
        raise SystemExit('source differs from baseline (expected for a port; not an extraction PASS)')


if __name__ == '__main__':
    main()
