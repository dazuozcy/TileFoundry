#!/usr/bin/env python3
"""Save short real-weight greedy token IDs for independent-process comparison."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--prompt', default='The capital of France is')
    parser.add_argument('--tokens', type=int, default=32)
    parser.add_argument('--module-root', type=Path, default=Path(__file__).resolve().parents[1],
                        help='optional independent control project directory containing model_run.py')
    args = parser.parse_args()
    if args.tokens < 1:
        parser.error('--tokens must be positive')
    if not (args.module_root / 'model_run.py').is_file():
        parser.error('--module-root must contain model_run.py')
    os.environ['MEGA_CHECKPOINT'] = args.checkpoint
    sys.path.insert(0, str(args.module_root.resolve()))
    import model_run as model
    # The historical control runner has a constant WDIR, rather than the env setting.
    model.WDIR = args.checkpoint
    decoder = model.Decoder()
    tokens, _, _ = decoder.generate(decoder.tok.encode(args.prompt), args.tokens)
    record = dict(prompt=args.prompt, requested_tokens=args.tokens,
                  tokens=tokens, text=decoder.tok.decode(tokens))
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
