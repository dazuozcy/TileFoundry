"""Greedy decode of Qwen3-1.7B with ONE AscendC mega-kernel launch per token.

  python run.py --prompt "..." --max-new-tokens 2048
  python run.py --bench ctx
"""
from model_run import *          # noqa: F401,F403 -- same CLI
import model_run

if __name__ == "__main__":
    model_run.main()
