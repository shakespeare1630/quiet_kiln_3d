"""Generate Three.js reconstructions through the subnet's witnessed Codex route."""

import argparse
import asyncio
from pathlib import Path
from urllib.parse import urlparse

from crown.funnel import generate

__all__ = ["generate"]


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--prompts", type=Path)
    inputs.add_argument("--image", action="append")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--parallel-prompts", type=int, default=4)
    parser.add_argument("--parallel-calls", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--wallet-name", default="miner")
    parser.add_argument("--wallet-hotkey", default="default")
    parser.add_argument("--wallet-path", default="~/.bittensor/wallets")
    parser.add_argument("--witness-config", type=Path, default=Path(__file__).with_name("witness.json"))
    args = parser.parse_args()
    sources = args.prompts.read_text().splitlines() if args.prompts else args.image
    mapping = {}
    for source in sources:
        source = source.strip()
        if not source:
            continue
        prompt_id = Path(urlparse(source).path).stem
        if prompt_id in mapping:
            parser.error(f"Duplicate prompt ID: {prompt_id}")
        mapping[prompt_id] = source
    if not mapping or not 0 <= args.seed < 2**64:
        parser.error("Provide images and a valid unsigned 64-bit round seed")
    from miner_runtime.batch import run

    await run(
        generate,
        mapping,
        args.seed,
        witness_config=args.witness_config,
        workers=args.parallel_prompts,
        parallel_calls=args.parallel_calls,
        attempts=args.attempts,
        wallet_name=args.wallet_name,
        wallet_hotkey=args.wallet_hotkey,
        wallet_path=args.wallet_path,
        api=False,
    )


if __name__ == "__main__":
    asyncio.run(main())
