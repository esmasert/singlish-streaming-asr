from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
from typing import Callable

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def remove_prefix(key: str, prefix: str) -> str:
    if key.startswith(prefix):
        return key[len(prefix):]
    return key


def transform_state_dict(
    state_dict: dict,
    transform: Callable[[str], str],
) -> dict:
    return {
        transform(key): value
        for key, value in state_dict.items()
    }


def choose_best_state_dict(
    checkpoint_state: dict,
    model_state: dict,
) -> tuple[str, dict]:
    """
    Try common Lightning/DDP wrapper prefixes and select the mapping
    with the largest number of exact key + tensor-shape matches.
    """

    candidates: dict[str, Callable[[str], str]] = {
        "direct": lambda key: key,
        "strip_module": lambda key: remove_prefix(key, "module."),
        "strip_model": lambda key: remove_prefix(key, "model."),
        "strip_module_model": lambda key: remove_prefix(
            remove_prefix(key, "module."),
            "model.",
        ),
        "strip_forward_module": lambda key: remove_prefix(
            key,
            "_forward_module.",
        ),
        "strip_orig_mod": lambda key: remove_prefix(
            key,
            "_orig_mod.",
        ),
    }

    best_name = ""
    best_state: dict = {}
    best_score = -1

    for name, transform in candidates.items():
        candidate = transform_state_dict(checkpoint_state, transform)

        score = 0
        for key, tensor in candidate.items():
            target = model_state.get(key)
            if target is not None and target.shape == tensor.shape:
                score += 1

        print(
            f"Candidate {name:>20}: "
            f"{score:,}/{len(model_state):,} exact tensor matches"
        )

        if score > best_score:
            best_name = name
            best_state = candidate
            best_score = score

    return best_name, best_state


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Load fine-tuned Nemotron weights into the pretrained "
            "Nemotron model and save a portable .nemo archive."
        )
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--base-model",
        default="nvidia/nemotron-speech-streaming-en-0.6b",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    if args.output.suffix != ".nemo":
        raise ValueError("Output filename must end with .nemo")

    import torch
    import nemo.collections.asr as nemo_asr

    print("=" * 72)
    print(f"Loading base model: {args.base_model}")
    print("=" * 72)

    # Restoring the real .nemo model first resolves tokenizer/config artifacts.
    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name=args.base_model,
        map_location="cpu",
    )

    print()
    print(f"Model class: {type(model).__name__}")
    print(f"Checkpoint: {args.checkpoint}")

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Expected a dictionary checkpoint, got {type(checkpoint)}"
        )

    checkpoint_state = checkpoint.get("state_dict")

    if checkpoint_state is None:
        # Supports a plain state_dict checkpoint as well.
        checkpoint_state = checkpoint

    if not isinstance(checkpoint_state, dict):
        raise TypeError(
            "Checkpoint does not contain a valid state_dict."
        )

    print(f"Checkpoint tensors: {len(checkpoint_state):,}")

    model_state = model.state_dict()
    print(f"Model tensors:      {len(model_state):,}")
    print()

    mapping_name, selected_state = choose_best_state_dict(
        checkpoint_state,
        model_state,
    )

    print()
    print(f"Selected key mapping: {mapping_name}")

    # First inspect mismatches without silently accepting them.
    result = model.load_state_dict(
        selected_state,
        strict=False,
    )

    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)

    print(f"Missing keys:    {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    if missing:
        print("\nFirst missing keys:")
        for key in missing[:30]:
            print(f"  MISSING: {key}")

    if unexpected:
        print("\nFirst unexpected keys:")
        for key in unexpected[:30]:
            print(f"  UNEXPECTED: {key}")

    # Do not create a misleading model archive if the architecture/keys differ.
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint did not match the base Nemotron model exactly. "
            "The .nemo file was not saved."
        )

    # Release the duplicate checkpoint dictionary before saving.
    del checkpoint
    del checkpoint_state
    del selected_state
    gc.collect()

    model.eval()
    model.to("cpu")

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(f"Saving portable model to: {args.output}")

    model.save_to(str(args.output))

    size_gb = args.output.stat().st_size / (1024 ** 3)

    print()
    print("=" * 72)
    print("EXPORT COMPLETED SUCCESSFULLY")
    print("=" * 72)
    print(f"Output: {args.output.resolve()}")
    print(f"Size:   {size_gb:.2f} GB")


if __name__ == "__main__":
    main()
