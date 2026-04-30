"""Predictor-only CLI for CFIE training helpers."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Sequence

import torch

from cfie_training.config import TrainingProjectConfig
from cfie_training.predictor import PredictorTrainer
from cfie_training.predictor.trainer import (
    DEFAULT_TRACE_FLUSH_EVERY_STEPS,
    PredictorTraceBuildProgress,
)
from cfie_training.profiles import (
    DEFAULT_TRAINING_PROFILE,
    SUPPORTED_TRAINING_PROFILES,
    build_profile_config,
)

LOGGER = logging.getLogger("cfie_training.cli")


class _PredictorTraceProgressBar:
    def __init__(self, *, total_steps: int, description: str) -> None:
        self._bar = None
        if total_steps < 1 or not sys.stderr.isatty():
            return
        try:
            from tqdm.auto import tqdm
        except Exception:
            return
        self._bar = tqdm(
            total=total_steps,
            desc=description,
            unit="step",
            dynamic_ncols=True,
            file=sys.stderr,
        )

    def update(self, progress: PredictorTraceBuildProgress) -> None:
        if self._bar is None:
            return
        delta = progress.completed_steps - int(self._bar.n)
        if delta > 0:
            self._bar.update(delta)
        postfix = {
            "examples": progress.example_count,
            "last_examples": progress.last_step_examples,
            "last_tokens": progress.last_step_tokens,
        }
        if progress.persisted_steps > 0:
            postfix["flushed"] = progress.persisted_steps
        self._bar.set_postfix(postfix, refresh=False)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


def _configure_cli_logging() -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _emit_stdout(text: str) -> None:
    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")


def _emit_stdout_path(path: Path) -> None:
    with path.open("r", encoding="utf-8") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), ""):
            if not chunk:
                break
            sys.stdout.write(chunk)


def _default_predictor_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_config(args: argparse.Namespace) -> TrainingProjectConfig:
    if args.config is not None:
        return TrainingProjectConfig.from_json_file(args.config).validate()
    return build_profile_config(args.profile).validate()


def _add_shared_profile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, help="Optional JSON config file.")
    parser.add_argument(
        "--profile",
        choices=SUPPORTED_TRAINING_PROFILES,
        default=DEFAULT_TRAINING_PROFILE,
        help="Training profile to use when --config is not provided.",
    )


def _add_shared_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Text or JSONL dataset file used to build tokenized trace batches.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="Optional tokenizer path. Defaults to the training profile model path.",
    )
    parser.add_argument(
        "--dataset-format",
        choices=("auto", "text", "jsonl"),
        default="auto",
        help="Dataset file format. Defaults to auto-detect from extension.",
    )
    parser.add_argument(
        "--dataset-text-key",
        default="text",
        help="JSONL text field to read when --dataset-format jsonl is used.",
    )


def _trace_dataset_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "steps": args.steps,
        "examples_per_step": args.examples_per_step,
        "samples": args.samples,
        "tokens_per_sample": args.tokens_per_sample,
        "dataset_path": None if args.dataset is None else str(args.dataset),
        "tokenizer_path": None if args.tokenizer is None else str(args.tokenizer),
        "dataset_format": args.dataset_format,
        "dataset_text_key": args.dataset_text_key,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cfie-train",
        description="Predictor trace and training helpers for CFIE.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    predictor_trace_parser = subparsers.add_parser(
        "predictor-trace",
        help="Build teacher-trace samples for predictor training.",
    )
    _add_shared_profile_args(predictor_trace_parser)
    predictor_trace_parser.add_argument(
        "--steps",
        type=int,
        default=2,
        help="Number of capture steps used to build trace examples.",
    )
    predictor_trace_parser.add_argument(
        "--examples-per-step",
        type=int,
        default=None,
        help="Optional override for the number of trace examples per step.",
    )
    predictor_trace_parser.add_argument(
        "--samples",
        type=int,
        default=2,
        help="Base samples per trace-capture step.",
    )
    predictor_trace_parser.add_argument(
        "--tokens-per-sample",
        type=int,
        default=256,
        help="Tokens per sample used during trace capture.",
    )
    _add_shared_dataset_args(predictor_trace_parser)
    predictor_trace_parser.add_argument(
        "--output",
        type=Path,
        help="Optional path that receives the captured predictor trace dataset as JSON.",
    )
    predictor_trace_parser.add_argument(
        "--flush-every-steps",
        type=int,
        default=DEFAULT_TRACE_FLUSH_EVERY_STEPS,
        help=(
            "When --output is set, flush in-progress trace data to disk every N "
            "capture steps. Use 0 to disable periodic flushes."
        ),
    )
    predictor_trace_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the predictor trace dataset as JSON.",
    )

    predictor_train_parser = subparsers.add_parser(
        "predictor-train",
        help="Train the predictor model on teacher traces.",
    )
    _add_shared_profile_args(predictor_train_parser)
    predictor_train_parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="Number of capture steps used when trace data is built inline.",
    )
    predictor_train_parser.add_argument(
        "--examples-per-step",
        type=int,
        default=None,
        help="Optional override for the number of trace examples per step.",
    )
    predictor_train_parser.add_argument(
        "--samples",
        type=int,
        default=2,
        help="Base samples per trace-capture step.",
    )
    predictor_train_parser.add_argument(
        "--tokens-per-sample",
        type=int,
        default=256,
        help="Tokens per sample used during trace capture.",
    )
    _add_shared_dataset_args(predictor_train_parser)
    predictor_train_parser.add_argument(
        "--trace-input",
        type=Path,
        help="Optional predictor trace dataset JSON used instead of recapturing traces.",
    )
    checkpoint_group = predictor_train_parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--init-from-checkpoint",
        type=Path,
        help="Optional predictor checkpoint used only to initialize model weights.",
    )
    checkpoint_group.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="Optional predictor checkpoint used to resume model and optimizer state.",
    )
    predictor_train_parser.add_argument(
        "--checkpoint-output",
        type=Path,
        help="Optional output path that receives the trained predictor checkpoint.",
    )
    predictor_train_parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for training epochs.",
    )
    predictor_train_parser.add_argument(
        "--log-every-steps",
        type=int,
        default=10,
        help="Emit one training log line every N optimizer steps. Use 0 to disable step logs.",
    )
    predictor_train_parser.add_argument(
        "--checkpoint-every-epochs",
        type=int,
        default=1,
        help="Persist a predictor checkpoint every N epochs when a checkpoint path is available.",
    )
    predictor_train_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the predictor training run trace as JSON.",
    )
    return parser


def _run_predictor_trace(
    args: argparse.Namespace,
    *,
    config: TrainingProjectConfig,
) -> int:
    trainer = PredictorTrainer(config)
    logger = LOGGER.getChild("predictor.trace")
    progress_bar = _PredictorTraceProgressBar(
        total_steps=args.steps,
        description="predictor-trace",
    )
    try:
        if args.output is not None:
            result = trainer.build_trace_dataset_to_json_file(
                output_path=args.output,
                flush_every_steps=args.flush_every_steps,
                progress_callback=progress_bar.update,
                **_trace_dataset_kwargs(args),
            )
            if args.json:
                _emit_stdout_path(args.output)
            else:
                logger.info(
                    "built predictor trace: profile=%s examples=%d window=%d candidate=%d executed=%d path=%s",
                    result.profile_name,
                    result.example_count,
                    result.window_layers,
                    result.candidate_experts_per_layer,
                    result.executed_experts_per_layer,
                    args.output,
                )
            return 0

        dataset = trainer.build_trace_dataset(
            progress_callback=progress_bar.update,
            **_trace_dataset_kwargs(args),
        )
    finally:
        progress_bar.close()

    if args.json:
        _emit_stdout(dataset.to_json())
    else:
        logger.info(
            "built predictor trace: profile=%s examples=%d window=%d candidate=%d executed=%d",
            dataset.profile_name,
            dataset.example_count,
            dataset.window_layers,
            dataset.candidate_experts_per_layer,
            dataset.executed_experts_per_layer,
        )
    return 0


def _run_predictor_train(
    args: argparse.Namespace,
    *,
    config: TrainingProjectConfig,
) -> int:
    trainer = PredictorTrainer(config)
    logger = LOGGER.getChild("predictor.train")

    model = None
    optimizer_state_dict = None
    initial_run_trace = None

    if args.init_from_checkpoint is not None:
        model, _ = trainer.load_checkpoint(args.init_from_checkpoint)
    if args.resume_checkpoint is not None:
        model, _, initial_run_trace, optimizer_state_dict = (
            trainer.load_training_checkpoint(args.resume_checkpoint)
        )

    checkpoint_save_path = args.checkpoint_output
    if checkpoint_save_path is None and args.resume_checkpoint is not None:
        checkpoint_save_path = args.resume_checkpoint

    if args.trace_input is not None:
        _, run_trace, _ = trainer.fit_trace_file(
            path=args.trace_input,
            epochs=args.epochs,
            model=model,
            optimizer_state_dict=optimizer_state_dict,
            initial_run_trace=initial_run_trace,
            logger=logger,
            log_every_steps=args.log_every_steps,
            checkpoint_output_path=checkpoint_save_path,
            checkpoint_every_epochs=args.checkpoint_every_epochs,
            device=_default_predictor_device(),
        )
    else:
        progress_bar = _PredictorTraceProgressBar(
            total_steps=args.steps,
            description="predictor-train trace",
        )
        try:
            dataset = trainer.build_trace_dataset(
                progress_callback=progress_bar.update,
                **_trace_dataset_kwargs(args),
            )
        finally:
            progress_bar.close()
        _, run_trace, _ = trainer.fit_dataset(
            dataset,
            epochs=args.epochs,
            model=model,
            optimizer_state_dict=optimizer_state_dict,
            initial_run_trace=initial_run_trace,
            logger=logger,
            log_every_steps=args.log_every_steps,
            checkpoint_output_path=checkpoint_save_path,
            checkpoint_every_epochs=args.checkpoint_every_epochs,
        )

    if args.json:
        _emit_stdout(run_trace.to_json())
    else:
        logger.info(
            "trained predictor: profile=%s examples=%d epochs=%d final_loss=%.6f recall@candidate=%.4f checkpoint=%s",
            run_trace.profile_name,
            run_trace.example_count,
            run_trace.epochs,
            run_trace.final_mean_loss,
            run_trace.final_recall_at_candidate_budget,
            checkpoint_save_path,
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    _configure_cli_logging()

    if args.command == "predictor-trace":
        if args.dataset is None:
            parser.error("predictor-trace requires --dataset")
        if args.flush_every_steps < 0:
            parser.error("predictor-trace --flush-every-steps must be >= 0")
    elif args.command == "predictor-train":
        if args.trace_input is None and args.dataset is None:
            parser.error("predictor-train requires --trace-input or --dataset")
        if args.log_every_steps < 0:
            parser.error("predictor-train --log-every-steps must be >= 0")
        if args.checkpoint_every_epochs < 1:
            parser.error("predictor-train --checkpoint-every-epochs must be >= 1")

    config = _load_config(args)

    if args.command == "predictor-trace":
        return _run_predictor_trace(args, config=config)
    if args.command == "predictor-train":
        return _run_predictor_train(args, config=config)

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
