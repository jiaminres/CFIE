"""Unit tests for predictor training behavior."""

from __future__ import annotations

from cfie_training.predictor import PredictorTraceDataset, PredictorTraceExample, PredictorTrainer
from cfie_training.profiles import build_profile_config


def test_predictor_training_skips_examples_below_min_insertion_layer() -> None:
    config = build_profile_config("qwen35-35b-a3b")
    config.predictor_trainer.min_insertion_layer_index = 1
    config.predictor_trainer.hard_target_loss_weight = 1.0
    config.predictor_trainer.router_distill_loss_weight = 0.0

    hidden_size = config.model_spec.hidden_size
    hidden0 = tuple(0.0 for _ in range(hidden_size))
    hidden1 = tuple(1.0 for _ in range(hidden_size))
    teacher_topk = (tuple(range(config.model_spec.num_experts_per_tok)),)

    dataset = PredictorTraceDataset(
        profile_name=config.profile_name,
        example_count=2,
        window_layers=config.predictor_routing.window_layers,
        candidate_experts_per_layer=config.predictor_routing.candidate_experts_per_layer,
        executed_experts_per_layer=config.predictor_routing.executed_experts_per_layer,
        examples=(
            PredictorTraceExample(
                example_index=0,
                step_index=0,
                token_index=0,
                insertion_layer_index=0,
                future_layer_indices=(1,),
                hidden_state=hidden0,
                future_teacher_topk_ids=teacher_topk,
            ),
            PredictorTraceExample(
                example_index=1,
                step_index=0,
                token_index=1,
                insertion_layer_index=1,
                future_layer_indices=(2,),
                hidden_state=hidden1,
                future_teacher_topk_ids=teacher_topk,
            ),
        ),
    )

    trainer = PredictorTrainer(config)
    model, run_trace, _ = trainer.fit_dataset(dataset, epochs=1)
    evaluation = trainer.evaluate_dataset(dataset, model=model)

    assert run_trace.example_count == 1
    assert evaluation.example_count == 1
