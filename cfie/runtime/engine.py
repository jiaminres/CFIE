"""CFIE 引擎主循环（Phase 1）。"""

from __future__ import annotations

from typing import Any

from cfie.api.protocol import InferenceResult
from cfie.api.streamer import TokenStreamer
from cfie.config.schema import EngineConfig
from cfie.loader.base_loader import BaseModelLoader
from cfie.loader.hf_loader import HFModelLoader
from cfie.request.request import InferenceRequest
from cfie.request.session import SessionManager
from cfie.runtime.executor import Executor, ModelOutput
from cfie.runtime.input_builder import InputBuilder
from cfie.runtime.model_runner import ModelRunner
from cfie.runtime.output_processor import OutputProcessor
from cfie.scheduler.scheduler import FCFSScheduler
from cfie.utils.logging import get_logger

logger = get_logger(__name__)


class Engine:
    """最小可运行推理引擎。"""

    def __init__(
        self,
        config: EngineConfig,
        *,
        loader: BaseModelLoader | None = None,
        executor: Executor | None = None,
        scheduler: FCFSScheduler | None = None,
        streamer: TokenStreamer | None = None,
    ) -> None:
        self.config = config.validate()
        self._running = False
        self._step_count = 0

        self._requests: dict[str, InferenceRequest] = {}
        self._session_manager = SessionManager()

        self._loader = loader or HFModelLoader()
        self._executor = executor
        self._scheduler = scheduler or FCFSScheduler(
            max_num_seqs=self.config.scheduler.max_num_seqs)
        self._streamer = streamer or TokenStreamer()
        self._input_builder = InputBuilder()
        self._output_processor = OutputProcessor()
        logger.debug("engine initialized: model=%s max_num_seqs=%s",
                     self.config.model.model, self.config.scheduler.max_num_seqs)

    @property
    def running(self) -> bool:
        return self._running

    @property
    def step_count(self) -> int:
        return self._step_count

    def _ensure_executor(self) -> None:
        if self._executor is not None:
            return

        # 惰性初始化：仅在首个有效调度轮次才加载模型，降低空启动成本。
        self._loader.download_model(self.config.model, self.config.load)
        model = self._loader.load_model(self.config.model, self.config.load)

        if not hasattr(self._loader, "load_tokenizer"):
            raise RuntimeError("Loader must provide load_tokenizer in Phase 1")

        tokenizer = self._loader.load_tokenizer(self.config.model,
                                                self.config.load)
        device = getattr(self._loader, "device", "cpu")
        model_runner = ModelRunner(model=model, tokenizer=tokenizer, device=device)
        self._executor = Executor(model_runner)
        logger.info("executor initialized: model=%s device=%s",
                    self.config.model.model, device)

    def add_request(self, req: InferenceRequest) -> None:
        if req.request_id in self._requests:
            raise ValueError(f"duplicate request_id: {req.request_id}")
        # 引擎侧维护 request 索引，同时同步进入会话与调度队列。
        self._requests[req.request_id] = req
        self._session_manager.attach_request(req.session_id, req.request_id)
        self._scheduler.add_request(req)
        logger.info("request added: request_id=%s session_id=%s max_new_tokens=%s",
                    req.request_id, req.session_id, req.max_new_tokens)

    def abort(self, request_id: str) -> None:
        req = self._requests.get(request_id)
        if req is None:
            return

        # 中断需要同时更新请求状态、调度器和执行器内部缓存。
        req.mark_aborted()
        self._scheduler.finish_requests([request_id])
        if self._executor is not None:
            self._executor.abort_request(request_id)
        logger.info("request aborted: request_id=%s", request_id)

    def start(self) -> None:
        self._running = True
        logger.debug("engine running")

    def stop(self) -> None:
        self._running = False
        logger.debug("engine stopped")

    def _collect_finished_request_ids(self, results: list[InferenceResult],
                                      model_output: ModelOutput) -> list[str]:
        finished = {result.request_id for result in results if result.finished}
        # 双重保障：若执行器层标记 finished，则也纳入回收。
        for step in model_output.step_outputs:
            if step.finished:
                finished.add(step.request_id)
        return list(finished)

    def step(self) -> list[InferenceResult]:
        if not self._running:
            self.start()

        self._step_count += 1
        # 阶段 1 主循环：schedule -> build -> execute -> process -> recycle
        plan = self._scheduler.schedule()
        if plan.empty():
            logger.debug("step=%s no runnable requests", self._step_count)
            return []

        self._ensure_executor()
        assert self._executor is not None

        model_inputs = self._input_builder.build(plan)
        model_output = self._executor.execute_model(model_inputs)
        results = self._output_processor.make(plan, model_output)

        self._streamer.emit_many(results)
        self._scheduler.update_from_output(plan, model_output)

        finished_ids = self._collect_finished_request_ids(results, model_output)
        if finished_ids:
            self._scheduler.finish_requests(finished_ids)
            logger.info("requests finished: request_ids=%s", ",".join(finished_ids))

        return results

    def run(self, steps: int = 1) -> None:
        self.start()
        for _ in range(max(0, steps)):
            self.step()
        self.stop()

    def generate(self,
                 prompt: str,
                 *,
                 max_new_tokens: int = 64,
                 session_id: str = "default") -> str:
        request = InferenceRequest(prompt=prompt,
                                   max_new_tokens=max_new_tokens,
                                   session_id=session_id)
        self.add_request(request)

        while not request.is_terminal:
            self.step()

        return request.output_text

    def drain_stream(self, request_id: str) -> list[InferenceResult]:
        return self._streamer.drain(request_id)

    def add_request_from_prompt(self,
                                prompt: str,
                                *,
                                max_new_tokens: int,
                                session_id: str = "default") -> InferenceRequest:
        request = InferenceRequest(prompt=prompt,
                                   max_new_tokens=max_new_tokens,
                                   session_id=session_id)
        self.add_request(request)
        return request

    def add_request_raw(self, req: Any) -> None:
        if not isinstance(req, InferenceRequest):
            raise TypeError("req must be InferenceRequest")
        self.add_request(req)
