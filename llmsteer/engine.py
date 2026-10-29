"""Per-channel conversation and serialized GPU generation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from .model import SteeringModel
from .steering import NeuronSteer


@dataclass
class ChannelState:
    steers: list[NeuronSteer] = field(default_factory=list)
    messages: list[dict[str, str]] = field(default_factory=list)


class GenerationBusy(RuntimeError):
    pass


class ChatEngine:
    """Keep Discord responsive while allowing only one model generation at a time."""

    def __init__(self, model: SteeringModel, *, history_turns: int = 6, max_queue: int = 4):
        self.model = model
        self.history_turns = history_turns
        self.states: dict[int, ChannelState] = {}
        self.lock = asyncio.Lock()
        self.max_queue = max_queue
        self.waiting = 0

    def state(self, channel_id: int) -> ChannelState:
        return self.states.setdefault(channel_id, ChannelState())

    def set_steering(self, channel_id: int, steers: list[NeuronSteer], clear_history: bool = True) -> None:
        state = self.state(channel_id)
        state.steers = list(steers)
        if clear_history:
            state.messages.clear()

    def clear_steering(self, channel_id: int, clear_history: bool = True) -> None:
        state = self.state(channel_id)
        state.steers.clear()
        if clear_history:
            state.messages.clear()

    def reset(self, channel_id: int) -> None:
        self.states.pop(channel_id, None)

    async def ask(
        self,
        channel_id: int,
        prompt: str,
        *,
        steers: list[NeuronSteer] | None = None,
        max_new_tokens: int | None = None,
        greedy: bool = False,
        keep_history: bool = True,
    ) -> str:
        if self.waiting >= self.max_queue:
            raise GenerationBusy("the generation queue is full; try again in a moment")
        self.waiting += 1
        try:
            async with self.lock:
                state = self.state(channel_id)
                if steers is not None:
                    state.steers = list(steers)
                if keep_history:
                    state.messages = state.messages[-self.history_turns * 2 :]
                    messages = [*state.messages, {"role": "user", "content": prompt}]
                else:
                    messages = [{"role": "user", "content": prompt}]

                answer = await asyncio.to_thread(
                    self.model.generate,
                    messages,
                    state.steers,
                    max_new_tokens=max_new_tokens,
                    greedy=greedy,
                )
                if keep_history:
                    state.messages.extend(
                        [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
                    )
                    state.messages = state.messages[-self.history_turns * 2 :]
                return answer
        finally:
            self.waiting -= 1
