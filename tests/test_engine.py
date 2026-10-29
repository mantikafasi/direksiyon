import asyncio

from llmsteer.engine import ChatEngine
from llmsteer.steering import NeuronSteer


class FakeModel:
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.seen = []

    def generate(self, messages, steers, **kwargs):
        self.seen.append([*messages])
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            import time

            time.sleep(0.02)
            return f"answer:{messages[-1]['content']}:{len(steers)}"
        finally:
            self.active -= 1


def test_generation_is_serialized_and_history_is_channel_local():
    async def scenario():
        model = FakeModel()
        engine = ChatEngine(model, history_turns=2, max_queue=4)
        steer = [NeuronSteer(17, 12250, 2.0)]
        first, second = await asyncio.gather(
            engine.ask(10, "one", steers=steer),
            engine.ask(11, "two"),
        )
        assert first == "answer:one:1"
        assert second == "answer:two:0"
        third = await engine.ask(10, "three")
        assert third == "answer:three:1"
        assert [message["content"] for message in model.seen[-1]] == ["one", first, "three"]
        assert [m["content"] for m in engine.state(10).messages] == ["one", first, "three", third]
        assert engine.state(11).messages == [
            {"role": "user", "content": "two"},
            {"role": "assistant", "content": second},
        ]

    asyncio.run(scenario())


def test_clear_steering_also_drops_history():
    async def scenario():
        engine = ChatEngine(FakeModel(), history_turns=2, max_queue=4)
        await engine.ask(10, "one", steers=[NeuronSteer(1, 2, 1.0)])
        engine.clear_steering(10)
        assert engine.state(10).steers == []
        assert engine.state(10).messages == []

    asyncio.run(scenario())
