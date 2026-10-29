"""Button panel for nudging one neuron's steering strength and testing it."""

from __future__ import annotations

import logging

import discord

from .config import Config
from .engine import ChatEngine, GenerationBusy
from .steering import NeuronSteer, SpecError, format_plan, nudge

log = logging.getLogger("llmsteer")

PROBE_PROMPT = "Who are you?"
PANEL_TIMEOUT_SECONDS = 600
MESSAGE_LIMIT = 2000


class SteerPanel(discord.ui.View):
    """+/- buttons that add to a neuron's channel-wide strength, plus a test generation."""

    def __init__(
        self,
        engine: ChatEngine,
        config: Config,
        *,
        owner_id: int,
        channel_id: int,
        layer: int,
        neuron: int,
    ):
        super().__init__(timeout=PANEL_TIMEOUT_SECONDS)
        self.engine = engine
        self.config = config
        self.owner_id = owner_id
        self.channel_id = channel_id
        self.layer = layer
        self.neuron = neuron
        self.message: discord.Message | None = None
        self.testing = False
        self.last_test: str | None = None

        for step in config.steer_steps:
            self._add_button(f"+{step:g}", discord.ButtonStyle.success, 0, self._on_step, step)
        for step in config.steer_steps:
            self._add_button(f"-{step:g}", discord.ButtonStyle.danger, 1, self._on_step, -step)
        self.test_button = self._add_button("Test", discord.ButtonStyle.primary, 2, self._on_test)
        self._add_button("Remove", discord.ButtonStyle.secondary, 2, self._on_remove)

    def _add_button(self, label, style, row, handler, *args) -> discord.ui.Button:
        button = discord.ui.Button(label=label, style=style, row=row)

        async def callback(interaction: discord.Interaction) -> None:
            await handler(interaction, *args)

        button.callback = callback
        self.add_item(button)
        return button

    @property
    def alpha(self) -> float:
        for steer in self.engine.state(self.channel_id).steers:
            if (steer.layer, steer.neuron) == (self.layer, self.neuron):
                return steer.alpha
        return 0.0

    def render(self) -> str:
        alpha = self.alpha
        current = f"{alpha:+g}" if alpha else "off"
        plan = format_plan(self.engine.state(self.channel_id).steers)
        text = (
            f"**Layer {self.layer} · Neuron {self.neuron}**: `{current}`\n"
            f"Channel steering: `{plan}`"
        )
        if self.testing:
            text += f"\n\n⏳ Generating _{PROBE_PROMPT}_ ..."
        if self.last_test:
            text += "\n\n" + self.last_test
        return text[:MESSAGE_LIMIT]

    def apply(self, delta: float | None) -> list[NeuronSteer]:
        """Add ``delta`` to this neuron, or remove it when ``delta`` is None."""
        current = self.engine.state(self.channel_id).steers
        step = -self.alpha if delta is None else delta
        steers = self.engine.model.validate(
            nudge(current, self.layer, self.neuron, step), self.config.alpha_max
        )
        self.engine.set_steering(
            self.channel_id, steers, clear_history=self.config.clear_history_on_steer
        )
        return steers

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id == self.owner_id:
            return True
        await interaction.response.send_message(
            "This panel belongs to someone else; run `.steer` to open your own.", ephemeral=True
        )
        return False

    async def _on_step(self, interaction: discord.Interaction, delta: float | None) -> None:
        try:
            self.apply(delta)
        except SpecError as exc:
            await interaction.response.send_message(f"Error: {exc}", ephemeral=True)
            return
        await interaction.response.edit_message(content=self.render(), view=self)

    async def _on_remove(self, interaction: discord.Interaction) -> None:
        await self._on_step(interaction, None)

    async def _on_test(self, interaction: discord.Interaction) -> None:
        plan = format_plan(self.engine.state(self.channel_id).steers)
        self.testing = True
        self.test_button.disabled = True
        await interaction.response.edit_message(content=self.render(), view=self)
        try:
            answer = await self.engine.ask(
                self.channel_id,
                PROBE_PROMPT,
                greedy=True,
                max_new_tokens=self.config.probe_max_new_tokens,
                keep_history=False,
            )
            result = answer or "(empty response)"
        except GenerationBusy as exc:
            result = f"Error: {exc}"
        except Exception:
            log.exception("panel test generation failed")
            result = "Generation failed; check the bot log."
        finally:
            self.testing = False
            self.test_button.disabled = False
        header = f"**Test** `{plan}`\n"
        self.last_test = None  # measure the panel without the previous result
        room = MESSAGE_LIMIT - len(self.render()) - len(header) - 2  # the answer shares the 2000-char message
        if len(result) > room:
            result = result[: max(room - 1, 0)].rstrip() + "…"
        self.last_test = header + result
        await interaction.edit_original_response(content=self.render(), view=self)

    async def on_timeout(self) -> None:
        for item in self.children:
            item.disabled = True
        if self.message is not None:
            try:
                await self.message.edit(view=self)
            except discord.HTTPException:
                pass
