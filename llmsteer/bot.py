"""Prefix-command Discord bot for the steering runtime."""

from __future__ import annotations

import asyncio
import logging
import re

import discord
from discord.ext import commands

from .config import Config, load_config
from .engine import ChatEngine, GenerationBusy
from .model import SteeringModel
from .panel import SteerPanel
from .steering import (
    NeuronSteer,
    SpecError,
    format_plan,
    parse_bare_coordinate,
    parse_spec,
    split_leading_specs,
)

log = logging.getLogger("llmsteer")



async def _reply_chunks(message: discord.Message, text: str) -> None:
    text = text or "(empty response)"
    for index in range(0, len(text), 1900):
        await message.reply(text[index : index + 1900], mention_author=False)


class SteeringBot(commands.Bot):
    def __init__(self, config: Config):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix=(".", "!"), intents=intents)
        self.config = config
        self.steering_model: SteeringModel | None = None
        self.engine: ChatEngine | None = None
        async def ask_command(ctx, *, prompt: str):
            await self.prefix_ask(ctx, prompt=prompt)

        async def steer_command(ctx, *, spec: str):
            await self.prefix_steer(ctx, spec=spec)

        async def steer_clear_command(ctx):
            await self.prefix_steer_clear(ctx)

        async def steer_show_command(ctx):
            await self.prefix_steer_show(ctx)

        async def probe_command(ctx, *, spec: str = ""):
            await self.prefix_probe(ctx, spec=spec)

        async def reset_command(ctx):
            await self.prefix_reset(ctx)

        async def status_command(ctx):
            await self.prefix_status(ctx)

        self.add_command(commands.Command(ask_command, name="ask", help="Ask the model; optionally start with steering"))
        self.add_command(commands.Command(steer_command, name="steer", help="Set neuron steering"))
        self.add_command(commands.Command(steer_clear_command, name="steer_clear", help="Clear steering"))
        self.add_command(commands.Command(steer_show_command, name="steer_show", help="Show steering"))
        self.add_command(commands.Command(probe_command, name="probe", help="Run a greedy identity probe"))
        self.add_command(commands.Command(reset_command, name="reset", help="Clear history and steering"))
        self.add_command(commands.Command(status_command, name="status", help="Show bot status"))

    async def setup_hook(self) -> None:
        # Model loading is intentionally outside the Discord event loop.
        self.steering_model = await asyncio.to_thread(SteeringModel, self.config)
        self.engine = ChatEngine(
            self.steering_model,
            history_turns=self.config.history_turns,
            max_queue=self.config.max_queue,
        )
        # Remove previously registered slash commands. This bot uses stable prefix commands.
        self.tree.clear_commands(guild=None)
        await self.tree.sync()
        if self.config.guild_id:
            guild = discord.Object(id=self.config.guild_id)
            self.tree.clear_commands(guild=guild)
            await self.tree.sync(guild=guild)
            log.info("cleared stale slash commands from guild %s", self.config.guild_id)
        else:
            log.info("cleared global slash commands")

    async def on_ready(self) -> None:
        log.info("ready as %s; %s", self.user, self.steering_model.describe())
        log.info(self.steering_model.vram_summary())


    def _require_engine(self) -> ChatEngine:
        if self.engine is None or self.steering_model is None:
            raise RuntimeError("model is still loading")
        return self.engine

    def _resolve(self, text: str) -> list[NeuronSteer]:
        engine = self._require_engine()
        return engine.model.validate(parse_spec(text), self.config.alpha_max)

    async def _ask_context(
        self,
        ctx: commands.Context,
        prompt: str,
        *,
        steers: list[NeuronSteer] | None = None,
        greedy: bool = False,
        max_new_tokens: int | None = None,
        keep_history: bool = True,
    ) -> None:
        engine = self._require_engine()
        try:
            async with ctx.typing():
                answer = await engine.ask(
                    ctx.channel.id,
                    prompt,
                    steers=steers,
                    greedy=greedy,
                    max_new_tokens=max_new_tokens,
                    keep_history=keep_history,
                )
            await _reply_chunks(ctx.message, answer)
        except (SpecError, GenerationBusy, ValueError) as exc:
            await ctx.send(f"Error: {exc}")
        except Exception:
            log.exception("generation failed")
            await ctx.send("Generation failed; check the bot log.")

    async def prefix_ask(self, ctx: commands.Context, *, prompt: str) -> None:
        inline, remainder = split_leading_specs(prompt)
        await self._ask_context(ctx, remainder if inline else prompt, steers=inline or None)

    async def prefix_steer(self, ctx: commands.Context, *, spec: str) -> None:
        try:
            coordinate = parse_bare_coordinate(spec)
            if coordinate is not None:
                await self._open_panel(ctx, *coordinate)
                return
            steers = self._resolve(spec)
            engine = self._require_engine()
            engine.set_steering(
                ctx.channel.id,
                steers,
                clear_history=self.config.clear_history_on_steer,
            )
            warning = ""
            if any(abs(steer.alpha) > 1.25 for steer in steers):
                warning = " Warning: strengths above 1.25 are more likely to become incoherent."
            await ctx.send(f"Steering set: `{format_plan(steers)}`.{warning}")
        except (SpecError, ValueError) as exc:
            await ctx.send(f"Error: {exc}")

    async def _open_panel(self, ctx: commands.Context, layer: int, neuron: int) -> None:
        engine = self._require_engine()
        # Range check only; the strength here is a placeholder at the allowed maximum.
        engine.model.validate([NeuronSteer(layer, neuron, self.config.alpha_max)], self.config.alpha_max)
        panel = SteerPanel(
            engine,
            self.config,
            owner_id=ctx.author.id,
            channel_id=ctx.channel.id,
            layer=layer,
            neuron=neuron,
        )
        panel.message = await ctx.send(panel.render(), view=panel)

    async def prefix_steer_clear(self, ctx: commands.Context) -> None:
        self._require_engine().clear_steering(
            ctx.channel.id, clear_history=self.config.clear_history_on_steer
        )
        await ctx.send("Steering cleared.")

    async def prefix_steer_show(self, ctx: commands.Context) -> None:
        state = self._require_engine().state(ctx.channel.id)
        await ctx.send(f"Active steering: `{format_plan(state.steers)}`")

    async def prefix_reset(self, ctx: commands.Context) -> None:
        self._require_engine().reset(ctx.channel.id)
        await ctx.send("Channel state reset.")

    async def prefix_probe(self, ctx: commands.Context, *, spec: str = "") -> None:
        try:
            steers = self._resolve(spec) if spec.strip() else None
        except (SpecError, ValueError) as exc:
            await ctx.send(f"Error: {exc}")
            return
        await self._ask_context(
            ctx,
            "Who are you?",
            steers=steers,
            greedy=True,
            max_new_tokens=self.config.probe_max_new_tokens,
            keep_history=False,
        )

    async def prefix_status(self, ctx: commands.Context) -> None:
        engine = self._require_engine()
        state = engine.state(ctx.channel.id)
        await ctx.send(
            f"Model: `{engine.model.describe()}`\n"
            f"Steering: `{format_plan(state.steers)}`\n"
            f"Queue: `{engine.waiting}/{engine.max_queue}`\n"
            f"Injection: `{self.config.inject_mode}`; r_L: `{self.config.r_mode}`"
        )


    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return
        await self.process_commands(message)
        if self.user is None or self.user not in message.mentions:
            return
        prompt = re.sub(rf"<@!?{self.user.id}>", "", message.content or "").strip()
        if not prompt:
            await message.reply(
                "Mention me followed by a prompt. If the prompt is missing, enable Message Content Intent "
                "in the Discord Developer Portal.",
                mention_author=False,
            )
            return
        inline, remainder = split_leading_specs(prompt)
        if not remainder:
            await message.reply("Mention me followed by a prompt.", mention_author=False)
            return
        engine = self._require_engine()
        try:
            async with message.channel.typing():
                answer = await engine.ask(message.channel.id, remainder, steers=inline or None)
            await _reply_chunks(message, answer)
        except Exception:
            log.exception("mention generation failed")
            await message.reply("Generation failed; check the bot log.", mention_author=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config()
    SteeringBot(config).run(config.discord_token)


if __name__ == "__main__":
    main()
