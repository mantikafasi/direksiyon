from pathlib import Path

import pytest

from llmsteer.steering import NeuronSteer, SpecError, parse_spec, split_leading_specs


def test_parse_signed_and_keyword_forms():
    assert parse_spec("L17N12250 +2.0") == [NeuronSteer(17, 12250, 2.0)]
    assert parse_spec("L17N12250 -0.75") == [NeuronSteer(17, 12250, -0.75)]
    assert parse_spec("L17N12250 down 0.5") == [NeuronSteer(17, 12250, -0.5)]
    assert parse_spec("L17N12250up1.25") == [NeuronSteer(17, 12250, 1.25)]

def test_parse_natural_neuron_label():
    assert parse_spec("Layer 0 · Neuron 1467 +2") == [NeuronSteer(0, 1467, 2.0)]
    steers, prompt = split_leading_specs("Layer 0 · Neuron 1467 +2 tell me who you are")
    assert steers == [NeuronSteer(0, 1467, 2.0)]
    assert prompt == "tell me who you are"


def test_parse_merges_repeated_coordinates():
    assert parse_spec("L1N2 +1.0, L1N2 -0.25, L3N4 +0.5") == [
        NeuronSteer(1, 2, 0.75),
        NeuronSteer(3, 4, 0.5),
    ]


def test_split_leading_specs_preserves_prompt():
    steers, prompt = split_leading_specs("L17N12250 +2.0, L8N1788 -0.75 explain this")
    assert steers == [NeuronSteer(17, 12250, 2.0), NeuronSteer(8, 1788, -0.75)]
    assert prompt == "explain this"

    steers, prompt = split_leading_specs("Let's discuss L17N12250 +2.0")
    assert steers == []
    assert prompt == "Let's discuss L17N12250 +2.0"


def test_bad_spec_is_explicit():
    with pytest.raises(SpecError, match="cannot read"):
        parse_spec("L17N12250")
    with pytest.raises(SpecError, match="strength 0"):
        NeuronSteer(1, 2, 0).coerce(32, 14336, 2.5)
    with pytest.raises(SpecError, match="out of range"):
        NeuronSteer(32, 2, 1).coerce(32, 14336, 2.5)


def test_format_is_human_readable():
    assert str(NeuronSteer(17, 12250, 2.0)) == "L17N12250 +2"
    assert str(NeuronSteer(17, 12250, -0.75)) == "L17N12250 -0.75"


def test_parse_bare_coordinate():
    from llmsteer.steering import parse_bare_coordinate

    assert parse_bare_coordinate("Layer 0 · Neuron 1467") == (0, 1467)
    assert parse_bare_coordinate(" L17N12250 ") == (17, 12250)
    assert parse_bare_coordinate("L17N12250 +2") is None
    assert parse_bare_coordinate("hello") is None


def test_nudge_adds_removes_and_avoids_float_dust():
    from llmsteer.steering import nudge

    steers = nudge([], 1, 2, 0.1)
    steers = nudge(steers, 1, 2, 0.2)
    assert steers == [NeuronSteer(1, 2, 0.3)]
    assert nudge(steers, 1, 2, -0.3) == []
    other = NeuronSteer(3, 4, 0.5)
    assert nudge([other], 1, 2, -0.8) == [other, NeuronSteer(1, 2, -0.8)]
