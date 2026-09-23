import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from affinity import (  # noqa: E402
    global_affinity_chain,
    global_affinity_single,
    token_span_for_word,
)

MODEL_NAME = "ai-forever/ruRoberta-large"


@pytest.fixture(scope="module")
def model_and_tok():
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    model.eval()
    return model, tok


def test_is_fast(model_and_tok):
    _, tok = model_and_tok
    assert tok.is_fast


def test_token_span_for_word():
    offsets = [(0, 0), (0, 3), (4, 7), (8, 13), (13, 14), (0, 0)]
    assert token_span_for_word(offsets, 8, 13) == [3]
    assert token_span_for_word(offsets, 0, 3) == [1]


def test_single_in_unit_range(model_and_tok):
    model, tok = model_and_tok
    text = "Бог его знает, где они сейчас."
    enc = tok(text, return_offsets_mapping=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping")[0].tolist()
    idxs = token_span_for_word(offsets, 8, 13)
    assert len(idxs) == 1
    p = global_affinity_single(
        model, tok, enc["input_ids"], enc["attention_mask"], idxs[0]
    )
    assert 0.0 <= p <= 1.0


def test_chain_k1_equals_single(model_and_tok):
    model, tok = model_and_tok
    text = "Бог его знает, где они сейчас."
    enc = tok(text, return_offsets_mapping=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping")[0].tolist()
    idxs = token_span_for_word(offsets, 8, 13)
    assert len(idxs) == 1
    p_single = global_affinity_single(
        model, tok, enc["input_ids"], enc["attention_mask"], idxs[0]
    )
    p_chain = global_affinity_chain(
        model, tok, enc["input_ids"], enc["attention_mask"], idxs
    )
    assert p_single == pytest.approx(p_chain, abs=1e-6)


def test_batched_scorer_matches_reference(model_and_tok):
    from score import Scorer

    model, tok = model_and_tok
    scorer = Scorer()
    text = "Достопримечательности Бог его знает где."
    targets = [
        {"char_start": 0, "char_end": 21, "text": "Достопримечательности"},
        {"char_start": 30, "char_end": 35, "text": "знает"},
    ]
    res = scorer.score_sentence(text, targets, [False, True])
    enc = tok(text, return_offsets_mapping=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping")[0].tolist()
    for t, r in zip(targets, res, strict=True):
        idxs = token_span_for_word(offsets, t["char_start"], t["char_end"])
        ref_chain = global_affinity_chain(
            model, tok, enc["input_ids"], enc["attention_mask"], idxs
        )
        assert r["p_chain"] == pytest.approx(ref_chain, rel=1e-4, abs=1e-7)
        if len(idxs) == 1:
            ref_single = global_affinity_single(
                model, tok, enc["input_ids"], enc["attention_mask"], idxs[0]
            )
            assert r["p_single"] == pytest.approx(ref_single, rel=1e-4, abs=1e-7)
            assert r["p_lemma"] >= r["p_single"]
    assert res[0]["multitoken"] and res[0]["p_single"] is None


def test_chain_multitoken_in_unit_range(model_and_tok):
    model, tok = model_and_tok
    text = "Достопримечательности были прекрасны."
    enc = tok(text, return_offsets_mapping=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping")[0].tolist()
    idxs = token_span_for_word(offsets, 0, 22)  # "Достопримечательности"
    assert len(idxs) > 1
    p_chain = global_affinity_chain(
        model, tok, enc["input_ids"], enc["attention_mask"], idxs
    )
    assert 0.0 <= p_chain <= 1.0
