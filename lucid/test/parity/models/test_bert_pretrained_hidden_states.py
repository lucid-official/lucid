"""Published BERT weights must agree at embedding and every encoder boundary."""

import numpy as np
import pytest

import lucid
from lucid.models import create_model

pytestmark = [pytest.mark.parity, pytest.mark.slow]


def test_bert_tiny_pretrained_hidden_states(ref) -> None:
    transformers = pytest.importorskip("transformers")
    try:
        ours = create_model("bert_tiny", pretrained=True).eval()
        source = transformers.AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2").eval()
    except OSError as exc:
        pytest.skip(f"pretrained download unavailable: {exc}")
    # Include padding IDs: a trained embedding row must not be zeroed during
    # conversion. No attention mask is requested on either side.
    ids = np.array([[101, 2023, 2003, 1037, 3231, 102, 0, 0]], dtype=np.int64)
    with lucid.no_grad(), ref.no_grad():
        got = ours(lucid.tensor(ids), output_hidden_states=True)
        wanted = source(ref.from_numpy(ids), output_hidden_states=True)
    assert got.hidden_states is not None
    assert len(got.hidden_states) == len(wanted.hidden_states) == 3
    for index, (actual, expected) in enumerate(zip(got.hidden_states, wanted.hidden_states)):
        np.testing.assert_allclose(
            actual.numpy(),
            expected.numpy(),
            atol=1e-4,
            rtol=1e-4,
            err_msg=f"first differing BERT boundary: {index}",
        )
