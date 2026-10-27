"""Tests of the networks, training, ONNX export and the train pipeline (needs PyTorch)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from handpredict.ml.export import export_onnx, onnx_parity  # noqa: E402
from handpredict.ml.inference import OnnxNetwork, WindowPredictor  # noqa: E402
from handpredict.ml.models import EnsembleModule, build_network  # noqa: E402
from handpredict.ml.normalize import Normalizer  # noqa: E402
from handpredict.ml.train import TrainSettings, fit_network, seeded_network, smooth_euclidean_loss  # noqa: E402
from handpredict.ml.windows import WindowSet, WindowSpec, build_windows  # noqa: E402
from handpredict.synthetic import make_trial_data, sample_reach_spec  # noqa: E402
from handpredict.kinematics import make_filter  # noqa: E402
from handpredict.trial import make_history, process_trial  # noqa: E402

GRU = {"cell": "gru", "num_layers": 1, "head_hidden": 16, "hidden_size": 8, "dropout": 0.0}
LSTM = {**GRU, "cell": "lstm"}
MLP = {"hidden_sizes": [16, 16], "dropout": 0.0}


@pytest.fixture(scope="module")
def small_data(cfg):
    rng = np.random.default_rng(3)
    spec = WindowSpec.from_config(cfg, 60.0)
    trials = []
    for k, (speed, tgt) in enumerate([("fast", "T1"), ("slow", "T2"), ("medium", "T3"), ("fast", "T2")]):
        rs = sample_reach_spec(rng, cfg["synthetic"], tgt, speed, 0.5)
        trials.append(make_trial_data(rs, cfg, rng, f"t{k}", speed, tgt, noise_px=1.0))
    ws = WindowSet.concat([build_windows(t, spec) for t in trials], spec)
    return spec, trials, ws, Normalizer.fit(ws, spec)


def _settings(cfg, **kw):
    s = TrainSettings.from_config(cfg)
    return TrainSettings(**{**s.__dict__, **kw})


def test_smooth_euclidean_loss_by_hand():
    pred = torch.tensor([[[3.0, 4.0], [0.0, 0.0]]])
    target = torch.zeros(1, 2, 2)
    mask = torch.tensor([[1.0, 0.0]])  # second horizon has no ground truth
    loss = smooth_euclidean_loss(pred, target, mask, torch.ones(2), eps=0.0)
    assert float(loss) == pytest.approx(5.0)  # sqrt(3^2 + 4^2)


@pytest.mark.parametrize("kind,params", [("recurrent", GRU), ("recurrent", LSTM), ("mlp", MLP)])
def test_onnx_output_equals_pytorch(kind, params, small_data, tmp_path):
    spec, _, ws, norm = small_data
    torch.manual_seed(0)
    net = build_network(kind, spec, params).eval()
    path = export_onnx(net, spec, tmp_path / "m.onnx")
    seq_n, static_n = norm.inputs(ws.seq[:200], ws.static[:200])
    assert onnx_parity(net, path, seq_n, static_n, tolerance=1e-5) < 1e-5
    ensemble = EnsembleModule([net, build_network(kind, spec, params)]).eval()
    assert onnx_parity(ensemble, export_onnx(ensemble, spec, tmp_path / "e.onnx"), seq_n, static_n, 1e-5) < 1e-5


def test_training_reduces_loss_and_is_repeatable(cfg, small_data):
    spec, _, ws, norm = small_data
    settings = _settings(cfg, max_epochs=15, patience=50, batch_size=32)  # ~300 optimiser steps
    a = fit_network(seeded_network("recurrent", spec, GRU, 7), ws, ws, norm, settings, seed=7)
    assert a.history["train_loss"].iloc[-1] < 0.8 * a.history["train_loss"].iloc[0]
    torch.manual_seed(999)  # whatever happened before must not matter
    b = fit_network(seeded_network("recurrent", spec, GRU, 7), ws, ws, norm, settings, seed=7)
    for pa, pb in zip(a.model.state_dict().values(), b.model.state_dict().values()):
        assert torch.equal(pa, pb)  # same seed -> exactly the same network


def test_onnx_predictor_single_and_batch_paths_agree(cfg, small_data, tmp_path):
    spec, trials, _, norm = small_data
    net = build_network("recurrent", spec, GRU).eval()
    predictor = WindowPredictor("gru", OnnxNetwork(export_onnx(net, spec, tmp_path / "g.onnx")), norm, "cpu")
    trial = process_trial(trials[0], make_filter(cfg), cfg)
    ws = build_windows(trials[0], spec)
    batch = predictor.predict_windowset(ws)
    horizons = np.asarray(spec.horizons_s)
    for i in (0, 10, len(ws) - 1):
        j = int(ws.frame_index[i])
        single = predictor.predict(make_history(trial, j), horizons)
        np.testing.assert_allclose(single.positions, trials[0].xy[j] + batch[i], atol=1e-5)
    early = predictor.predict(make_history(trial, 3), horizons)  # not enough history yet
    assert early.fallback == "no_window" and np.isnan(early.positions).all()
    with pytest.raises(ValueError):
        predictor.predict(make_history(trial, 50), horizons[:3])
