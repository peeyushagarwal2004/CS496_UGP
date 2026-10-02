"""Tests for the model, MX quantisation wrapper and campaign driver.

Skipped entirely when torch is unavailable -- the numpy core is independent of
it, so the rest of the suite still runs on an interpreter without torch.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn                                          # noqa: E402
from torch.utils.data import TensorDataset                     # noqa: E402

from mxfi.campaign import Evaluator, run_campaign, summarise, weighted_rate  # noqa: E402
from mxfi.faults import Fault                                  # noqa: E402
from mxfi.models import (ResNet8, count_parameters, fold_batchnorm,           # noqa: E402
                         weight_layer_names)
from mxfi.sampling import (FaultSite, proportional_allocation,                # noqa: E402
                           sample_stratified, sample_uniform)
from mxfi.torch_mx import MXConfig, MXModel                     # noqa: E402


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return ResNet8().eval()


@pytest.fixture(scope="module")
def data():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(32, 3, 32, 32, generator=g)
    y = torch.randint(0, 10, (32,), generator=g)
    return TensorDataset(x, y)


# --------------------------------------------------------------------- model

def test_resnet8_shape_and_size(net):
    """MLPerf Tiny ResNet8: ~78K parameters, 10 injectable weight layers."""
    assert count_parameters(net) == 78042
    assert len(weight_layer_names(net)) == 10
    with torch.no_grad():
        assert net(torch.randn(4, 3, 32, 32)).shape == (4, 10)


def test_bn_folding_preserves_the_function(net, data):
    """Folded and unfolded models must agree to float32 roundoff."""
    x = data.tensors[0]
    with torch.no_grad():
        a, b = net(x), fold_batchnorm(net)(x)
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)


def test_bn_folding_removes_every_batchnorm(net):
    folded = fold_batchnorm(net)
    assert not any(isinstance(m, nn.BatchNorm2d) for m in folded.modules())
    assert len(weight_layer_names(folded)) == 10


def test_folding_does_not_touch_the_original(net):
    before = net.conv1.weight.detach().clone()
    fold_batchnorm(net)
    assert torch.equal(net.conv1.weight, before)


# ------------------------------------------------------------ MX quantisation

def test_quantisation_perturbs_but_does_not_break(net, data):
    """MX weights should shift logits slightly, not destroy the model."""
    folded = fold_batchnorm(net)
    x = data.tensors[0]
    with torch.no_grad():
        clean = folded(x)

    mx = MXModel(folded, MXConfig(fmt="e4m3", block_size=32)).quantize_weights()
    with torch.no_grad():
        quant = mx.module(x)

    assert torch.isfinite(quant).all()
    assert not torch.equal(clean, quant)
    assert (clean - quant).abs().max() < 0.5 * clean.abs().max()


def test_all_weight_layers_are_captured(net):
    mx = MXModel(fold_batchnorm(net), MXConfig()).quantize_weights()
    assert set(mx.weight_tensors()) == set(weight_layer_names(net))


def test_conv_weights_are_blocked_along_the_reduction_axis(net):
    """A conv weight is viewed as (out, in*kh*kw) the way MX hardware reads it."""
    mx = MXModel(fold_batchnorm(net), MXConfig(block_size=32)).quantize_weights()
    q = mx.weight_tensors()["layer1.conv1"]
    w = dict(net.named_modules())["layer1.conv1"].weight
    assert q.orig_shape == (w.shape[0], w.shape[1] * w.shape[2] * w.shape[3])


def test_restore_weights_is_exact(net, data):
    folded = fold_batchnorm(net)
    x = data.tensors[0]
    with torch.no_grad():
        clean = folded(x).clone()
    mx = MXModel(folded, MXConfig()).quantize_weights()
    mx.restore_weights()
    with torch.no_grad():
        assert torch.equal(folded(x), clean)


# ------------------------------------------------------------ fault injection

def test_fault_context_restores_the_quantised_weights(net, data):
    mx = MXModel(fold_batchnorm(net), MXConfig()).quantize_weights()
    x = data.tensors[0]
    with torch.no_grad():
        golden = mx.module(x).clone()
    with mx.fault(FaultSite("layer1.conv1", Fault("element", (0, 0, 0), 7))):
        pass
    with torch.no_grad():
        assert torch.equal(mx.module(x), golden)


@pytest.mark.parametrize("K", [8, 32])
def test_blast_radius_shows_up_in_the_weights(net, K):
    """Element flip changes one weight; scale flip changes K."""
    mx = MXModel(fold_batchnorm(net), MXConfig(block_size=K)).quantize_weights()
    name = "layer2.conv1"
    lay = mx.layers[name]
    clean = torch.from_numpy(lay.mx.decode()).reshape(lay.torch_shape)

    with mx.fault(FaultSite(name, Fault("element", (0, 0, 0), 6))):
        assert int((lay.module.weight != clean).sum()) == 1
    with mx.fault(FaultSite(name, Fault("scale", (0, 0), 1))):
        assert int((lay.module.weight != clean).sum()) == K


@pytest.mark.parametrize("site,index,mask", [
    ("element", (1, 0, 3), 0b0000_0001),
    ("element", (1, 0, 3), 0b1000_0100),
    ("scale", (2, 1), 0b0000_0010),
    ("scale", (2, 1), 0b0001_1000),
])
def test_word_fault_matches_the_whole_tensor_path(net, site, index, mask):
    """A multi-bit word fault equals XOR-ing the stored word and re-decoding."""
    mx = MXModel(fold_batchnorm(net), MXConfig()).quantize_weights()
    name = "layer2.conv1"
    lay = mx.layers[name]
    ref = lay.mx.copy()
    (ref.codes if site == "element" else ref.scales)[index] ^= np.uint8(mask)
    want = torch.from_numpy(ref.decode()).reshape(lay.torch_shape)
    clean = lay.module.weight.detach().clone()

    with mx.word_fault(name, site, index, mask):
        assert torch.equal(lay.module.weight, want)
    assert torch.equal(lay.module.weight, clean)


def test_single_bit_word_fault_equals_fault(net):
    mx = MXModel(fold_batchnorm(net), MXConfig()).quantize_weights()
    name, w = "layer1.conv1", mx.layers["layer1.conv1"].module.weight
    with mx.fault(FaultSite(name, Fault("scale", (0, 0), 3))):
        a = w.detach().clone()
    with mx.word_fault(name, "scale", (0, 0), 1 << 3):
        assert torch.equal(w, a)


def test_word_fault_rejects_out_of_range_masks(net):
    mx = MXModel(fold_batchnorm(net), MXConfig(fmt="e3m2")).quantize_weights()
    for mask in (0, 1 << 6):
        with pytest.raises(ValueError):
            with mx.word_fault("layer1.conv1", "element", (0, 0, 0), mask):
                pass


def test_scale_msb_is_catastrophic(net, data):
    """Bit 7 of a shared scale moves the block past the float32 range."""
    mx = MXModel(fold_batchnorm(net), MXConfig()).quantize_weights()
    with mx.fault(FaultSite("layer2.conv1", Fault("scale", (0, 0), 7))):
        with torch.no_grad():
            out = mx.module(data.tensors[0])
    assert not torch.isfinite(out).all()


def test_unknown_layer_is_rejected(net):
    mx = MXModel(fold_batchnorm(net), MXConfig()).quantize_weights()
    with pytest.raises(KeyError):
        with mx.fault(FaultSite("nope", Fault("element", (0, 0, 0), 0))):
            pass


# ------------------------------------------------------------------- campaign

def test_evaluator_is_deterministic(net, data):
    mx = MXModel(fold_batchnorm(net), MXConfig()).quantize_weights()
    ev = Evaluator(data)
    a, _ = ev.predict(mx.module)
    b, _ = ev.predict(mx.module)
    assert np.array_equal(a, b)


def test_golden_reference_matches_labels(net, data):
    mx = MXModel(fold_batchnorm(net), MXConfig()).quantize_weights()
    ev = Evaluator(data)
    g = ev.golden(mx.module)
    assert g.n == 32
    assert g.accuracy == pytest.approx((g.predictions == g.labels).mean())


def test_campaign_runs_and_tabulates(net, data):
    mx = MXModel(fold_batchnorm(net), MXConfig()).quantize_weights()
    ev = Evaluator(data)
    g = ev.golden(mx.module)

    rng = np.random.default_rng(0)
    faults = sample_uniform(mx.weight_tensors(), 25, rng)
    df = run_campaign(mx, faults, ev, g, progress=False)

    assert len(df) == 25
    for col in ("site", "bit", "sdc", "changed", "acc_drop", "log_severity",
                "blast_radius", "nonfinite"):
        assert col in df.columns
    assert df["changed"].between(0, g.n).all()
    assert set(df["site"]) <= {"element", "scale"}


def test_campaign_records_blast_radius_per_site(net, data):
    mx = MXModel(fold_batchnorm(net), MXConfig(block_size=32)).quantize_weights()
    ev = Evaluator(data)
    g = ev.golden(mx.module)
    faults = [FaultSite("layer2.conv1", Fault("element", (0, 0, 0), 6)),
              FaultSite("layer2.conv1", Fault("scale", (0, 0), 1))]
    df = run_campaign(mx, faults, ev, g, progress=False)
    assert df.loc[df.site == "element", "blast_radius"].iloc[0] == 1
    assert df.loc[df.site == "scale", "blast_radius"].iloc[0] == 32


def test_campaign_is_side_effect_free(net, data):
    """The model is byte-identical after a campaign as before it."""
    mx = MXModel(fold_batchnorm(net), MXConfig()).quantize_weights()
    ev = Evaluator(data)
    g = ev.golden(mx.module)
    before = {n: l.module.weight.detach().clone() for n, l in mx.layers.items()}

    rng = np.random.default_rng(1)
    run_campaign(mx, sample_uniform(mx.weight_tensors(), 10, rng), ev, g,
                 progress=False)

    for n, l in mx.layers.items():
        assert torch.equal(l.module.weight, before[n])


def test_summarise_reports_intervals(net, data):
    mx = MXModel(fold_batchnorm(net), MXConfig()).quantize_weights()
    ev = Evaluator(data)
    g = ev.golden(mx.module)
    rng = np.random.default_rng(2)
    df = run_campaign(mx, sample_uniform(mx.weight_tensors(), 30, rng), ev, g,
                      progress=False)
    s = summarise(df, by=("site",))
    assert set(s.columns) >= {"site", "n", "sdc_rate", "ci_lo", "ci_hi"}
    assert (s["ci_lo"] <= s["sdc_rate"]).all()
    assert (s["sdc_rate"] <= s["ci_hi"]).all()
    assert s["n"].sum() == 30


def test_stratified_campaign_carries_weights(net, data):
    mx = MXModel(fold_batchnorm(net), MXConfig()).quantize_weights()
    ev = Evaluator(data)
    g = ev.golden(mx.module)
    rng = np.random.default_rng(3)
    alloc = proportional_allocation(mx.weight_tensors(), 20)
    faults, w = sample_stratified(mx.weight_tensors(), alloc, rng)
    df = run_campaign(mx, faults, ev, g, weights=w, progress=False)
    assert df["weight"].notna().all()
    est = weighted_rate(df)
    assert 0.0 <= est.rate <= 1.0


# --------------------------------------------------------------------- RepVGG

@pytest.fixture(scope="module")
def repvgg():
    """Small RepVGG with populated BN statistics (fusion is a no-op otherwise)."""
    from mxfi.models import RepVGG
    torch.manual_seed(1)
    m = RepVGG([1, 1, 1, 1], [8, 8, 16, 16, 32]).train()
    for _ in range(20):
        m(torch.randn(16, 3, 32, 32))
    return m.eval()


def test_repvgg_a0_geometry():
    """A0 = blocks [1,2,4,14,1], widths from a=0.75 / b=2.5."""
    from mxfi.models import RepVGG_A0, reparameterize
    m = RepVGG_A0()
    fused = reparameterize(m)
    assert count_parameters(fused) < count_parameters(m)   # BN + branches absorbed
    assert len(weight_layer_names(fused)) == 23            # 22 convs + fc
    with torch.no_grad():
        assert fused(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_reparameterisation_is_exact(repvgg):
    """Fused RepVGG must match the multi-branch model to float32 roundoff."""
    from mxfi.models import reparameterize
    x = torch.randn(8, 3, 32, 32)
    with torch.no_grad():
        a, b = repvgg(x), reparameterize(repvgg)(x)
    assert (a - b).abs().max() / a.abs().max() < 1e-5


def test_fusion_preserves_the_relu(repvgg):
    """Regression: `fuse` once returned a bare conv, dropping every ReLU.

    Without the nonlinearity the deploy model is a stack of linear maps, so
    it stays finite and plausible-looking while being a different function.
    """
    from mxfi.models import reparameterize
    fused = reparameterize(repvgg)
    assert any(isinstance(m, nn.ReLU) for m in fused.modules())
    # a linear-only stack would pass a negation straight through
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        assert not torch.allclose(fused(x), -fused(-x), atol=1e-3)


def test_repvgg_has_no_branches_or_bn_after_fusion(repvgg):
    from mxfi.models import RepVGGBlock, reparameterize
    fused = reparameterize(repvgg)
    assert not any(isinstance(m, (RepVGGBlock, nn.BatchNorm2d)) for m in fused.modules())


def test_repvgg_quantises_and_injects(repvgg):
    """The MX pipeline works unchanged on a fused RepVGG."""
    from mxfi.models import reparameterize
    fused = reparameterize(repvgg)
    mx = MXModel(fused, MXConfig("e4m3", 32)).quantize_weights()
    assert set(mx.weight_tensors()) == set(weight_layer_names(fused))

    name = weight_layer_names(fused)[3]
    lay = mx.layers[name]
    clean = torch.from_numpy(lay.mx.decode()).reshape(lay.torch_shape)
    with mx.fault(FaultSite(name, Fault("scale", (0, 0), 1))):
        assert int((lay.module.weight != clean).sum()) == 32
    with mx.fault(FaultSite(name, Fault("element", (0, 0, 0), 6))):
        assert int((lay.module.weight != clean).sum()) == 1


def test_checkpoint_names_are_seed_aware():
    """Seeded retrains must not overwrite seed 0, whose name predates seeds."""
    from mxfi.train import checkpoint_stem
    assert checkpoint_stem("resnet8_w8") == "resnet8_w8_cifar10"
    assert checkpoint_stem("resnet8_w8", 0) == "resnet8_w8_cifar10"
    assert checkpoint_stem("resnet8_w8", 3) == "resnet8_w8_s3_cifar10"
    assert len({checkpoint_stem("resnet8_w8", s) for s in range(5)}) == 5


def test_cifar100_models_are_named_and_sized_apart():
    """A `_c100` name selects CIFAR-100, a 100-class head, and its own checkpoint."""
    from mxfi.data import dataset_of
    from mxfi.models import build_model
    from mxfi.train import checkpoint_stem
    assert dataset_of("vit_small") == "cifar10"
    assert dataset_of("vit_small_c100") == "cifar100"
    assert checkpoint_stem("vit_small_c100") == "vit_small_cifar100"
    assert checkpoint_stem("vit_small_c100", 2) == "vit_small_s2_cifar100"
    assert checkpoint_stem("vit_small", 2) == "vit_small_s2_cifar10"
    for name in ("resnet8_c100", "repvgg_a0_c100", "vit_small_c100"):
        with torch.no_grad():
            assert build_model(name).eval()(torch.zeros(2, 3, 32, 32)).shape == (2, 100)
    assert build_model("resnet8")(torch.zeros(1, 3, 32, 32)).shape == (1, 10)


# ------------------------------------------------------------------- ViT

def test_vit_geometry_and_layers():
    """Small CIFAR ViT: every weight matrix is a Linear the quantiser can see."""
    from mxfi.models import build_model
    m = build_model("vit_small").eval()
    assert count_parameters(m) == 2693578
    names = weight_layer_names(m)
    assert len(names) == 26                      # patch embed + 6*(qkv,proj,fc1,fc2) + head
    assert "patch_embed.proj" in names and "head" in names
    with torch.no_grad():
        assert m(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_vit_deploy_is_identity():
    """A ViT has no BatchNorm and no branches, so deploy must not alter it."""
    from mxfi.models import build_model, to_deploy
    torch.manual_seed(0)
    m = build_model("vit_small").eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(m(x), to_deploy(m)(x))


def test_vit_quantises_and_injects():
    """Blast radius holds on Linear weights blocked along the model dimension."""
    from mxfi.models import build_model, to_deploy
    m = to_deploy(build_model("vit_small").eval())
    mx = MXModel(m, MXConfig("e4m3", 32)).quantize_weights()

    # LayerNorm must not be quantised -- only Linear/Conv2d are fault sites
    assert not any("norm" in n for n in mx.weight_tensors())

    name = "blocks.0.attn.qkv"
    lay = mx.layers[name]
    clean = torch.from_numpy(lay.mx.decode()).reshape(lay.torch_shape)
    with mx.fault(FaultSite(name, Fault("element", (0, 0, 0), 7))):
        assert int((lay.module.weight != clean).sum()) == 1
    with mx.fault(FaultSite(name, Fault("scale", (0, 0), 1))):
        assert int((lay.module.weight != clean).sum()) == 32
