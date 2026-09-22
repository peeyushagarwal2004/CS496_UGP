"""CIFAR-10 reference models for the fault-injection study.

``ResNet8`` is the MLPerf Tiny image-classification model: three residual
stacks of two 3x3 convolutions each, on a 16/32/64 channel ladder, with a
stem convolution and a linear head -- eight weight layers, ~78K parameters.
It is small enough that its MX bit space can be swept close to exhaustively,
which is what the drafts want as ground truth.

Batch normalisation
-------------------
:func:`fold_batchnorm` rewrites each ``Conv2d -> BatchNorm2d`` pair into a
single convolution with the same function.  This matters for reliability, not
just speed: an *unfolded* BatchNorm re-normalises whatever the corrupted
convolution produced, partially masking the fault and understating damage.
Deployed MX accelerators fold BN, so campaigns should run on the folded model
-- otherwise the measured failure rates describe a network nobody ships.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ResNet8", "BasicBlock", "fold_batchnorm", "count_parameters",
           "weight_layer_names", "RepVGG", "RepVGGBlock", "RepVGG_A0",
           "reparameterize", "MODELS", "build_model", "to_deploy"]


class BasicBlock(nn.Module):
    """Two 3x3 convolutions with an identity or projection shortcut."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet8(nn.Module):
    """MLPerf Tiny ResNet8 for 32x32 inputs."""

    def __init__(self, num_classes: int = 10, width: int = 16):
        super().__init__()
        w = width
        self.conv1 = nn.Conv2d(3, w, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(w)
        self.layer1 = BasicBlock(w, w, stride=1)
        self.layer2 = BasicBlock(w, 2 * w, stride=2)
        self.layer3 = BasicBlock(2 * w, 4 * w, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4 * w, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        return self.fc(torch.flatten(self.pool(x), 1))


# ------------------------------------------------------------------ BN folding

def _fold_pair(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Absorb `bn` into `conv`, returning an equivalent bias-carrying conv."""
    w = conv.weight.detach().clone()
    b = (conv.bias.detach().clone() if conv.bias is not None
         else torch.zeros(w.shape[0], dtype=w.dtype))

    gamma = bn.weight.detach()
    beta = bn.bias.detach()
    mean = bn.running_mean.detach()
    var = bn.running_var.detach()
    scale = gamma / torch.sqrt(var + bn.eps)

    folded = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                       conv.stride, conv.padding, conv.dilation, conv.groups,
                       bias=True)
    with torch.no_grad():
        folded.weight.copy_(w * scale.reshape(-1, 1, 1, 1))
        folded.bias.copy_((b - mean) * scale + beta)
    return folded


def fold_batchnorm(model: nn.Module) -> nn.Module:
    """Return a copy of `model` with every Conv2d->BatchNorm2d pair fused.

    The result is functionally identical in ``eval()`` mode but has no
    BatchNorm left to re-normalise a corrupted convolution, which is the
    behaviour an MX accelerator actually exhibits.
    """
    model = copy.deepcopy(model).eval()

    # ResNet8's conv/bn pairs are named attributes, plus the shortcut Sequential
    for block in [model] + [m for m in model.modules() if isinstance(m, BasicBlock)]:
        for cname, bname in (("conv1", "bn1"), ("conv2", "bn2")):
            conv = getattr(block, cname, None)
            bn = getattr(block, bname, None)
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                setattr(block, cname, _fold_pair(conv, bn))
                setattr(block, bname, nn.Identity())

    for m in model.modules():
        if isinstance(m, nn.Sequential) and len(m) == 2 \
                and isinstance(m[0], nn.Conv2d) and isinstance(m[1], nn.BatchNorm2d):
            m[0] = _fold_pair(m[0], m[1])
            m[1] = nn.Identity()

    return model.eval()


# ---------------------------------------------------------------- inspection

def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad or not trainable_only)


def weight_layer_names(model: nn.Module) -> list[str]:
    """Names of the Conv2d/Linear layers a campaign can target."""
    return [n for n, m in model.named_modules()
            if isinstance(m, (nn.Conv2d, nn.Linear))]


# ------------------------------------------------------------------- RepVGG

class RepVGGBlock(nn.Module):
    """Training-time multi-branch block: 3x3 + 1x1 + optional identity, each BN'd.

    :meth:`fuse` collapses all branches into a single 3x3 convolution with
    bias, followed by the block's ReLU.  That matters for this study beyond speed: after fusion there is no
    branch redundancy left to absorb a corrupted weight, and the fused kernel
    has a markedly different value distribution from either branch alone --
    so the MX block statistics an accelerator sees are the *fused* ones.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.in_ch, self.out_ch, self.stride = in_ch, out_ch, stride
        self.conv3 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1, stride, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn_id = (nn.BatchNorm2d(out_ch)
                      if in_ch == out_ch and stride == 1 else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn3(self.conv3(x)) + self.bn1(self.conv1(x))
        if self.bn_id is not None:
            out = out + self.bn_id(x)
        return F.relu(out)

    # -- fusion ----------------------------------------------------------
    def _fuse_conv_bn(self, conv_w: torch.Tensor, bn: nn.BatchNorm2d):
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        return conv_w * scale.reshape(-1, 1, 1, 1), bn.bias - bn.running_mean * scale

    def _identity_kernel(self) -> torch.Tensor:
        k = torch.zeros(self.out_ch, self.in_ch, 1, 1)
        for i in range(self.in_ch):
            k[i, i, 0, 0] = 1.0
        return k

    def fuse(self) -> nn.Sequential:
        w, b = self._fuse_conv_bn(self.conv3.weight.detach(), self.bn3)
        w1, b1 = self._fuse_conv_bn(self.conv1.weight.detach(), self.bn1)
        w = w + F.pad(w1, [1, 1, 1, 1])          # 1x1 sits at the 3x3 centre
        b = b + b1
        if self.bn_id is not None:
            wi, bi = self._fuse_conv_bn(self._identity_kernel(), self.bn_id)
            w = w + F.pad(wi, [1, 1, 1, 1])
            b = b + bi

        fused = nn.Conv2d(self.in_ch, self.out_ch, 3, self.stride, 1, bias=True)
        with torch.no_grad():
            fused.weight.copy_(w)
            fused.bias.copy_(b)
        # the block's ReLU is part of the block, not of the branches -- it must
        # survive fusion or the deploy model loses every nonlinearity
        return nn.Sequential(fused, nn.ReLU())


class RepVGG(nn.Module):
    """RepVGG for 32x32 inputs. ``RepVGG_A0`` gives the A0 width/depth."""

    def __init__(self, blocks: list[int], widths: list[int],
                 num_classes: int = 10):
        super().__init__()
        self._init_done = False
        self.stage0 = RepVGGBlock(3, widths[0], stride=1)
        stages, in_ch = [], widths[0]
        # stage1 keeps 32x32; stages 2-4 halve, giving 32 -> 16 -> 8 -> 4
        for i, (n, w) in enumerate(zip(blocks, widths[1:])):
            layers = []
            for j in range(n):
                layers.append(RepVGGBlock(in_ch, w,
                                          stride=2 if (j == 0 and i > 0) else 1))
                in_ch = w
            stages.append(nn.Sequential(*layers))
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, num_classes)
        # a 22-layer plain net needs a sane init to train at all
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stages(self.stage0(x))
        return self.fc(torch.flatten(self.pool(x), 1))


def RepVGG_A0(num_classes: int = 10, a: float = 0.75, b: float = 2.5) -> RepVGG:
    """RepVGG-A0: blocks [1,2,4,14,1], base widths [64,64,128,256,512]."""
    widths = [min(64, int(64 * a)), int(64 * a), int(128 * a), int(256 * a),
              int(512 * b)]
    return RepVGG([1, 2, 4, 14, 1][1:], widths, num_classes)


def reparameterize(model: nn.Module) -> nn.Module:
    """Return a copy of `model` with every RepVGGBlock fused to one 3x3 conv."""
    model = copy.deepcopy(model).eval()

    def _walk(mod: nn.Module) -> None:
        for name, child in list(mod.named_children()):
            if isinstance(child, RepVGGBlock):
                setattr(mod, name, child.fuse())
            else:
                _walk(child)

    _walk(model)
    return model.eval()


# ------------------------------------------------------------------ registry

def _resnet8_width(w: int):
    """ResNet8 at a given base width -- the capacity axis of the width sweep."""
    def make(num_classes: int = 10) -> ResNet8:
        return ResNet8(num_classes=num_classes, width=w)
    make.__name__ = f"resnet8_w{w}"
    return make


from .vit import ViT, vit_small_cifar  # noqa: E402  (registry entry below)

MODELS = {"resnet8": ResNet8, "repvgg_a0": RepVGG_A0,
          "vit_small": vit_small_cifar}
# width variants share ResNet8's topology exactly, so a difference between them
# is attributable to capacity alone -- which is what F18 needs to isolate.
MODELS.update({f"resnet8_w{w}": _resnet8_width(w) for w in (8, 16, 24, 32, 48, 64)})


def build_model(name: str, **kw) -> nn.Module:
    """Construct a model by registry name."""
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; known: {sorted(MODELS)}")
    return MODELS[name](**kw)


def to_deploy(model: nn.Module) -> nn.Module:
    """Convert a trained model to the form an accelerator would actually run.

    RepVGG re-parameterises its multi-branch blocks into single convolutions;
    everything else gets its BatchNorms folded.  Both remove the structure
    that would otherwise mask a fault, so campaigns must run on this form.
    """
    if any(isinstance(m, RepVGGBlock) for m in model.modules()):
        return reparameterize(model)
    return fold_batchnorm(model)
