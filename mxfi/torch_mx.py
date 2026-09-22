"""PyTorch integration: MX-quantised models with injectable fault sites.

The model runs in float32 throughout; MX enters as *fake quantisation* --
values are encoded to MX and decoded straight back, so every weight and
activation carries exactly the value an MX accelerator would compute with.
Faults are injected in the MX domain (a bit of a code or of a shared scale)
and only then decoded, which is what makes a shared-scale flip perturb all K
elements of its block.

Blocking follows the reduction axis, the way MX hardware consumes a tensor:

* ``Linear``  weight ``(out, in)``            -> blocks along ``in``
* ``Conv2d``  weight ``(out, in, kh, kw)``    -> viewed as ``(out, in*kh*kw)``
  and blocked along that flattened reduction axis
* activations                                  -> channel axis for conv,
  feature axis for linear

Usage::

    mx = MXModel(net, MXConfig(fmt="e4m3", block_size=32))
    mx.quantize_weights()
    golden = evaluate(mx.module)
    with mx.fault(FaultSite("layer1.conv", Fault("scale", (0, 3), 7))):
        faulty = evaluate(mx.module)      # weights restored on exit
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np
import torch
import torch.nn as nn

from .codec import MXTensor, quantize
from .formats import decode_e8m0
from .faults import Fault, inject
from .sampling import FaultSite

__all__ = ["MXConfig", "MXModel", "QUANTISABLE"]

QUANTISABLE = (nn.Linear, nn.Conv2d)


@dataclass(frozen=True)
class MXConfig:
    """One cell of the study grid."""

    fmt: str = "e4m3"
    block_size: int = 32
    scale_mode: str = "ocp"
    quantise_weights: bool = True
    quantise_activations: bool = False
    skip_first: bool = False      # first/last layers are often kept at higher
    skip_last: bool = False       # precision in deployed quantised models

    def label(self) -> str:
        a = "wa" if self.quantise_activations else "w"
        return f"{self.fmt}-K{self.block_size}-{self.scale_mode}-{a}"


@dataclass
class _Layer:
    """Bookkeeping for one quantised layer."""

    name: str
    module: nn.Module
    original: torch.Tensor            # pristine float32 weight
    mx: MXTensor                      # MX form of the 2-D reduction view
    torch_shape: torch.Size           # shape to restore after decoding
    act_axis: int


class MXModel:
    """An ``nn.Module`` whose weights (and optionally activations) are MX.

    The wrapper never copies the model: it mutates weights in place and
    restores them, so evaluation code can keep using ``mx.module`` directly.
    """

    def __init__(self, module: nn.Module, config: MXConfig,
                 include: Sequence[str] | None = None):
        self.module = module
        self.config = config
        self.layers: dict[str, _Layer] = {}
        self._hooks: list = []
        self._act_fault: tuple[str, Fault] | None = None
        self._include = set(include) if include is not None else None
        self.module.eval()

    # ------------------------------------------------------------- selection
    def _targets(self) -> list[tuple[str, nn.Module]]:
        found = [(n, m) for n, m in self.module.named_modules()
                 if isinstance(m, QUANTISABLE)]
        if self._include is not None:
            found = [(n, m) for n, m in found if n in self._include]
        if self.config.skip_first and found:
            found = found[1:]
        if self.config.skip_last and found:
            found = found[:-1]
        return found

    @staticmethod
    def _reduction_view(w: torch.Tensor) -> torch.Tensor:
        """Flatten a weight to ``(out, reduction)`` -- how MX hardware reads it."""
        return w.reshape(w.shape[0], -1)

    # ------------------------------------------------------------- weights
    def quantize_weights(self) -> "MXModel":
        """Encode every target weight to MX and write the decoded value back."""
        if not self.config.quantise_weights:
            return self
        for name, mod in self._targets():
            w = mod.weight.detach().to(torch.float32)
            view = self._reduction_view(w).cpu().numpy()
            mx = quantize(view, self.config.fmt, self.config.block_size,
                          axis=-1, scale_mode=self.config.scale_mode)
            self.layers[name] = _Layer(
                name=name, module=mod, original=w.clone(), mx=mx,
                torch_shape=w.shape,
                act_axis=1 if isinstance(mod, nn.Conv2d) else -1,
            )
            self._write(name, mx)
        return self

    def _write(self, name: str, mx: MXTensor) -> None:
        lay = self.layers[name]
        dec = torch.from_numpy(mx.decode()).reshape(lay.torch_shape)
        with torch.no_grad():
            lay.module.weight.copy_(dec.to(lay.module.weight.dtype))

    def restore_weights(self) -> None:
        """Put every pristine float32 weight back."""
        with torch.no_grad():
            for lay in self.layers.values():
                lay.module.weight.copy_(lay.original.to(lay.module.weight.dtype))

    def weight_tensors(self) -> dict[str, MXTensor]:
        """The MX weight tensors -- the fault space a campaign samples from."""
        return {n: l.mx for n, l in self.layers.items()}

    # -------------------------------------------------------------- faults
    def _affected(self, lay: "_Layer", f: Fault):
        """Which weights a fault touches, and their new decoded values.

        Returns ``(row, first_col, values)`` in the ``(out, reduction)`` view,
        or ``None`` when the fault lands on a padding lane that carries no
        model value.  An element fault yields one value, a scale fault yields
        the whole block -- which is exactly the blast-radius asymmetry the
        study measures, here made explicit in the fast path.
        """
        mx = lay.mx
        K = mx.block_size
        red = int(np.prod(lay.torch_shape[1:]))
        table = mx.fmt.table()

        if f.site == "element":
            row, blk, lane = f.index
            col = blk * K + lane
            if col >= red:
                return None
            code = int(mx.codes[row, blk, lane]) ^ (1 << f.bit)
            val = table[code] * decode_e8m0(mx.scales[row, blk])
            return row, col, np.asarray([val], dtype=np.float32)

        row, blk = f.index
        col0 = blk * K
        n = min(K, red - col0)
        if n <= 0:
            return None
        byte = np.uint8(int(mx.scales[row, blk]) ^ (1 << f.bit))
        vals = table[mx.codes[row, blk, :n]] * decode_e8m0(byte)
        return row, col0, np.asarray(vals, dtype=np.float32)

    @contextmanager
    def fault(self, site: FaultSite) -> Iterator[None]:
        """Apply one fault for the duration of the block, then undo it."""
        if site.tensor not in self.layers:
            raise KeyError(f"{site.tensor!r} is not a quantised layer")
        lay = self.layers[site.tensor]
        patch = self._affected(lay, site.fault)
        if patch is None:                      # padding lane: nothing to change
            yield
            return

        row, col, vals = patch
        w = lay.module.weight
        flat = w.view(w.shape[0], -1)          # shares storage with the weight
        span = slice(col, col + len(vals))
        new = torch.as_tensor(vals, dtype=w.dtype, device=w.device)
        with torch.no_grad():
            old = flat[row, span].clone()
            flat[row, span] = new
        try:
            yield
        finally:
            with torch.no_grad():
                flat[row, span] = old

    @contextmanager
    def _fault_reference(self, site: FaultSite) -> Iterator[None]:
        """The original whole-tensor path, kept to verify the fast one."""
        lay = self.layers[site.tensor]
        try:
            self._write(site.tensor, inject(lay.mx, site.fault))
            yield
        finally:
            self._write(site.tensor, lay.mx)

    @contextmanager
    def activation_fault(self, site: FaultSite) -> Iterator[None]:
        """Apply one fault to a layer's *input* activations while active."""
        if not self.config.quantise_activations:
            raise RuntimeError("config.quantise_activations is False")
        prev = self._act_fault
        self._act_fault = (site.tensor, site.fault)
        try:
            yield
        finally:
            self._act_fault = prev

    # --------------------------------------------------------- activations
    def enable_activation_quantisation(self) -> "MXModel":
        """Install pre-forward hooks that push inputs through the MX codec."""
        if not self.config.quantise_activations:
            return self
        self.remove_hooks()
        for name, mod in self._targets():
            axis = 1 if isinstance(mod, nn.Conv2d) else -1
            self._hooks.append(
                mod.register_forward_pre_hook(self._make_hook(name, axis)))
        return self

    def _make_hook(self, name: str, axis: int):
        cfg = self.config

        def hook(_mod, inputs):
            x = inputs[0]
            arr = x.detach().to(torch.float32).cpu().numpy()
            mx = quantize(arr, cfg.fmt, cfg.block_size, axis=axis,
                          scale_mode=cfg.scale_mode)
            if self._act_fault is not None and self._act_fault[0] == name:
                mx = inject(mx, self._act_fault[1])
            out = torch.from_numpy(mx.decode()).to(device=x.device,
                                                   dtype=x.dtype)
            return (out,) + tuple(inputs[1:])

        return hook

    def activation_space(self, sample: torch.Tensor) -> dict[str, MXTensor]:
        """MX form of each layer's input for one batch.

        Activations are input-dependent, so a campaign must re-derive this per
        batch; the returned tensors define the fault space for that batch.
        """
        captured: dict[str, MXTensor] = {}
        handles = []
        cfg = self.config

        def make(name, axis):
            def hook(_m, inputs):
                arr = inputs[0].detach().to(torch.float32).cpu().numpy()
                captured[name] = quantize(arr, cfg.fmt, cfg.block_size,
                                          axis=axis, scale_mode=cfg.scale_mode)
            return hook

        for name, mod in self._targets():
            axis = 1 if isinstance(mod, nn.Conv2d) else -1
            handles.append(mod.register_forward_pre_hook(make(name, axis)))
        try:
            with torch.no_grad():
                self.module(sample)
        finally:
            for h in handles:
                h.remove()
        return captured

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __del__(self):
        try:
            self.remove_hooks()
        except Exception:
            pass
