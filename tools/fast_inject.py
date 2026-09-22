"""Make weight-fault injection O(K) instead of O(parameters), idempotently.

``MXModel.fault`` decoded the *whole* weight tensor through NumPy twice per
injection -- once to apply the fault, once to undo it.  That is invisible on a
slow CPU forward pass but dominates as soon as the forward pass is fast:
measured on an A100, ResNet8-w48 spent 31.8 ms per injection of which only a
few ms was the forward pass.

A fault only ever touches one value (element site) or one block of K values
(scale site), so the new path decodes just that span and writes it straight
into the existing weight tensor, leaving the rest untouched.  The original
whole-tensor path is kept as ``_fault_reference`` so the two can be compared.

Run once per checkout::

    python tools/fast_inject.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

OLD_FAULT = '''    @contextmanager
    def fault(self, site: FaultSite) -> Iterator[None]:
        """Apply one fault for the duration of the block, then undo it."""
        if site.tensor not in self.layers:
            raise KeyError(f"{site.tensor!r} is not a quantised layer")
        lay = self.layers[site.tensor]
        try:
            self._write(site.tensor, inject(lay.mx, site.fault))
            yield
        finally:
            self._write(site.tensor, lay.mx)
'''

NEW_FAULT = '''    def _affected(self, lay: "_Layer", f: Fault):
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
'''


def main() -> None:
    path = ROOT / "mxfi" / "torch_mx.py"
    src = path.read_text(encoding="utf-8")

    if "_fault_reference" in src:
        print("torch_mx.py: fast injection already present")
        return
    if src.count(OLD_FAULT) != 1:
        raise SystemExit("torch_mx.py: fault() does not match the expected text")

    src = src.replace(OLD_FAULT, NEW_FAULT)
    src = src.replace("from .codec import MXTensor, quantize",
                      "from .codec import MXTensor, quantize\n"
                      "from .formats import decode_e8m0")
    path.write_text(src, encoding="utf-8")
    print("torch_mx.py: fast injection applied")


if __name__ == "__main__":
    sys.exit(main())
