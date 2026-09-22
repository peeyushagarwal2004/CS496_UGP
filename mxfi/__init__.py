"""mxfi -- reliability-aware fault injection for microscaling (MX) DNNs.

The core (formats, codec, faults, sampling, stats) is pure numpy and has no
torch dependency, so the representation-level analysis runs anywhere.  The
PyTorch integration lives in ``mxfi.torch_mx`` and is imported explicitly.
"""

from .codec import MXTensor, headroom, quantize, quantize_dequantize
from .faults import (Fault, element_impact_table, fault_space, inject,
                     scale_impact_table)
from .formats import ELEMENT_FORMATS, ElementFormat, get_format
from .sampling import (FaultSite, Stratum, bit_population, neyman_allocation,
                       proportional_allocation, sample_stratified,
                       sample_uniform)
from .stats import (Estimate, StratumCount, binomial_rate, injection_reduction,
                    required_samples, stratified_rate, wilson_interval)

__all__ = [
    # formats + codec
    "ElementFormat", "ELEMENT_FORMATS", "get_format",
    "MXTensor", "quantize", "quantize_dequantize", "headroom",
    # faults
    "Fault", "inject", "fault_space",
    "element_impact_table", "scale_impact_table",
    # sampling
    "FaultSite", "Stratum", "bit_population", "sample_uniform",
    "sample_stratified", "proportional_allocation", "neyman_allocation",
    # stats
    "Estimate", "StratumCount", "wilson_interval", "binomial_rate",
    "stratified_rate", "required_samples", "injection_reduction",
]
__version__ = "0.1.0"
