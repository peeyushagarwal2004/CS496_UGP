# Mirror of the GPU cluster's outputs

Everything the IITK A100 machine (`/data/rajeshr/peeyush/bitflip`) produced, copied
here verbatim on 2026-10-03 and checked file-by-file against the cluster's MD5 sums
(425 checkpoint and result files, all matching).

| path | contents |
|---|---|
| `checkpoints/` | every network trained on the cluster: the ten-seed ResNet8 width sweep (F29–F31), RepVGG-A0 and the ViT on CIFAR-10 (3 seeds each), and all CIFAR-100 models (E24) |
| `results/` | every campaign run on the cluster, including the ten-seed sweep, both exhaustive campaigns (E06), E18–E24 |
| `logs/` | run logs |
| `archive/` | superseded outputs kept for the record, e.g. `e04_buggy_codec/` (the activation campaigns before the NaN fix, F69) |
| `*.sh` | the cluster-side runners that produced them |

## Why this is a separate folder, not merged into `../checkpoints` and `../results`

Training is not bit-reproducible across machines. For the CIFAR-10 ResNet8 width
sweep seeds 0–4 and RepVGG-A0 seed 0, the laptop and the cluster each trained their
**own** networks under the **same file names**, and ran their own campaigns on them:
57 checkpoint/log files and 57 result files share a name between the two machines
but differ in content. Merging would silently pair one machine's network with the
other's campaigns -- the mistake recorded as F50 in `docs/findings.md`.

So:

* `../checkpoints/`, `../results/` -- the laptop's networks and campaigns;
* `cluster/checkpoints/`, `cluster/results/` -- the cluster's.

A campaign belongs with the checkpoint in the same tree. Two cross-machine
identities are known and verified by replay: the laptop's `resnet8_w16` is the
cluster's `resnet8_w16_s9`, and the e3m2/e4m3 exhaustive campaigns were recorded on
cluster train-seeds 0 and 9 respectively. To check any other pairing:

```bash
PYTHONPATH=. python -m tools.verify_e01 <campaign.csv> --model <m> --fmt <f> --train-seed <s>
PYTHONPATH=. python -m tools.verify_ground_truth <exhaustive.csv> --fmt <f> --train-seed <s>
```

To run anything against the cluster's networks, point the loader at this folder,
e.g. `load_trained(model, path=Path("cluster/checkpoints/<stem>.pt"))`, or copy the
checkpoint into `../checkpoints/` under a name that does not collide.

The CIFAR datasets are not duplicated here: the cluster's archives are byte-identical
to `../data/`.
