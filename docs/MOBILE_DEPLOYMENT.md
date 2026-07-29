# Mobile Deployment

Mobile deployment is a research target, not a verified release claim.

## Benchmark record

| Field | Value |
|---|---|
| Experiment and checkpoint | TBD |
| Export format and opset | TBD |
| Precision / quantization | Pending verification |
| Input shape and preprocessing | TBD |
| Device and OS build | TBD |
| Runtime and version | TBD |
| CPU / GPU / NPU backend | TBD |
| Thread and power settings | TBD |
| Warm-up and measured runs | TBD |
| Latency statistic and unit | TBD |
| Model package size | TBD |
| Peak memory | TBD |
| Unsupported or delegated operations | TBD |

## Reporting protocol

Report preprocessing, inference, postprocessing, and end-to-end latency separately. Include warm-up, synchronization, run count, median, tail percentile, device thermal state, power mode, and runtime delegate. Accuracy after conversion or quantization must be evaluated on the same frozen split used for the source checkpoint.

Do not use “real-time,” “production-ready,” or “on-device validated” until the threshold, named device, runtime, and complete evidence are recorded.

## Privacy and safety

User-contributed images may contain people, locations, device metadata, or farm-identifying context. A future application must define consent, retention, deletion, transport security, and human-review boundaries before collecting images.
