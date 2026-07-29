# Datasets

No dataset is approved as the integrated `main` source of truth, and no data is redistributed here.

## Dataset registry

| Dataset ID | Track | Source | License | Version | Access | Status |
|---|---|---|---|---|---|---|
| TBD-CLS | Classification | TBD | Pending verification | TBD | TBD | Pending verification |
| TBD-SEG | Instance segmentation | TBD | Pending verification | TBD | TBD | Pending verification |

## Required dataset card

Before a dataset is accepted, record:

- canonical name, source URL, citation, owner, and access date;
- redistribution and derivative-artifact terms;
- version, immutable fingerprint, image count, class ontology, and instance count;
- annotation format, annotator protocol, quality review, and known uncertainty;
- train/validation/test counts and split-manifest hash;
- specimen, burst, video, location, or source grouping used to prevent leakage;
- duplicate and near-duplicate detection method;
- preprocessing, excluded records, and every transformation before training;
- known demographic, geographic, device, environmental, and label limitations;
- handling of personal information, farm/location metadata, and user-contributed images.

## Leakage gate

Random image-level splitting is not accepted when related frames, captures, specimens, or sources can cross partitions. The grouping unit and leakage audit must be explicit and machine-verifiable.

## Data governance

Repository access does not grant dataset reuse rights. Data, annotations, previews, cached downloads, and trained weights must not be committed until their licenses and privacy constraints are documented and approved.
