---
name: cvio-evidence-guard
description: Review CVio report claims against locked project evidence and local artifacts; detect fabricated, stale, overclaimed, or provenance-ambiguous results before report checkpoints.
---

# Evidence guard

Read PROJECT_CONTEXT_LOCK and LOCAL_ASSET_MAP.

For each changed report section:
- enumerate every numerical result;
- identify the artifact/table/source that supports it;
- flag values sourced only from memory or visual estimation;
- preserve negative results;
- distinguish official retained results from hardware-shifted reproductions;
- reject causal wording unsupported by matched experiments;
- reject mobile claims based on Kaggle/desktop GPU timing;
- reject confusion matrices/classwise metrics if the required prediction ledger is unavailable.

Return findings only; do not edit report source unless the main agent explicitly delegates a disjoint file.
