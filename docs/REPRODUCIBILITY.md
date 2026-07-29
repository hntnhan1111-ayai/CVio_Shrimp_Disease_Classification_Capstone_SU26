# Reproducibility

## Current readiness

`main` contains documentation and validation tooling but no integrated training or inference runtime. Training, evaluation, export, and benchmark commands are therefore not invented.

| Command group | Readiness | Blocking evidence |
|---|---|---|
| Environment setup | Pending | Package metadata and lock file |
| Dataset validation | Pending | Dataset card, schema, and manifest checker |
| Classification training | Pending | Entrypoint and frozen config |
| Segmentation training | Pending | Entrypoint and frozen config |
| Evaluation | Pending | Metric implementation and evidence format |
| Export | Pending | Selected checkpoint and target format |
| Mobile benchmark | Pending | App/runtime, device, and protocol |

## Documentation checks

These commands are executable from the repository root:

```powershell
python scripts/docs/generate_visual_assets.py
python scripts/docs/generate_shrimp_gif.py
python scripts/docs/check_readme_links.py
python scripts/docs/check_assets.py
python scripts/docs/validate_result_tables.py
```

Run `git diff --check` after regeneration. Generated assets must remain byte-stable across repeated runs.

## Integrated release checklist

- Pin supported Python and framework versions.
- Commit a dependency lock suitable for the supported platforms.
- Provide immutable dataset and split manifests without redistributing restricted data.
- Add a one-command smoke test for each track.
- Record hardware and deterministic limitations.
- Publish exact commands only after they execute in a clean environment.
- Verify streaming/non-streaming or desktop/mobile paths separately when applicable.
