# Security Policy

## Reporting

Do not open a public issue for a suspected credential, private-dataset, personal-data, dependency, mobile-application, model-artifact, or code-execution vulnerability.

A private security contact has not yet been published. Until maintainers add one, use GitHub’s private vulnerability-reporting feature if it is enabled for this repository. If it is unavailable, contact a known project maintainer through an already established private channel. Do not include live secrets or private samples in an initial report.

## Sensitive research artifacts

- Revoke and rotate exposed credentials; deleting a file is not sufficient when it exists in history.
- Treat pickled models, checkpoints, notebooks, archives, and downloaded datasets as untrusted inputs.
- Scan archives before extraction and prevent path traversal outside the intended directory.
- Do not upload raw user or farm imagery without consent, minimization, retention, and deletion rules.
- Report vulnerable dependencies with the affected version, reachable path, and a minimal redacted reproduction.

## Supported versions

No software release is currently published from `main`. Supported-version and response-time commitments are therefore pending the first release.
