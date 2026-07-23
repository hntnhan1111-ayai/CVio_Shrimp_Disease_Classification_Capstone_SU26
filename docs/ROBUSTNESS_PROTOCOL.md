# Mobile-camera robustness protocol

The deployment context is an individual shrimp removed from the pond and photographed onshore using a consumer mobile phone.

Ten families are evaluated at five deterministic severity levels: low-light sensor noise, overexposure clipping, flash glare, handheld motion blur, autofocus defocus, white-balance shift, JPEG recompression, low-resolution resampling, uneven shadow, and lens smudge/fingerprint.

All transforms are photometric or optical, so ground-truth bounding boxes remain unchanged.

Top-five ranking uses mean mAP50-95 gain over the matched baseline, then mAP50, retention, and recall. The selected set is N07, N06, N01, N03, and N10.
