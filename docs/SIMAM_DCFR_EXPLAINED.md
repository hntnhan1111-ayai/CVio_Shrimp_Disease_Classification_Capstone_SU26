# SimAM-DCFR explained

## What the block does

SimAM-DCFR recalibrates the final spatial feature tensor immediately before the
classification head. It joins a SimAM-style energy response with two learned
branches and returns a residual result with the same shape as its input.

![Block diagram showing texture and channel gates followed by residual fusion](assets/diagrams/simam_dcfr_block.svg)

*Figure 1. The two learned branches preserve the input channel count and spatial
shape. “DCFR” denotes this project's composite recalibration design.*

## Definitions

- A **tensor** is a multidimensional numeric array; here `X` has shape
  `[batch, channel, height, width]`.
- A **feature map** is the height × width activation grid in a convolutional
  representation.
- A **feature channel** is one learned feature-map plane.
- A **depthwise 3 × 3 convolution** applies a separate 3 × 3 spatial filter to
  each channel (`groups=channels`), without mixing channels.
- A **pointwise 1 × 1 convolution** mixes information across channels at each
  spatial position.
- A **sigmoid texture mask** maps the texture branch to values from 0 to 1 for
  spatial/channel-wise modulation.
- **Global average pooling** averages every spatial location in each channel,
  producing one summary value per channel.
- A **channel gate** is a learned sigmoid weight per feature channel.
- **Residual fusion** adds the recalibrated response back to the unchanged input,
  preserving a direct information path.

## Exact forward computation

For each channel, the code estimates spatial squared deviation and variance,
then computes the SimAM-style response:

```text
d = (X - mean_spatial(X))^2
v = sum_spatial(d) / max(1, H*W - 1)
S = X * sigmoid(d / (4 * (v + 1e-4)) + 0.5)
```

The learned texture and channel branches are:

```text
T = X * sigmoid(pointwise_1x1(depthwise_3x3(X)))
G = sigmoid(pointwise_1x1(global_average_pool(X)))
```

Residual fusion returns:

```text
output = X + 0.5 * G * (S + T)
```

The implementation is in
[`simam_dcfr.py`](../src/cvio_asl_ldam/attention/simam_dcfr.py), while insertion
before `Classify` is in
[`patch_yolo.py`](../src/cvio_asl_ldam/attention/patch_yolo.py).

## What is published and what is new here

[SimAM](https://proceedings.mlr.press/v139/yang21o) is a published
parameter-free attention method based on a closed-form neuron-energy response.
The depthwise texture branch, learned channel gate, their averaging, the
residual fusion rule, and the late pre-head injection together are
project-created. The complete SimAM-DCFR block therefore has learned parameters
even though the SimAM term itself is parameter-free.

## Verification implication

The audited proposed EXT-3 checkpoint contains the wrapper, the SimAM-DCFR
module, depthwise 3 × 3 and pointwise 1 × 1 convolutions, both sigmoid gates,
and a `[1, 3]` output. The rejected SDI-4 SHA `4305e491...38c1` contains none of
that late wrapper and is indistinguishable from the audited CE architecture.
