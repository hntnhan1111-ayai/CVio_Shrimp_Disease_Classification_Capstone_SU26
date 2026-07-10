# Custom Attention Model Builder Notes

These experiments keep the clean baseline data pipeline, grouped split, training
arguments, evaluation logic, and export cells. The only architecture change is a
small attention module inserted into YOLO11n-seg.

## Parser strategy

Ultralytics `parse_model` normally needs explicit cases for custom modules so it
can infer output channels. The notebooks avoid editing the installed package on
disk. Instead they:

1. Define the custom PyTorch modules in a notebook cell.
2. Add the classes to `ultralytics.nn.tasks`.
3. Patch `ultralytics.nn.tasks.parse_model` in memory.
4. Build the model from the experiment YAML.

No global Ultralytics source file is modified by these experiments.

## YAML strategy

The YAML files use explicit YOLO11n channel sizes rather than the generic
`scales` block. This keeps the channel values transparent for custom attention
modules:

- P3 output: 64 channels
- P4 output: 128 channels
- P5 output: 256 channels
- Segment prototype channel argument: 64

The modules are shape-preserving. For single-input modules, parser output
channels are set to `ch[f]`. For `AttentionGate`, parser output channels are set
to the first input feature, and the semantic gate feature is resized with
nearest-neighbor interpolation inside the module.

## Build checks

Minimum local checks:

```bash
python custom_attention/common/sanity_check_attention.py
```

The notebook also includes a shape sanity-check cell before training and a model
smoke-build cell after writing the YAML.
