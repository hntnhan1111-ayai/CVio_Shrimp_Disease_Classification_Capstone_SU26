"""TFLite interpreter sanity checks. Never trains."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def tflite_sanity_check(model_path: str | Path) -> dict[str, Any]:
    """Run a LiteRT/TFLite interpreter on a dummy input and return shapes/dtypes.

    Tries `ai_edge_litert` first, then `tensorflow`'s interpreter.
    """
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"TFLite model not found: {model_path}")

    interpreter_cls = None
    backend = None
    try:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore

        interpreter_cls = Interpreter
        backend = "ai_edge_litert"
    except Exception:
        try:
            from tensorflow.lite.python.interpreter import Interpreter  # type: ignore

            interpreter_cls = Interpreter
            backend = "tensorflow"
        except Exception as exc:
            raise ImportError(
                "Neither ai_edge_litert nor tensorflow is installed; cannot run "
                "TFLite sanity check. Install requirements-export-litert.txt."
            ) from exc

    import numpy as np

    interpreter = interpreter_cls(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_shape = tuple(int(x) for x in input_details["shape"])
    input_dtype = str(input_details["dtype"])
    dummy = np.zeros(input_shape, dtype=input_details["dtype"])
    interpreter.set_tensor(input_details["index"], dummy)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])
    output_shape = tuple(int(x) for x in output.shape)
    output_dtype = str(output_details["dtype"])

    return {
        "status": "ok",
        "backend": backend,
        "model_path": str(model_path),
        "input_shape": list(input_shape),
        "input_dtype": input_dtype,
        "output_shape": list(output_shape),
        "output_dtype": output_dtype,
        "output_is_finite": bool(np.isfinite(output).all()),
    }
