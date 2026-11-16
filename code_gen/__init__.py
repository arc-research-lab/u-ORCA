"""DeepSets and MLP code generation for AMD AI Engines."""

from .spec import ModelSpec, SpecError, build_mlp_spec, build_model_spec

__all__ = ["ModelSpec", "SpecError", "build_mlp_spec", "build_model_spec"]
