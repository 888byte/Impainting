from __future__ import annotations

from typing import Any, Dict, Tuple


def unpack_model_components(model_components: Any) -> Tuple[Dict[str, Any], Any, Any, Any, Any, Any]:
    if isinstance(model_components, dict):
        return (
            model_components.get('cfg', {}),
            model_components['denoiser'],
            model_components['conditioner'],
            model_components['schedule'],
            model_components.get('color_encoder', None),
            model_components.get('cond_predictor', None),
        )
    return model_components
