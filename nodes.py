import torch
from typing import Dict, Union, Tuple, List, Any

class CachingCLIPTextEncodeFlux:
    """A caching CLIP text encoder for FLUX that stores previous encodings to avoid redundant processing."""
    
    def __init__(self) -> None:
        self.cache: Dict[str, Union[str, torch.Tensor, None]] = {
            "clip_l": None,
            "t5xxl": None,
            "cond": None,
            "pooled_output": None,
        }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", ),
                "clip_l": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "t5xxl": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "guidance": ("FLOAT", {
                    "default": 3.5,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "advanced/conditioning/flux"

    def encode(self, clip: Any, clip_l: str, t5xxl: str, guidance: float) -> Tuple[List[List[Any]], ...]:
        """
        Encode text using CLIP with caching mechanism.
        
        Args:
            clip: CLIP model instance
            clip_l: Main CLIP text input
            t5xxl: T5XXL text input
            guidance: Guidance scale for conditioning
            
        Returns:
            Tuple containing conditioning and pooled outputs
        """
        if clip_l != self.cache["clip_l"] or t5xxl != self.cache["t5xxl"]:
            # Generate new encodings if inputs changed
            tokens = clip.tokenize(clip_l)
            tokens["t5xxl"] = clip.tokenize(t5xxl)["t5xxl"]

            output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
            cond = output.pop("cond")
            output["guidance"] = guidance

            # Update cache
            self.cache.update({
                "clip_l": clip_l,
                "t5xxl": t5xxl,
                "cond": cond,
                "pooled_output": output
            })

            return ([[cond, output]],)
        
        # Return cached results if inputs unchanged
        return ([[self.cache["cond"], self.cache["pooled_output"]]],)

NODE_CLASS_MAPPINGS = {
    "CachingCLIPTextEncodeFlux": CachingCLIPTextEncodeFlux,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CachingCLIPTextEncodeFlux": "Caching CLIP Text Encode for FLUX",
}