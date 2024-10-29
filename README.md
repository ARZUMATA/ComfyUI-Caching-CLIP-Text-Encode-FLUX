# ComfyUI-ARZUMATA
Random nodes for ComfyUI for various purposes.

# Nodes list

### Caching CLIP Text Encode for FLUX

A simple performance-optimized FLUX node that stores CLIP to CONDITIONING conversion results in it's cache.
Caching eliminates redundant processing time by reusing previous results, significantly improving workflow efficiency and saving a lot of time.
It only activates when there is a change in one of the text inputs, clip model change or cache expired.

What node caches:
- CLIP model reference in memory (you can switch clip model and try it without losing cache from previous clip model should you decide to switch back) 
- clip_l string
- t5xxx string

Guidance value is  not cached as it can be updated on the fly.

There is also a cache limit option to limit cache size, it will delete the oldest cache when the cache size is exceeded.

### Caching CLIP Text Encode

Same functionality as Caching CLIP Text Encode for FLUX, but only one text field and no guidance value.

## Installation

Clone this repository to `ComfyUI/custom_nodes` directory.

## Credit
discus0434/[comfyui-caching-embeddings](https://github.com/discus0434/comfyui-caching-embeddings) - inspired by.

**And, for all ComfyUI custom node developers**