# realbot-HDR

Local experiment scripts for pure HDR video training on realbot painting videos.
Pipeline: generate 320x384 stitched videos, sample 61 RGB frames, precache Wan2.2 TI2V latents with shape 16x48x24x20, train Causal-Forcing HDR video with vertical levels [1,2,4,8,16], then run fix5 inference on one heldout episode per source.
