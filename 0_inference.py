

import time, torch
#from model_segmamba.segmamba import SegMamba
#from token_priority.token_priority_segmamba import SegMamba
#from model_causal.segmamba_causal import SegMamba
import sys
sys.path.insert(0, "/home/huiseung/Workspace/SegMamba_reproduce/mamba")
#from token_priority.token_priority_segmamba_stride_modify import SegMamba
#from token_priority.token_priority_segmamba_new import SegMamba
#from model_causal.Adventurer_mamba_layer import SegMamba
#from token_priority.fast_sort_token_priority_segmamba import SegMamba
#from window_mhsa_assm_glbtoken.irv2_mamba import SegMamba
#from model_segmamba.segmamba import SegMamba
#from model_assm_only.irv2_mamba_assm_only import SegMamba
#from ase_glbtoken_3d.irv2_mamba_assm_only import SegMamba
#from Proposed.model_modified import SegMamba
#from model_segmamba.segmamba import SegMamba
from ase_glbtoken_3d_768.irv2_mamba_assm_only import SegMamba

#t1 = torch.rand(1, 4, 128, 128, 128).cuda()
t1 = torch.rand(1,4, 128, 128, 128, device='cuda').requires_grad_()
#t1 = torch.rand(1, 4, 64, 64, 64).cuda()

model = SegMamba(in_chans=4,
                 out_chans=4,
                 depths=[2,2,2,2],
                 feat_size=[48, 96, 192, 384]).cuda()


torch.cuda.synchronize(); t0 = time.time()
out = model(t1)
print(f"final out: {out.shape}")
loss = out.sum()
loss.backward()
torch.cuda.synchronize()
print("Elapsed (F+B):", time.time()-t0)
"""
out = model(t1)

print(out.shape)

print(f"Current GPU memory allocated: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
print(f"Current GPU memory reserved:  {torch.cuda.memory_reserved() / (1024**2):.2f} MB")
"""


