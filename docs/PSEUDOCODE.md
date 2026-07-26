# TPTAF paper-level pseudocode

```text
F_ir = embed(I_ir) + position
F_vi = embed(I_vi) + position
LL, LH, HL, HH = DWT(F)
F_P = PCSB(LL)
F_D = DDEB(LH, HL, HH)
F_det_input = GLA(merge(preprocess(I_ir), preprocess(I_vi)))
raw_detection, F_det = EXTERNAL_DETECTOR(F_det_input)
T_d, P_d, reference, offsets = DPG(F_det)
F_task = TAM(F_P_ir, F_D_ir, F_P_vi, F_D_vi, F_det)
I_fu = decoder(F_task)
L_total = UWL(L_sime, L_deco, L_grad, L_box, L_cls, L_dfl)
```

The external detector is not redistributed. A compatible adapter must provide a semantic feature map and, during joint training, differentiable `box`, `cls`, and `dfl` losses.

Validation can load pre-generated fused images or call a programmatic fusion runner, compute SSIM/VIF/QABF/SCD/CC/FMI-dct, optionally accumulate detection AP/mAP through a detector callback, and export CSV/JSON summaries.
