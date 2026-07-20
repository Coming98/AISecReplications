# Replication Plan — LIDMark (CVPR'26)

Paper: *All in One: Unifying Deepfake Detection, Tampering Localization, and Source Tracing with a
Robust Landmark-Identity Watermark* (Wu, Wang, Guo — CVPR 2026).

---

## 0. What "replicating" means here

The repo already contains a full implementation (`model/`, `trainer.py`, `tester.py`, `test.py`,
`main.py`, `configurations/*.yaml`). So "replication" is mostly about:
1. Reconstructing the exact data (images + paired LIDMark `.npy`) the paper trained/tested on.
2. Obtaining the third-party deepfake-generator checkpoints the stochastic manipulation operator needs.
3. Running the two-stage training + unified test with the paper's hyperparameters.
4. Reproducing the paper's tables (imperceptibility, BER/AED under `Mc`/`Md`, cross-dataset on LFW,
   loss-ablation) and the detection/localization threshold analysis.

Everything below is organized so each numbered section is an actionable checklist item.

---

## 1. Environment Setup

From `README.public.md`:

```bash
conda create -n LIDMark python=3.8 -y
conda activate LIDMark
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

`requirements.txt` currently lists: numpy, scipy, opencv-python, face-alignment==1.4.1, scikit-image,
pandas, seaborn, tensorboard, PyYAML, tqdm, matplotlib, pillow.

**Gap found:** `model/losses.py` imports `kornia` (used for `PSNRLoss`/`SSIMLoss`), but `kornia` is
**not** in `requirements.txt`. Need to `pip install kornia` separately — flag this so the env isn't
silently broken at first `import`.

Paper trained on **NVIDIA A40 GPUs**. `configurations/train_distortions.yaml` sets `gpu_ids: "0, 1"`
(multi-GPU) while `tune_deepfakes.yaml`/`test.yaml` use `gpu_ids: "0"` (single GPU) — confirm whether
`trainer.py` actually wraps the model in `DataParallel`/`DistributedDataParallel` for the pretrain
stage, or whether that field is vestigial, before assuming multi-GPU pretraining is required.

---

## 2. Dataset Preparation

**CelebA-HQ**: 30,000 images
- Download - Official page: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- Mapping file: `CelebAMask-HQ/CelebA-HQ-to-CelebA-mapping.txt` is used to map the HQ index to the original CelebA filename, then the official CelebA `list_eval_partition.txt` is applied to assign each HQ image to train/val/test.
- Evaluation Partition - Official page: [Google Drive](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg) -> Eval
- Key Scripts: 
 - `scripts/resize_dataset.py` — crop/resize CelebA-HQ images to 128×128 or 256×256.
 - `scripts/dataset_split.py` — split CelebA-HQ into train/val/test using the mapping + partition file.

**LFW (Labelled Faces in the Wild)**: — used only for cross-dataset generalization test, 5,749 identities;
- Download - Kaggle mirror: https://www.kaggle.com/datasets/ashfaqsyed/labelled-faces-in-the-wild?select=lfw.tgz only need to download the `lfw.tgz`
- Key Script: `scripts/lfw_random_select.py` — per the paper, the LFW dataset is too large to evaluate on in full, so the paper randomly selects 2,000 images for testing. 

**watermak_152**: the pre-generated paired LIDMark data (`.npy` files) is available for download from the authors' Google Drive (see README).


## Final expected directory layout
```
LIDMark/dataset/
├── image/
│   ├── celeba-hq_128/{train,val,test}/*.jpg
│   └── celeba-hq_256/{train,val,test}/*.jpg
└── watermark_152/
    └── celeba-hq/
        ├── 128/{train,val,test}/*.npy
        └── 256/{train,val,test}/*.npy
```

## 3. Model Architecture (already implemented — verify against paper, don't rebuild)

Cross-checked `model/lidmark.py`, `model/modules.py`, `model/discriminator.py`,
`model/distortions.py` against Sec. 3.2 of the paper:

- **Encoder (`LIDMarkEncoder`):** two-stream — image stream (`ConvBlock` + `SEResNet`) and watermark
  stream (`Linear` → reshape to 1×16×16 → `ConvBlock` + `DiffusionNet` (spatial upsample) +
  `SEResNet`×2), fused channel-wise, concatenated with the original image via a global skip
  connection, then a final 1×1 conv → `I_wm`. Matches Fig. 3(a) description.
- **FHD (`FHD` class):** shared backbone (`ConvBlock` → `SEResNetDecoder` → `ConvBlock` → `SEResNet`
  → `ConvBlock`) feeding two parallel FC heads: `landmark_head` (136-D regression) and `id_head`
  (16-D classification logits). Matches the "factorized heads" description in Sec. 3.2/Fig. 3(c).
- **Discriminator:** stack of `ConvBlock`s → global average pool → `Linear` → scalar logit. Matches
  Sec. 3.2's discriminator description.
- **Stochastic Manipulation Operator (`DistortionSimulator` in `model/distortions.py`):** implements
  `Identity`, `Resize`, `GaussianBlur`, `MedBlur`, `JpegTest`, `JpegMask`, and a `RandomDistortion`
  wrapper that picks one per mini-batch — matches the "single-task-per-batch" strategy in Sec. 3.2.
  Deepfake operators (`SimSwap`, `UniFace`, `CSCS`, `StarGAN`, `InfoSwap` wrapper classes) live in
  `model/deepfakes.py`.

This part of the plan is "verify, don't reimplement" — no architectural gaps were found relative to
the paper description.

---

## 4. Loss Functions (verify against Sec. 3.3 equations)

`model/losses.py` provides `LandmarkL2Loss` (Eq. 2, mean per-point Euclidean distance) and
`PSNRLoss`/`SSIMLoss` (kornia-based, for reporting Table 1/5 metrics — not training losses per se).
The BCE identifier loss (Eq. 3), encoder MSE loss (Eq. 1), adversarial/discriminator losses (Eqs. 5–6),
generative consistency loss (Eq. 7), and decoder stability loss (`L_stab`, used only in fine-tuning)
appear to be composed directly inside `trainer.py` (`train_batch_common` / `train_batch_deepfake`) —
worth a targeted read-through of `trainer.py` to confirm the weighted sums in Eq. 8 (`L_G1`) and
Eq. 9 (`L_G2`) are assembled exactly as specified, in particular that `λ_stab = λ_L` is hardcoded
per the paper's stated choice, before trusting reported numbers.

---

## 5. Deepfake Manipulation Operator — third-party assets to acquire

Per README + confirmed by imports in `model/deepfakes.py`, five generators are needed:

| Generator | Official repo | Used in |
|---|---|---|
| SimSwap | https://github.com/neuralchen/SimSwap | pretrain? no — fine-tune (`Md`) + test |
| UniFace | https://github.com/xc-csc101/UniFace | fine-tune + test |
| CSCS | https://github.com/ICTMCG/CSCS | fine-tune + test |
| StarGAN-v2 | https://github.com/clovaai/stargan-v2 | fine-tune + test |
| InfoSwap | https://github.com/GGGHSL/InfoSwap-master | **test only** (held out, unseen-attack generalization check) |

Fine-tuning stage (`M_d`) samples only from {SimSwap, UniFace, CSCS, StarGAN-v2}; InfoSwap is
strictly reserved for testing to measure generalization to an unseen manipulation.

Expected local layout (from README + confirmed by the exact submodule imports in
`model/deepfakes.py`):
```
LIDMark/model/
├── SimSwap/       # must expose models.models.create_model, options.test_options.TestOptions
├── UniFace/       # must expose generate_swap
├── CSCS/          # must expose model.arcface.iresnet.iresnet100,
│                  #   model.arcface.iresnet_adapter.iresnet100_adapter,
│                  #   model.faceshifter.layers.faceshifter.layers_arcface.AEI_Net
├── StarGAN/       # must expose core.solver.Solver
└── InfoSwap/      # must expose modules.encoder128.Backbone128, modules.iib.IIB,
                   #   modules.aii_generator.AII512, modules.decoder512.UnetDecoder512,
                   #   preprocess.mtcnn.MTCNN
```
Each must be cloned from its official repo with its pretrained checkpoint downloaded per that repo's
own instructions (license restrictions mean weights aren't redistributed here). Need to confirm the
exact checkpoint filenames/config each wrapper class in `model/deepfakes.py` expects (a closer read of
each `*Model` class's `__init__` — currently only skimmed for import paths).

Attack simulation detail (Sec. 3.2): identity-swap generators need a *target* (pose/structure) and a
*source* (new identity) image — the repo uses an "intra-batch rolling" strategy, pairing each image
with another image from the same mini-batch as the identity source, rather than needing a separate
paired dataset.

---

## 6. Training — two-stage strategy

### Stage 1 — Pretraining on common distortions (`train_distortions`)
```
python main.py train_distortions --res 128
python main.py train_distortions --res 256
```
Per `configurations/train_distortions.yaml` + paper Sec. 4.1:
- 100 epochs, Adam optimizer, batch size 32, lr = 4.3e-4
- Manipulation pool `M_c`: `Identity()`, `Resize(0.5)`, `GaussianBlur(2,3)`, `MedBlur(3)`,
  `JpegTest(50)`, `JpegMask(50)` — one drawn at random per mini-batch (`RandomDistortion`).
- Loss: `L_G1 = λ_enc·L_enc + L_dec + λ_adv·L_adv`, with `L_dec = λ_L·L_L + λ_ID·L_ID`.
  Paper's reported weights: `[λ_L, λ_ID] = [11.5, 14.7]`. Config file additionally specifies
  `encoder_weight: 1.97` and `discriminator_weight: 0.007` for `λ_enc`/`λ_adv` (not stated explicitly
  as numbers in the paper body — trust the config as the authoritative source since it's what
  actually ran).
- Checkpoints land in `./weights/<res>_152/checkpoints_distortions/`.

### Stage 2 — Fine-tuning on deepfakes (`tune_deepfakes`)
```
python main.py tune_deepfakes --res 128
python main.py tune_deepfakes --res 256
```
Requires Stage 1 checkpoint at `./weights/<res>_152/checkpoints_distortions/checkpoint_epoch_100.pth`
(loaded via `configs.epoch: 100` in `tune_deepfakes.yaml`).
- 100 epochs, batch size reduced to **8** (deepfake generators are GPU-memory heavy), lr = 4.0e-4.
- Manipulation pool `M_d`: SimSwap, UniFace, CSCS, StarGAN-v2 (InfoSwap excluded — reserved for test).
- Loss: `L_G2 = λ_enc·L_enc + L_dec + λ_adv·L_adv + λ_gen·L_gen + λ_stab·L_stab`.
  Paper's reported weights: `[λ_L, λ_ID] = [4.2, 1.0]`, with `λ_stab = λ_L` (explicitly stated design
  choice — regression needs stricter pixel accuracy than classification). Config additionally sets
  `encoder_weight: 11.7`, `generative_weight: 4.1`, `discriminator_weight: 0.02`.
- `L_stab = L_L_stab + L_ID_stab`, computed on the *unattacked* `I_wm` to prevent catastrophic
  forgetting of the identity-mapping case.

Both stages set `seed: 42` for reproducibility (`main.py` calls `set_seed` from the loaded config).

---

## 7. Testing / Evaluation Protocol

```
python main.py test --res 128
python main.py test --res 256
```
Driven by `configurations/test.yaml` (`manipulation_mode: unified`), which re-applies both the common
distortion pool and all five deepfake operators (now including InfoSwap) against the fine-tuned
checkpoint at `./weights/<res>_152/checkpoints_deepfakes/checkpoint_epoch_100.pth` (`epoch: 100`).

### 7.1 Metrics to reproduce
- **Imperceptibility (Table 1, 5):** PSNR / SSIM between `I_co` and `I_wm`, on CelebA-HQ and LFW, at
  both resolutions. Paper's headline numbers: 128×128 → PSNR 40.22 / SSIM 0.98; 256×256 → PSNR 44.31 /
  SSIM 0.99 (CelebA-HQ, Table 1); LFW numbers in Table 5.
- **BER (identifier bit-error rate, Eq. 10)** — per distortion/manipulation, Tables 2–4.
- **AED (landmark average Euclidean distance, pixel space = `L_L` scaled by image size)** — Tables
  2–4 report this "Ours"-only column since AED requires the landmark payload no baseline has.
- **Cross-dataset generalization (Table 4/5, Sec. 4.4):** apply the CelebA-HQ-trained checkpoint
  directly to LFW without retraining.
- **Ablation (Table 6, Sec. 4.5):** rerun with `λ_L`/`λ_ID` toggled on/off (3 configs × 2 resolutions)
  to reproduce the loss-interdependence study. This needs separate training runs, not just eval —
  plan for 4 extra training runs (2 ablation configs × 2 resolutions; the 3rd row per resolution is
  the main full-loss run already covered by §6).

### 7.2 Detection & Localization ("intrinsic-extrinsic" consistency check, Sec. 4.3)
This is a **post-hoc analysis on top of the trained model's outputs**, not something baked into
`main.py test` by default — plan to implement/verify it as an analysis step:
1. For each test image (after any distortion/manipulation `M`), get FHD-recovered intrinsic landmarks
   `Ŵ_L`.
2. Re-run the `face-alignment` library on the manipulated image `M(I_wm)` to get extrinsic landmarks
   `W_new`.
3. Compute AED(`Ŵ_L`, `W_new`) per image.
4. **Global detection:** on a validation set, run ROC analysis over common-distortion vs.
   deepfake-manipulation AED distributions to derive a decision threshold via Youden's J statistic.
   Paper reports AUC = 0.9388 and threshold = 3.2375 px — since threshold is dataset/run-dependent,
   expect our reproduced threshold to be close but not necessarily bit-identical; report both AUC and
   derived threshold from our own validation run.
5. **Local localization:** for images flagged fake by the global check, compute AED per semantic
   landmark group (jawline / left eyebrow / right eyebrow / left eye / right eye / nose / mouth,
   per the landmark ordering baked into `W_L` — see Fig. 3(e)) against the same threshold to flag
   which regions were tampered.
6. Cross-check `tester.py` / `test.py` for any existing partial implementation of this ROC/AED
   analysis before writing new analysis code — worth a dedicated read before treating this as a
   from-scratch task.

### 7.3 Baselines (optional — only if a full comparison table is desired)
MBRS, CIN, SepMark, EditGuard, LampMark, DiffMark, KAD-NET are compared against in every table. None
of their code is vendored in this repo. Reproducing the comparison tables exactly would require
separately cloning/training each baseline from its own repo — treat this as an explicit stretch goal,
not part of the core LIDMark replication, unless requested.

---

## 8. Known Gaps / Risks Before Running Anything

1. **`scripts/dataset_split.py` is empty** — the official CelebA-HQ split procedure isn't actually
   present in the repo; must be reconstructed from the CelebA-HQ↔CelebA identity mapping + official
   partition file, or obtained from the authors.
2. **No watermark-generation script** for building `.npy` LIDMark pairs from raw images — only the
   pre-generated download is directly usable out of the box; generating our own for new images means
   writing this ourselves (face-alignment extraction + SHA-256 ID hashing + the exact `.npy`
   layout `utils.py`'s dataset loader expects).
3. **`kornia` missing from `requirements.txt`** despite being imported in `model/losses.py`.
4. **Config YAMLs hardcode the authors' absolute paths** (`/home/wjj/LIDMark/...`) — must be edited
   to local `dataset/`/`weights/` paths before running.
5. **Five third-party deepfake generators must be manually assembled** under `model/<Name>/` with
   exact submodule paths (§5) and their own pretrained checkpoints — this is the single largest
   external dependency and biggest source of possible non-reproducibility (checkpoint versions/config
   differences between our download and the authors' unspecified exact checkpoint source).
6. **Detection/localization threshold (3.2375 px, AUC 0.9388)** is derived empirically from a
   validation run in the paper — treat as a target to approximately reproduce, not to match exactly.
7. **Multi-GPU (`gpu_ids: "0, 1"`) handling** in `train_distortions.yaml` — confirm `trainer.py`
   actually uses both GPUs before assuming this is required infrastructure.

---

## 9. Suggested Execution Order

1. Set up conda env; add `kornia` to the install step.
2. Download CelebA-HQ + LFW; reconstruct/apply the official split; preprocess to 128×128 and 256×256.
3. Download the authors' pre-generated LIDMark `.npy` package (fastest path to parity) and lay out
   `dataset/` per §2.6; fix absolute paths in the three YAML configs.
4. Read through `trainer.py` once to confirm loss assembly matches Eqs. 1–9 exactly (§4) — do this
   before the first training run so any mismatch is caught early, not after a 100-epoch run.
5. Run Stage 1 (`train_distortions`) at 128 and 256 — verify PSNR/SSIM/BER/AED trend sanity on a few
   epochs before committing to the full 100.
6. Acquire the four deepfake generators needed for fine-tuning (SimSwap, UniFace, CSCS, StarGAN-v2)
   and wire them under `model/<Name>/` per §5; acquire InfoSwap separately for test-time-only use.
7. Run Stage 2 (`tune_deepfakes`) at 128 and 256.
8. Run `main.py test` at both resolutions; collect PSNR/SSIM/BER/AED tables (§7.1) and compare against
   Tables 1–3.
9. Repeat training with LFW substituted at test time only (no retrain) for the cross-dataset numbers
   (§7.1, Tables 4–5).
10. Implement/run the intrinsic-extrinsic consistency check analysis (§7.2) to reproduce the
    detection ROC/AUC and localization behavior (Fig. 5, and the qualitative Fig. 4 rows 4–6).
11. Run the two loss-ablation configs (§7.1 ablation) to reproduce Table 6.
12. (Optional/stretch) Reproduce baseline numbers by cloning MBRS/CIN/SepMark/EditGuard/LampMark/
    DiffMark/KAD-NET separately.