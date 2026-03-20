# PPT Model Figure Page Script

This file is the direct implementation layer of the figure atlas.

Use it when you actually draw the figures in PowerPoint.

## 1. Canvas Setup

Reference template:

- `C:\Users\admin\Desktop\Framework.pptx`

Verified canvas:

- slide width: `13.333 in`
- slide height: `7.5 in`
- aspect ratio: `16:9`

Recommended safe area:

- left / right margin: `0.45 in`
- top title band: `0.48 in`
- bottom footnote band: `0.28 in`
- usable content area: roughly `12.43 in x 6.55 in`

Recommended alignment grid:

- 24-column visual grid
- 0.18-0.22 in spacing between major modules
- 0.10-0.14 in spacing between small inner elements

## 2. Shared Style Presets

### Style A: inherited backbone

- fill: `#EEF2F7`
- outline: `#6F8197`
- outline width: `1.5 pt`
- label: bold, 11-12 pt

### Style B: proposed module

- fill: `#FFF1E3`
- outline: `#E4872E`
- outline width: `1.8 pt`
- label: bold, 11-12 pt

### Style C: condition or prior block

- fill: `#EDF8EF`
- outline: `#63A66F`
- outline width: `1.5 pt`
- label: 10.5-11.5 pt

### Style D: optional or training-only branch

- fill: `#F4F5F7`
- outline: `#A5ABB5`
- outline width: `1.2 pt`
- dashed outline or dashed arrow

### Style E: placeholder image

- fill: `#F8F8F8`
- outline: `#BFC5CC`
- outline width: `1.0 pt`
- centered placeholder text: 9.5-10.5 pt

Shared arrow rules:

- main deployment arrows: `2.0-2.4 pt`
- inner module arrows: `1.4-1.8 pt`
- optional branch arrows: dashed `1.2 pt`
- Stage I -> Stage II transfer arrow: `3.0 pt`, gold `#D9A441`

## 3. Core Slide Scripts

These are the pages you should draw first.

### Slide 1 = Fig.1 Overall Two-Stage Framework

**Page title**

`Overall Pipeline of the Proposed Two-Stage Restoration Framework`

**Layout template**

- dominant overview + two small detail callouts

**Object placement**

- Title band: `x=0.55, y=0.20, w=7.80, h=0.30`
- Stage I outer container: `x=0.55, y=0.92, w=5.25, h=4.95`
- Stage II outer container: `x=7.12, y=0.92, w=5.65, h=4.95`
- Cross-stage transfer ribbon: `x=5.95, y=2.55, w=0.95, h=1.10`
- Bottom-left callout: `x=0.70, y=6.05, w=5.00, h=0.70`
- Bottom-right callout: `x=7.25, y=6.05, w=5.20, h=0.70`

**Stage I internal order**

- `Faded RGB / Lab Observation`
- `RGB-Only Spectral Bridge`
- `Conditional Pigment Diffusion Denoiser`
- `Restored Pigment RGB / Lab`
- `3D Pigment LUT Builder`

Teacher path:

- place `Multimodal Teacher Conditioning` above the denoiser
- use dashed gray arrows only

**Stage II internal order**

- `Degraded Mural Image`
- `Hole Mask`
- `LUT Prior Composer`
- `Mu Cleaner`
- `Prior-Guided Texture U-Net`
- `Official-Compatible Enhanced Reverse SDE`
- `Restored Mural Image`

**Visual emphasis**

- strongest visual center: two-stage containers + gold transfer ribbon
- secondary emphasis: Stage II backbone
- tertiary emphasis: Stage I teacher-only branch and bottom callouts

**Recommended callout text**

- left callout: `Stage I detail: RGB-only bridge and LUT construction`
- right callout: `Stage II detail: prior injection, MGLC, and enhanced reverse SDE`

### Slide 2 = Fig.2 Dual-Baseline Reference Map

**Page title**

`Dual-Baseline Reference Map: From SSD-TS and StrDiffusion to Our Full System`

**Layout template**

- baseline left / right, ours in the middle or lower center

**Object placement**

- Left baseline panel: `x=0.60, y=1.00, w=3.55, h=4.85`
- Center additions panel: `x=4.55, y=1.00, w=4.15, h=4.85`
- Right baseline panel: `x=9.10, y=1.00, w=3.60, h=4.85`
- Bottom summary strip: `x=0.70, y=6.00, w=12.10, h=0.62`

**Content rule**

- left panel only shows Stage I baseline bones
- right panel only shows Stage II baseline bones
- center panel lists your added modules grouped by stage

**Style note**

- both baseline panels should be low saturation blue-gray
- center panel should use orange module chips

### Slide 3 = Fig.6 RGB-Only Spectral Bridge

**Page title**

`RGB-Only Spectral Bridge for Pseudo Spectral Conditioning`

**Layout template**

- left-to-right pipeline + top memory bank

**Object placement**

- Input placeholder: `x=0.62, y=2.42, w=1.15, h=1.05`
- `Color Evidence Encoder`: `x=1.95, y=2.35, w=1.55, h=1.12`
- `Pseudo-Spectrum Predictor`: `x=3.82, y=1.55, w=1.80, h=0.95`
- `Prototype Posterior Estimator`: `x=3.82, y=3.35, w=1.80, h=0.95`
- `Spectral Prototype Bank`: `x=6.00, y=0.95, w=1.65, h=1.25`
- `Retrieval Branch`: `x=6.05, y=2.25, w=1.55, h=0.95`
- `Posterior-Retrieval Confidence Gate`: `x=8.05, y=2.25, w=1.55, h=1.15`
- `Pseudo Spectral Condition`: `x=10.00, y=2.32, w=1.45, h=1.00`
- `Conditional Pigment Diffusion Denoiser`: `x=11.55, y=2.10, w=1.15, h=1.45`

**Extra notes**

- draw the prototype bank as a cylinder or stacked slots
- predictor and posterior blocks should be aligned vertically
- the gate can be a rounded diamond or highlighted fusion node
- add a small gray note below the title:
  - `true spectral condition is used only during training`

### Slide 4 = Fig.10 3D Pigment LUT Construction Pipeline

**Page title**

`Stage-I Deployment Product: 3D Pigment LUT Construction`

**Layout template**

- compact four-step pipeline

**Object placement**

- `RGB Grid Sampling`: `x=0.85, y=2.20, w=2.05, h=1.35`
- `Batch Single-Color Inference`: `x=3.25, y=2.20, w=2.35, h=1.35`
- `Confidence / Uncertainty Diagnostics`: `x=5.95, y=2.20, w=2.45, h=1.35`
- `Optional Stabilization`: `x=8.70, y=2.20, w=1.85, h=1.35`
- `3D Pigment LUT Builder`: `x=10.85, y=1.92, w=1.95, h=1.90`
- output-key note box: `x=9.55, y=4.35, w=3.10, h=1.05`

**Style note**

- use a cube placeholder in the last block
- output note box lists `lut_rgb`, `lut_lab`, `lut_conf`, `lut_std`, `lut_cdiff`, `lut_cret`

### Slide 5 = Fig.12 Enhanced Stage-II Framework on Top of StrDiffusion

**Page title**

`Enhanced Stage-II Framework on Top of StrDiffusion`

**Layout template**

- one dominant overview + right-side detail callouts

**Object placement**

- Main architecture container: `x=0.55, y=0.95, w=9.05, h=5.05`
- Right detail callout 1: `x=9.95, y=1.02, w=2.75, h=1.45`
- Right detail callout 2: `x=9.95, y=2.78, w=2.75, h=1.45`
- Right detail callout 3: `x=9.95, y=4.54, w=2.75, h=1.45`

**Main path order**

- `Degraded Mural`
- `Hole Mask`
- `LUT Prior Composer`
- `Mu Cleaner`
- `Pixel Condition Encoder`
- `Prior-Guided Texture U-Net`
- `Official-Compatible Enhanced Reverse SDE`
- `Restored Mural`

**Callout content**

- callout 1: `Pixel Condition Encoder`
- callout 2: `MGLC inside the backbone`
- callout 3: `reverse_sde(...) compatible enhanced inference`

**Style note**

- keep the main U-Net container visually closest to StrDiffusion baseline
- proposed modules sit around and inside it, not replacing it

### Slide 6 = Fig.13 LUT Prior Composer

**Page title**

`LUT Prior Composer for Color Prior and Confidence Construction`

**Layout template**

- central horizontal chain

**Object placement**

- Input mural placeholder: `x=0.75, y=2.35, w=1.15, h=1.00`
- LUT cube placeholder: `x=0.75, y=1.05, w=1.15, h=0.95`
- `Trilinear LUT Mapper`: `x=2.20, y=2.15, w=1.75, h=1.20`
- `Hole-Region Inpainting`: `x=4.35, y=2.15, w=1.75, h=1.20`
- `Spatial Confidence Estimation`: `x=6.50, y=2.15, w=1.95, h=1.20`
- `Confidence Fusion`: `x=8.90, y=2.15, w=1.60, h=1.20`
- `Color Prior`: `x=10.95, y=1.65, w=1.15, h=0.95`
- `Confidence Map`: `x=10.95, y=3.05, w=1.15, h=0.95`

**Style note**

- the last two outputs should be vertically stacked
- confidence output should use a green heatmap placeholder

### Slide 7 = Fig.14 Pixel Condition Encoder and Multi-Scale Injection

**Page title**

`Pixel Condition Encoder and Multi-Scale Feature Injection`

**Layout template**

- left branch + right backbone levels

**Object placement**

- Combined input box: `x=0.65, y=2.30, w=1.75, h=1.15`
- `Pixel Condition Encoder`: `x=2.75, y=1.80, w=2.00, h=2.15`
- Projection blocks column:
  - `x=5.10, y=1.20, w=1.20, h=0.62`
  - `x=5.10, y=2.05, w=1.20, h=0.62`
  - `x=5.10, y=2.90, w=1.20, h=0.62`
  - `x=5.10, y=3.75, w=1.20, h=0.62`
- U-Net level boxes:
  - `x=7.10, y=1.10, w=2.05, h=0.80`
  - `x=7.10, y=2.00, w=2.05, h=0.80`
  - `x=7.10, y=2.90, w=2.05, h=0.80`
  - `x=7.10, y=3.80, w=2.05, h=0.80`
- Small note box: `x=9.65, y=1.60, w=2.55, h=2.65`

**U-Net labels**

- `Encoder Level 1`
- `Encoder Level 2`
- `Encoder Level 3`
- `Bottleneck`

**Right note box text**

- `Condition channels: noisy image, mask_hole, color prior, confidence`
- `Side branch is aligned with the main U-Net`
- `Zero-conv projections enable stable multi-scale injection`

### Slide 8 = Fig.16 Mask-Gated Local Context Block

**Page title**

`Mask-Gated Local Context Block`

**Layout template**

- anatomy page with central gating

**Object placement**

- Input feature map: `x=0.80, y=2.55, w=1.15, h=0.95`
- `Local Branch`: `x=2.35, y=1.60, w=1.65, h=1.00`
- `Context Branch (sem_lite)`: `x=2.35, y=3.55, w=1.90, h=1.00`
- `mask_hole`: `x=2.10, y=5.05, w=1.05, h=0.70`
- `Boundary Band`: `x=4.65, y=4.90, w=1.45, h=0.80`
- `Mask Gate`: `x=5.05, y=2.55, w=1.25, h=1.05`
- `Residual Fusion`: `x=7.10, y=2.50, w=1.60, h=1.10`
- Output feature map: `x=9.15, y=2.55, w=1.25, h=0.95`
- Footnote box: `x=10.75, y=1.65, w=1.80, h=2.80`

**Footnote text**

- `branch_mode: local / context / both`
- `supports sem_lite backend`
- `used at bottleneck and decoder`
- `mask gate uses hole region and boundary cues`

### Slide 9 = Fig.17 Mu Cleaner

**Page title**

`Mu Cleaner Before SDE`

**Layout template**

- short horizontal pre-processing chain

**Object placement**

- Input block: `x=1.05, y=2.45, w=1.60, h=1.10`
- `Blind-Spot Corruption`: `x=3.15, y=2.45, w=1.90, h=1.10`
- `Mu Cleaner`: `x=5.65, y=2.30, w=2.10, h=1.40`
- `mu_clean`: `x=8.35, y=2.45, w=1.35, h=1.10`
- note box: `x=10.10, y=1.95, w=2.10, h=2.10`

**Note box text**

- `Inputs: degraded RGB, mask_known, confidence`
- `Self-supervised blind-spot training`
- `Only known region is preserved`
- `Applied before state generation / reverse SDE`

### Slide 10 = Fig.18 Official-Compatible Enhanced Reverse SDE

**Page title**

`Official-Compatible Enhanced Reverse SDE`

**Layout template**

- central inference flow + upper optional branches

**Object placement**

- Entry block: `x=0.75, y=2.25, w=2.05, h=1.20`
- Conditioned score block: `x=3.30, y=2.10, w=2.30, h=1.50`
- `pred_full`: `x=6.15, y=2.30, w=1.25, h=1.10`
- Composition block: `x=7.95, y=2.15, w=1.85, h=1.40`
- Final output placeholder: `x=10.40, y=2.20, w=1.55, h=1.25`
- Upper condition nodes:
  - `Color Prior`: `x=3.15, y=0.95, w=1.20, h=0.72`
  - `Confidence`: `x=4.55, y=0.95, w=1.20, h=0.72`
  - `mask_hole`: `x=5.95, y=0.95, w=1.20, h=0.72`
  - `Optional Structure Guidance`: `x=7.55, y=0.85, w=1.95, h=0.88`
  - `Optional Discriminator Guidance`: `x=9.80, y=0.85, w=2.05, h=0.88`
- Lower note strip: `x=0.78, y=5.30, w=11.90, h=0.80`

**Lower note strip text**

- `partial mode: known input + predicted hole`
- `full mode: direct full prediction`
- `compatibility preserved by keeping official reverse_sde(...) entry`

### Slide 11 = Fig.19 Stage-II Training vs Inference Semantics

**Page title**

`Stage-II Training and Inference Semantics`

**Layout template**

- two-column rule sheet

**Object placement**

- Training column: `x=0.75, y=1.10, w=5.70, h=4.95`
- Inference column: `x=6.90, y=1.10, w=5.70, h=4.95`
- Bottom shared legend: `x=0.90, y=6.15, w=11.85, h=0.48`

**Training column entries**

- `mask: 1 = hole in dataset semantics`
- `mask_for_sde = 1 - mask`
- `color prior generation`
- `mu_clean before random state generation`
- `gt_mode: full / partial / mixed`
- `mu_denoiser.* stored with main checkpoint`

**Inference column entries**

- `mask_known and mask_hole are explicit`
- `auto prior generation if prior/confidence not provided`
- `known-region prior synced with latest LUT mapping`
- `mu_clean only on known region`
- `partial/full final composition`
- `intermediate export for debugging`

## 4. Appendix Slide Scripts

These pages are useful but lower priority.

### Slide 12 = Fig.3 Original SSD-TS Baseline Skeleton

- minimal blue-gray page
- only show original diffusion backbone and basic condition path
- no new modules, no LUT

### Slide 13 = Fig.5 Multimodal Teacher Conditioning

- Raman and XRD inputs at the top
- `Multimodal Spectral Encoder` in the center
- dashed arrow to Stage I condition path
- add `teacher-only` label in gray

### Slide 14 = Fig.7 Conditional Pigment Diffusion Backbone

- observation, condition, and time-step embeddings enter from left/top
- stacked denoising blocks in the center
- restored pigment output on the right

### Slide 15 = Fig.8 Stage-I Training Objectives

- split losses into three horizontal rows:
  - diffusion reconstruction
  - bridge alignment
  - optional physics regularization

### Slide 16 = Fig.9 Stage-I Inference Stabilization

- show `confidence`, `diffusion uncertainty`, `retrieval confidence`, `Kalman/RTS`
- best drawn as a post-processing chain

### Slide 17 = Fig.11 Official StrDiffusion Baseline Skeleton

- preserve original `G / Gs / Dis` logic
- use only blue-gray, no orange except tiny note showing "not yet inserted"

### Slide 18 = Fig.15 Prior-Guided Texture U-Net

- large U-Net container with marked insertion positions
- label `PCE injection`, `MGLC at bottleneck`, `MGLC at decoder`

### Slide 19 = Fig.20 Cross-Stage Interface Specification

- left: Stage I `pigment LUT`
- right: Stage II `Trilinear LUT Mapper -> LUT Prior Composer`
- center: one gold transfer arrow

### Slide 20 = Fig.21 Naming and Semantics Legend

- divide page into 4 quadrants:
  - color legend
  - line legend
  - naming legend
  - mask semantics

### Slide 21 = Fig.22 Selection Guide

- three blocks:
  - `Main Paper`
  - `Supplementary`
  - `Defense`
- each block contains figure IDs only

## 5. Text Assets

Use these exact short labels whenever possible:

- `Faded RGB / Lab Observation`
- `RGB-Only Spectral Bridge`
- `Pseudo Spectral Condition`
- `Conditional Pigment Diffusion Denoiser`
- `3D Pigment LUT Builder`
- `Degraded Mural Image`
- `Hole Mask`
- `Color Prior`
- `Confidence`
- `Mu Cleaner`
- `Pixel Condition Encoder`
- `Prior-Guided Texture U-Net`
- `Mask-Gated Local Context Block`
- `Official-Compatible Enhanced Reverse SDE`
- `Optional Structure Guidance`
- `Optional Discriminator Guidance`
- `Restored Mural Image`

## 6. Drawing Order

Use this implementation order inside PowerPoint:

1. place outer containers
2. place main-path boxes
3. place all main arrows
4. add condition and optional branches
5. add placeholders
6. add footnotes and legends
7. unify color and stroke
8. only then replace placeholders with actual thumbnails or icons

## 7. Do-Not-Do List

- do not let optional branches be visually stronger than proposed modules
- do not use different greens for prior and mask in the same page
- do not put too much text inside main-path boxes
- do not turn `Fig.12` into a full decoder-level detail page
- do not mix Stage I training-only teacher path into deployment path
- do not draw `ChromaticResidualRefiner`
- do not erase the official `reverse_sde(...)` entry

## 8. Best Next Step

If you continue implementation after this file, the most efficient move is:

1. build slides `1, 3, 5, 7, 8, 10`
2. export low-fidelity screenshots
3. check whether the paper story already reads cleanly
4. only then draw appendix pages
