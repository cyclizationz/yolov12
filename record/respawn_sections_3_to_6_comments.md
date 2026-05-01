# RESPAWN Paper Comments for Sections 3-6

This note compares the current paper text in `RESPAWN_2026 (1).pdf` against the current implementation in `deployment/`.

## Implementation Walkthrough for `offline_processor.cpp:2785-3407`

The code block in `offline_processor.cpp:2785-3407` implements the first major half of the per-frame server pipeline: it decides, for each detected region, whether the region should be treated as reference-reconstructable or left as raw video content; records the template identity needed for later client reconstruction; constructs the union of pixels that will be modified in the encoded stream; and finally applies the fill-and-feather masking operation. Conceptually, this block corresponds to the server-side stages of identity matching, reliability gating, metadata preparation, and adaptive masking.

The logic begins with a per-detection loop over all segmentation outputs that survived model inference. Detections below the confidence threshold or with empty masks are ignored. Every remaining detection contributes to the total occurrence count, after which the implementation follows one of two paths depending on whether the frame is being processed in pixel-game mode or YOLO mode.

### Pixel-Game Path

In pixel-game mode, the implementation assumes that the detection already corresponds to a stable template-like sprite track. The server therefore does not perform latent-key matching. Instead, it chooses a masking color, paints the sprite region directly into the server-side output frame, accumulates the sprite mask into a full-frame union mask, and emits a metadata record describing the bounding box and template path. By default, the fill color is derived from the sprite kind (`kind_color_from_kind_id`), which improves temporal prediction by keeping visually similar sprite categories painted with consistent colors. If the user explicitly requests an alternate masking strategy such as dominant-color masking, black, brown, or a specific RGB value, that request overrides the default sprite-specific color.

For each pixel-mode detection, the code also appends a region record into the `sei_regions` array. This record contains the region ID, bounding-box coordinates, a reference flag, the sprite kind as `class_id`, and a relative template path (`seiPath`) that later reconstruction code uses for overlay. In other words, pixel mode behaves as a direct template-referencing path with no latent-bank identity assignment.

### YOLO Path: Overview

For YOLO detections, the pipeline is more involved. The code first clips the detection box to the frame, extracts the detection mask inside the valid box, and then chooses one of three server behaviors:

1. an upper-bound masking mode that paints every detected mask regardless of recoverability;
2. a latent-key mode that uses the YOLO segmentation head's 32-dimensional mask coefficients as identity embeddings; or
3. a legacy matcher family based on pHash, IoU-only reuse, or RGB-histogram similarity.

The latent-key path is the main implementation of the RESPAWN matching design.

### Upper-Bound Masking Mode

If `yoloForceMaskAll` is enabled, the server treats every detected region as maskable, regardless of whether the client could reconstruct it correctly. In this mode, the code simply ORs the detection mask into the full-frame union mask, marks the region as masked, stores the detection box for later fill application, and increments an accounting counter. No identity validation or heal-only logic is applied. This mode is explicitly a bitrate upper bound: it estimates the compression benefit of removing all detected object texture, but the recovered output is not guaranteed to be correct.

### Latent-Key Identity Assignment

In latent-key mode, the code first L2-normalizes the 32-dimensional `maskCoeff` vector emitted by the segmentation head, producing a unit embedding suitable for cosine similarity comparison. It then decides whether the current frame should be allowed to mint a new template aggressively or should only attempt reuse. This decision is governed by a hybrid sampling policy:

- in build-index mode, the code forces minting so that the latent bank is populated aggressively;
- otherwise, minting is periodically forced every `latentSamplePeriod` frames;
- and independently, recent motion can trigger a temporary mint-every-frame boost.

The motion-triggered boost is based on the most recent latent assignment for the same class. The code finds the previously tracked instance of that class with the highest bounding-box overlap, then measures three geometric signals between the previous and current box: IoU, center displacement in pixels, and relative area change. If the overlap drops below `latentMotionIouThr`, the center shift exceeds `latentMotionCenterPx`, or the scale change exceeds `latentMotionScaleThr`, the code opens a boost window lasting `latentMotionBoostFrames` frames, during which periodic minting is forced. This mechanism is not a hard correctness gate; rather, it is a sampling heuristic intended to improve fluency and reduce stale reuse under motion.

After that, the code attempts a temporally stable reuse decision. For the current object class, it searches the recent per-class latent history and selects the previous instance with the highest overlap. If that overlap is at least `0.7` and the cosine similarity between the previous embedding and the current embedding is at least `latentCosineThreshold - 0.01`, the code immediately reuses the previous `template_id`. This is a temporal bias designed to suppress flicker and unnecessary ID changes for objects that are still geometrically aligned with a recent instance.

If temporal reuse does not resolve the identity, or if forced minting is active, the code performs a global search over the latent bank. It scans all stored templates of the same class and computes cosine similarity against each template embedding. In non-heal-only mode, it additionally applies a tight size filter, requiring width and height to match within two pixels before a template is considered. The highest-scoring candidate is recorded as `(bestId, bestSim)`.

The final identity decision follows four cases:

1. If no temporal match was chosen and forced minting is not active, the code reuses `bestId` when `bestSim >= latentCosineThreshold`.
2. If forced minting is active but the best existing template is already extremely similar, the code skips the mint and reuses the existing ID when `bestSim >= latentPeriodSkipThr`.
3. If the code is still about to mint, but the best existing template is above an even more conservative merge threshold, it merges the would-be new template into that existing ID when `bestSim >= latentMergeThr`.
4. Otherwise, it allocates a fresh `template_id`, extracts an RGBA crop from the current frame and mask, writes that crop to `dictDir/<template_id>.png`, and appends the new template to the latent bank.

Thus, the latent-key implementation is not a pure thresholded nearest-neighbor classifier. It combines temporal bias, periodic resampling, motion-aware minting, optional redundancy suppression, and online template-bank growth.

### Heal-Only Reliability Gating in Latent Mode

Once a `template_id` has been selected, the code separately decides whether the server is allowed to mask the corresponding region. In ordinary non-heal-only mode, all selected regions are maskable. In `yoloHealOnly` mode, however, the implementation requires that the region be recoverable using a template that already exists in the server-side template pool and whose alpha mask remains geometrically compatible with the current segmentation.

This recoverability test is implemented through the `try_id` lambda. For a candidate template ID, the code loads the corresponding RGBA template from cache or disk, resizes it to the current detection size if necessary, and then performs a local alignment search around the detection box. The alignment step uses alpha-masked RGB mean-squared error to estimate the best small integer offset between the stored template and the current frame. The candidate is rejected if the alignment error exceeds `yoloHealAppearanceMseThr`.

Even after appearance alignment succeeds, the code applies two additional safeguards. First, it computes the fraction of template-alpha pixels that still overlap the current segmentation mask and rejects the candidate if that coverage falls below `yoloHealMinMaskCoverage`. Second, it checks whether the aligned template alpha matches the current segmentation mask under the configured heal thresholds: `template_alpha_matches_mask` enforces a spill constraint (`yoloHealMaskExtraThr`) and, if enabled, a mask-IoU lower bound (`yoloHealMaskIouThr`). Only if all of these checks succeed is the region considered healable.

The heal-only logic tries candidates in three stages. It first tests the currently selected `chosen_id` when that ID represents a reuse rather than a newly minted template. If that fails, it sorts all same-class latent templates by cosine similarity and tests the best few candidates that remain above the latent threshold. If no candidate passes, it finally falls back to the most similar template used recently within a short temporal window, using a lower similarity threshold at approximate GOP boundaries. This fallback is intended to increase masking ratio while preserving short-term temporal coherence.

### Match-Only Mode from a Prebuilt Latent Bank

If `latentNoMint` is enabled, the code disables all minting after identity selection. In that configuration, the system behaves as a match-only simulator against a prebuilt latent bank: if no reusable template can be found, the region is skipped and left raw. This mode is useful when evaluating a previously constructed template pool without allowing the online run to add new templates.

### Server-Side Bookkeeping After Latent Matching

After choosing an ID, the code records that ID in `det_tplids`, marks whether it was newly minted in `det_is_new`, stores the eventual stitching box, and updates per-class latent history so that future frames can reuse recent assignments. The history is maintained as a short vector per class, matched by IoU, allowing the implementation to distinguish multiple active objects of the same class. The code also updates counts such as template mint totals, reuse totals, motion-boost mint totals, and template use frequency.

If the run is in build-index mode, the pipeline stops here for the current detection. No masking or metadata emission is performed; the purpose of the pass is simply to populate the template pool and latent bank for later heal-only evaluation.

### Converting a Healable Latent Match into Masking and Metadata

If the run is not build-index-only, the code next decides whether the selected region should contribute pixels to the mask union and whether a metadata region should be emitted. In non-heal-only mode, every selected region qualifies. In heal-only mode, only regions that passed the recoverability checks qualify.

When a latent region is accepted for masking, the code adds either the segmentation mask or the aligned template alpha mask into the union mask, depending on the mode. In ordinary mode, it uses the segmentation mask directly. In heal-only mode, it uses the template alpha after alignment because that is the exact region the client can reliably stitch back. At the same time, it accumulates statistics such as masked bounding-box area and masked alpha area, marks the detection as actually masked, and emits a `SEIRegion` record containing the chosen `template_id`, the stitching box, the region flag (`0` for newly minted, `1` for reference reuse), and the class ID. In latent-key mode, the path field is intentionally left empty because reconstruction uses the numeric template ID alone.

### Legacy Matchers: pHash, IoU-Only, and RGB Histogram

If the active YOLO matcher is not `latent_key`, the code falls back to a legacy dictionary-based path. It first computes a perceptual hash for the detection mask ROI. It then tries a temporal shortcut using the last matched template for the same class, requiring `IoU >= 0.7` before reuse. Depending on the configured matcher, reuse is then based on direct last-box continuity (`iou_only`), Hamming distance between pHashes (`phash`), or reuse of the previous RGB-histogram template (`rgb_hist`).

If the temporal shortcut fails, the code performs a global lookup: pHash mode searches the dictionary for the nearest perceptual hash within threshold, while RGB-histogram mode compares the current ROI against stored templates using cached histogram descriptors. If a match is found, the existing dictionary item is reused and its count is incremented. Otherwise, the server creates a new canonical key using `sha256_norm(maskROI)` and adds the detection to the template dictionary.

This legacy path also supports heal-only masking. If `yoloHealOnly` is active and the template is not new, the code loads the template image, optionally resizes it, performs a small offset search to minimize spill outside the current mask, and then verifies template-mask agreement via `template_alpha_matches_mask`. Only if that agreement check succeeds does the region become maskable. Accepted legacy regions are then accumulated into the union mask, recorded in `det_masked`, stored with their stitching box, and emitted as SEI records containing the canonical hash in the `path` field.

### Fill-Frame Construction and Feathered Mask Application

After all detections have been processed, the code performs the actual server-side masking operation for YOLO mode. Up to this point, the implementation has only decided which pixels are eligible to be modified; it has not yet written the final filled pixels into the output frame. The union mask `mask_union_u8` now represents the combined set of all pixels that will be altered in the encoded stream.

The server builds a fill frame by calling `make_fill_frame_bgr`, which can synthesize the replacement pixels in several ways depending on configuration: a solid color, a blurred frame, an exponential-moving-average background estimate, or an inpainted background. If the fill frame is valid and the union mask is non-empty, the code iterates over all masked detections and applies `apply_fill_with_feather_bbox` within each detection box. This routine copies fill pixels inside the masked region and, when feathering is enabled, blends the fill with the original frame over a narrow band near the boundary. The effect is to create an encoder-friendly, low-texture masked stream while reducing the cost of sharp edges.

The elapsed time for this entire first-pass server processing stage is recorded as `paint_ms`, and the code folds dictionary and gating cost into the per-frame postprocessing time so that later reporting treats identity assignment and reliability gating as part of the overall postprocess stage.

### Build-Index Exit

Finally, if the run is in build-index mode, the code increments the frame counter, prints periodic indexing progress, and immediately continues to the next frame without generating masked or recovered outputs. In other words, build-index mode uses the same per-detection identity logic as the normal server pass, but truncates the pipeline before masking, encoding, and reconstruction so that it can serve as an offline template-bank construction pass.

## Server-Side Flow Graph for `offline_processor.cpp:2785-3407`

The same words "geometry agreement" and similar variable names are used for several different decisions in this block. The graph below separates them into three distinct checks:

- `Geometry check A`: motion-triggered mint boost.
- `Geometry check B`: temporal reuse bias toward the last latent assignment.
- `Geometry check C`: heal-only recoverability / template-mask agreement.

```mermaid
flowchart TD
    A[Start per-frame pass 1] --> B[Loop over detections]
    B --> C{confidence >= thr\nand boxMask non-empty?}
    C -- no --> B
    C -- yes --> D{pixelMode?}

    D -- yes --> P1[Choose pixel mask color]
    P1 --> P2[Paint processed frame directly]
    P2 --> P3[OR detection mask into mask_union_u8]
    P3 --> P4[Emit SEI region with seiPath]
    P4 --> B

    D -- no --> Y1[Clip bbox and extract maskROI]
    Y1 --> Y2{safeBox valid\nand maskROI non-empty?}
    Y2 -- no --> B
    Y2 -- yes --> Y3{yoloForceMaskAll?}

    Y3 -- yes --> F1[OR seg mask into mask_union_u8]
    F1 --> F2[Mark det_masked and det_stitch_boxes]
    F2 --> B

    Y3 -- no --> Y4{matcher == latent_key?}

    Y4 -- yes --> L1[L2-normalize maskCoeff to emb]
    L1 --> L2[Set force_mint from buildIndex or latentSamplePeriod]
    L2 --> GMA[Geometry check A:\ncompare current box to recent same-class box\nIoU / center shift / area delta]
    GMA --> L3{motion exceeds thresholds?}
    L3 -- yes --> L4[Open latent_boost_until_frame]
    L3 -- no --> L5[No boost update]
    L4 --> L6[Apply boost window to force_mint]
    L5 --> L6

    L6 --> GMB[Geometry check B:\nfind best-overlap recent latent for same class]
    GMB --> L7{best IoU >= 0.7\nand cosine >= thr-0.01?}
    L7 -- yes --> L8[Reuse last template_id as chosen_id]
    L7 -- no --> L9[No temporal reuse]
    L8 --> L10[Global latent-bank search if needed]
    L9 --> L10

    L10 --> L11[Scan same-class latent bank\noptional size filter in non-heal-only mode]
    L11 --> L12[bestId, bestSim]
    L12 --> L13{reuse by latentCosineThreshold?}
    L13 -- yes --> L14[chosen_id = bestId]
    L13 -- no --> L15{skip mint by latentPeriodSkipThr?}
    L15 -- yes --> L16[chosen_id = bestId;\nforce_mint = false]
    L15 -- no --> L17{merge mint by latentMergeThr?}
    L17 -- yes --> L18[chosen_id = bestId;\nforce_mint = false]
    L17 -- no --> L19{need mint?}
    L19 -- yes --> L20[Mint new template_id\nextract RGBA\nsave PNG\nappend latent_bank]
    L19 -- no --> L21[Reuse existing chosen_id]

    L14 --> H0
    L16 --> H0
    L18 --> H0
    L20 --> H0
    L21 --> H0

    H0{latentNoMint?} -- yes --> H1{chosen_id exists?}
    H1 -- no --> H2[Skip region as raw]
    H1 -- yes --> H3[Proceed]
    H0 -- no --> H3

    H3 --> HC0{yoloHealOnly?}
    HC0 -- no --> HM1[Maskable immediately]
    HC0 -- yes --> GMC[Geometry check C:\nheal-only recoverability]

    GMC --> HC1[try_id(candidate):\nload template, resize, local offset search,\nappearance MSE, mask coverage,\ntemplate_alpha_matches_mask]
    HC1 --> HC2{chosen reuse candidate passes?}
    HC2 -- yes --> HC5[healable_now = true]
    HC2 -- no --> HC3[Try top-K cosine candidates]
    HC3 --> HC4{any candidate passes?}
    HC4 -- yes --> HC5
    HC4 -- no --> HC6[Fallback to recent latent within short window]
    HC6 --> HC7{fallback candidate passes?}
    HC7 -- yes --> HC5
    HC7 -- no --> HC8[healable_now = false]

    HC5 --> HM0
    HC8 --> HM0
    HM1 --> HM0

    HM0{maskable?} -- no --> HS1[Count heal skip / raw region]
    HM0 -- yes --> HM2{yoloHealOnly?}
    HM2 -- yes --> HM3[OR aligned template alpha into mask_union_u8]
    HM2 -- no --> HM4[OR segmentation mask into mask_union_u8]
    HM3 --> HM5[Mark det_masked, det_stitch_boxes,\nemit SEI with template_id]
    HM4 --> HM5
    HM5 --> B
    HS1 --> B

    Y4 -- no --> R1[Legacy matcher path]
    R1 --> R2[Compute pHash for maskROI]
    R2 --> R3[Try last template for same class\nwith IoU >= 0.7]
    R3 --> R4{matched?}
    R4 -- no --> R5[Global lookup:\npHash or RGB-hist]
    R4 -- yes --> R6[Reuse matched canonical key]
    R5 --> R7{matched?}
    R7 -- yes --> R6
    R7 -- no --> R8[Create canon = sha256_norm(maskROI)\nand add to dictionary]

    R6 --> RH0
    R8 --> RH0

    RH0{yoloHealOnly?} -- no --> RH4[Maskable immediately]
    RH0 -- yes --> RH1[Legacy heal-only:\nload template, resize,\noffset search, template_alpha_matches_mask]
    RH1 --> RH2{healable_now?}
    RH2 -- yes --> RH4
    RH2 -- no --> RH3[Skip region as raw]

    RH4 --> RH5[OR seg mask into mask_union_u8]
    RH5 --> RH6[Mark det_masked, det_stitch_boxes,\nemit SEI with canonical hash path]
    RH6 --> B
    RH3 --> B

    B --> Z{all detections processed?}
    Z -- no --> B
    Z -- yes --> M1{pixelMode?}
    M1 -- yes --> END1[Finish pass 1]
    M1 -- no --> M2[Build fill_bgr via make_fill_frame_bgr]
    M2 --> M3{mask_union_u8 non-empty?}
    M3 -- no --> M5[Record paint_ms / masking_ms]
    M3 -- yes --> M4[For each masked detection:\napply_fill_with_feather_bbox]
    M4 --> M5
    M5 --> BI{buildIndexOnly?}
    BI -- yes --> BI2[Increment frameIdx and continue]
    BI -- no --> END1[Pass 1 complete;\nmasked frame ready for later recovery/encode]
```

Legend:
- `Mismatch`: the paper currently states behavior that does not match the implementation.
- `Clarify`: the paper may be directionally correct, but the wording is too strong, too generic, or leaves out important implementation details.

## Section 3

### Section 3.1 Key Abstractions

- `Mismatch`: The paper says Ref frames contain "up to `k = 3` detected-object regions" and that the prototype uses `k = 3` for compactness. The implementation does not enforce a top-3 cap; `offline_processor.cpp` currently emits as many gated regions as survive processing.
- `Clarify`: The paper uses the generic term `RMD`, while the implementation uses an `MSK1` payload format with explicit fields in `deployment/sei_parser.h` and `deployment/sei_parser.cpp`. If `RMD` is meant as the abstract concept and `MSK1` as the concrete wire format, say that explicitly.
- `Clarify`: The payload does contain a version tag, frame counter, PTS, frame-level mode flag, and region records, so the overall abstraction is fine. However, the actual schema is already specific enough to describe directly.
- `Clarify`: The region `path` field still exists in the payload format for backward compatibility, but in latent-key mode it is intentionally left empty and the client uses `template_id` only. The paper currently reads as if all region fields are always semantically active.

### Section 3.2 Design Principles

- `Mismatch`: "Compression-friendly masking" is described as using a fill chosen to approximate the frame's dominant tone. In the code, dominant-color masking is optional. The default mask color in `offline_processor.h` is still `green`, and dominant-color updates only happen when `maskColor == "dominant"`.
- `Mismatch`: "Fail-safe reconstruction" is presented as a global invariant of the implementation. In code, that invariant only holds when `yoloHealOnly` is enabled. The default configuration does not enforce heal-only gating, and there is also an explicit `yoloForceMaskAll` debug mode that intentionally breaks correctness.
- `Clarify`: The transport discussion is broader than the current implementation. The current offline prototype builds `MSK1` payloads and dumps `msk1_payloads.bin`; if in-band muxing is only partially implemented in the prototype, the paper should distinguish the conceptual design from the current evaluation artifact.

### Section 3.3 Offline Evaluation Prototype

- `Clarify`: The text says FFmpeg handles the full media pipeline. The implementation does use FFmpeg for controlled encoding, but the prototype logic is centered in `offline_processor.cpp` and the offline "client" is simulated in the same evaluation program rather than as a separate deployed client.
- `Clarify`: If the evaluation is intentionally an emulation of server and client behavior in one offline process, say that more directly. That makes later claims in Sections 4 and 6 easier to interpret.

## Section 4

### Section 4.1 Per-Frame Processing

- `Mismatch`: The paper says the server converts frames from `YUV` to `RGB` before inference. In the current offline prototype, frames are read by OpenCV as `BGR`, then converted to `RGB` in `YOLO_V8::PreProcess` in `deployment/inference.cpp`. If YUV is the intended live-system path, the paper should scope that statement to the live design rather than the current prototype.
- `Match with minor clarification`: The paper's statement about letterbox resizing to `640x640` matches the implementation. `deployment/main.cpp` sets `imgSize = {640, 640}`, and `deployment/inference.cpp` performs centered letterbox padding with value `114`.
- `Match`: The mask-coefficient vector is indeed 32-D, is L2-normalized in `offline_processor.cpp`, and cosine similarity reuse uses the default threshold `0.95`.
- `Mismatch`: The paper says geometric consistency rejects reuse when IoU or center-displacement checks fail, and frames then go `Raw`. The implementation does not enforce that as the general latent-key reuse rule. The actual behavior is:
- `Clarify`: There is a temporal preference for the previous template when overlap is high (`IoU >= 0.7`) and similarity is high enough.
- `Clarify`: There is a separate motion-triggered minting heuristic based on IoU, center distance in pixels, and area change.
- `Clarify`: Global latent-bank reuse is otherwise cosine-based and class-filtered, with optional size filtering, not a strict "IoU + center gate before reuse" pipeline.
- `Mismatch`: The paper says the server periodically refreshes all identity assignments by re-evaluating latent similarity from scratch. The implementation instead uses hybrid periodic minting via `latentSamplePeriod` plus optional skip/merge thresholds. That is not the same as a full identity refresh.
- `Mismatch`: The paper says the flat fill color is expressed in `YUV`. In practice, masking is applied on `BGR` frames and later encoded to `yuv420p`. This should be reworded unless you want to describe the conceptual encoder-side color space rather than the exact implementation.
- `Match`: The current text on edge feathering now matches the implementation well. The code does distance-based blending inside the mask boundary, which is consistent with the revised "edge feathering" wording.
- `Clarify`: Metadata emission is currently `MSK1` payload generation with version `4` and `frame_flags`, not just an abstract `RMD` record list.

### Section 4.2 Reliability Gating

- `Mismatch`: The paper states strict thresholds `mask IoU >= 0.995` and `spill <= 0.005`. The implementation defaults are different in `offline_processor.h`: `yoloHealMaskIouThr = 0.0` and `yoloHealMaskExtraThr = 0.02`. In other words, the default code path disables the IoU threshold and allows 2% spill.
- `Clarify`: The implementation also applies additional checks not described in the paper: alpha-masked RGB MSE, a small translation search, minimum alpha-mask coverage, and a recent-template fallback window. These materially affect whether a region is considered healable.
- `Clarify`: The paper says template availability means confirmed client-cache presence. In the offline prototype, availability is approximated by whether the template is already in the simulated pool and not newly minted for the current frame. There is no real asynchronous client acknowledgement path in this evaluator.
- `Mismatch`: The paper presents the heal-only policy as the system's central invariant. In implementation, that statement is only true for the `yoloHealOnly` mode, not for all operating modes.

### Section 4.3 Handling New Objects

- `Mismatch`: The paper says the server schedules template delivery over an out-of-band control channel and waits for client acknowledgement before later masking. The offline implementation mints the PNG immediately into `dictDir` and marks new templates as unrecoverable for the current frame, but there is no explicit OOB transport queue or acknowledgement state machine.
- `Clarify`: If this section is describing the intended live deployment rather than the current offline simulator, say so explicitly.

### Section 4.4 Server-Side Template Pool

- `Mismatch`: The paper says templates are versioned in the server-side pool. The current latent template bank tracks `id`, class, size, path, use count, minted frame, and last-used frame, but there is no explicit per-template version field in the current implementation.
- `Clarify`: The "larger than client cache" discussion is conceptually sound, but the current offline evaluator does not implement an actual bounded client cache with eviction.

### Section 4.5 Template Dictionary Population

- `Clarify`: This section reads as implementation detail, but it is really deployment discussion and future strategy. The text itself already hints at this. It should be clearly marked as design space / discussion rather than part of the implemented system.
- `Mismatch`: Several mechanisms described here are not implemented in the current evaluator, including frequency-ranked top-k delivery, cross-session promotion logic, crowd-sourced template promotion, and explicit client-cache-capacity-aware delivery decisions.

### Section 4.6 Pixel-Game Mode

- `Match`: The implementation does use template matching, Kalman filtering, and Lucas-Kanade sparse optical flow.
- `Clarify`: The paper's description is a simplified version of the actual implementation. The code also does periodic re-bootstrap, multi-template scanning, band selection, multi-peak extraction, and additional heuristics. The text is acceptable as a high-level summary, but it should not imply the algorithm is only "single template NCC + Kalman + sparse flow".
- `Clarify`: The paper says the highest normalized cross-correlation score is taken as the initial detection. In implementation, there can be multiple active templates and multiple peaks, especially in band-scan mode.

## Section 5

### Section 5.1 Data Preparation

- `Clarify`: The paper gives a concrete OWLv2 + DINOv2 + SAM2 data-prep pipeline, but I did not find a reproducible script in the repo that implements this exact pipeline. If this was done offline in notebooks or external tooling, the paper should say so.
- `Clarify`: Because this pipeline is central to reproducibility, it would help to state whether the labels were generated fully by scripts, by an external annotation workflow, or by a one-off manual process using those models.

### Section 5.2 Model Architecture and Training

- `Match`: The training scripts do support `epochs=300`, `imgsz=640`, and `batch=16`.
- `Mismatch`: The paper explicitly says training uses `AdamW` with cosine learning-rate scheduling. The current training scripts do not set either `optimizer=AdamW` or `cos_lr=True`, so this claim is not directly supported by the checked-in scripts.
- `Clarify`: The `70%/30%` train/validation split is plausible from the `train/` and `val/` dataset structure, but the exact split policy is not encoded in the training scripts themselves. If this is fixed in data preparation, say that.
- `Clarify`: The augmentation claims in the paper mostly line up with recorded run summaries in `record/final2/model_table_v4.md`, but they are not explicit in the training scripts. If you want the paper to be fully reproducible, cite the actual Ultralytics run arguments or logs.
- `Mismatch`: The optional self-training / pseudo-labeling step is described as part of the training pipeline, but I did not find an implementation of that step in the current repo.

### Section 5.3 Deployment

- `Match`: Export to ONNX and runtime inference through ONNX Runtime with CUDA support are reflected in `train_fc5.py`, `train_racing.py`, and `deployment/inference.cpp`.
- `Mismatch`: The paper says the model outputs object outlines as polygons and the server converts them into a binary pixel map. The current implementation does not use polygon outlines. It decodes dense prototype masks, thresholds them, and stores a raster `boxMask` directly.
- `Clarify`: The statement that both outputs come from a single forward pass is correct and well supported by the code.

## Section 6

### Section 6.1 Metadata Parsing

- `Match`: The implementation does have a concrete parser for versioned metadata with frame counter, PTS, frame-level mode flag, and region records in `deployment/sei_parser.cpp`.
- `Clarify`: The paper should connect `RMD` to the concrete `MSK1` packet format to avoid sounding more abstract than the current implementation actually is.

### Section 6.2 Reconstruction (Stitching)

- `Match`: The implementation resizes templates with nearest-neighbor interpolation and performs alpha-based overlay, matching the paper.
- `Clarify`: The current "client" is an offline reconstruction pass inside `offline_processor.cpp`, not a standalone runtime client. That is fine for evaluation, but the text should make the prototype boundary explicit.
- `Mismatch`: The paper says Raw regions are displayed as-is and implies that non-reconstructable regions are simply left unmodified by the server. In the current non-heal-only modes, the server can still paint masked regions that the simulated client will not reconstruct if they are newly minted or otherwise unavailable. This is another reason to either scope claims to heal-only mode or state the evaluation mode explicitly.

### Section 6.3 Caching and Robustness

- `Mismatch`: The paper still contains a TODO for fallback thresholds. The implementation does have a concrete fallback heuristic in the offline client simulator: it reuses the most recent recoverable template of the same class when the boxes overlap with `IoU >= 0.5`.
- `Clarify`: The paper describes a persistent client cache updated over an out-of-band channel. The evaluator instead uses disk-backed template files and an in-process simulated cache (`last_client_tpls_by_class`) for fallback behavior. If this is meant to model the intended client, say that.
- `Clarify`: The statement that reconstruction is just a hash-table lookup plus alpha compositing is true at the conceptual level, but the offline evaluator also performs disk reads and some per-frame bookkeeping that a real client would ideally avoid.

## Suggested Framing Changes

- For Sections 3-4 and 6, consistently distinguish:
- `Implemented in current offline prototype`
- `Intended live-system design`
- For Sections 4.2 and 6.2, explicitly state whether the reported experiments use `heal-only` mode. Many of the strongest correctness claims only hold under that mode.
- For Section 4.5, move the content into discussion/future deployment strategy unless you plan to implement and evaluate those mechanisms.
- For Section 5.1-5.2, either point to the exact scripts / logs that realize the training and labeling pipeline or soften the wording to "we used a workflow based on ..." rather than presenting every step as fully reproducible from the current repo.
