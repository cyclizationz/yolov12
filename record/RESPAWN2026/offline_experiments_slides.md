---
marp: true
paginate: true
---

# Offline Experiments 1, 3, 4, 5

**Question:** where does RESPAWN save bandwidth, which knobs matter, and when does the method stop helping?

**Split:** Exp1 = fixed-VBV RD (authoritative rate sweep). Exp3/5 = uniform **CRF23 + open GOP** intake. Exp4 = ablations from existing artifacts. **Exp2 (online pipeline) deferred** — not runnable in the current offline tranche.

---

# Exp1: Fixed-VBV RD Frontier

**Protocol:** `medium` + open GOP + VBV 8–24 Mbps; matched `pure_streaming` reference.

| Game | 8 Mbps | 16 Mbps | 20 Mbps | All rates (25 pts) |
| --- | ---: | ---: | ---: | ---: |
| FC5 | −0.21% | −0.05% | +0.09% | **−0.01%** |
| FM6 | −0.39% | −0.51% | −0.43% | **−0.43%** |
| Mario | −0.52% | +9.41% | +16.78% | **+9.29%** |

**Takeaway:** Mario wins under fixed VBV when the baseline spends bits on brick detail; FC5/FM6 sit near break-even when VBV pegs both arms.

Tables: `figures/exp1_refresh/exp1_fixed_budget_table.tex`

---

# Exp4: Feather Sweep (Strongest Result)

| Game | Best feather | Saving % | Recovered SSIM |
| --- | ---: | ---: | ---: |
| FC5 | 4 px | 6.89 | 0.936 |
| FM6 | 4 px | 15.53 | 0.954 |

**Takeaway:** Moderate feather (4–8 px) improves bitrate and quality; 16 px buys marginal SSIM at higher cost. Default: **4 px**.

---

# Exp4: Fill Strategy

| Trace | Fill | Saving % | Rec SSIM | Status |
| --- | --- | ---: | ---: | --- |
| FC5 | black | 5.33 | 0.920 | supported |
| FC5 | dominant | 5.27 | 0.927 | supported (preferred) |
| FC5 | inpaint | −2.53 | 0.899 | CRF23 fc5_00 only |
| FC5 | blur | N/A | N/A | no saved run |
| Mario 5s | dominant | 20.46 | 0.965 | short trace |

**Takeaway:** Dominant color is the default; inpaint improves quality but hurts bandwidth on FC5.

---

# Exp4: Matching Strategy (First Pass)

| Game | Matcher | BSP / saving | Ref ratio |
| --- | --- | ---: | ---: |
| FC5 | latent-key | 13.5% | 0.47 |
| FC5 | IoU-only | 15.1% | 0.12 |
| FM6 | latent-key | 9.4% | 0.97 |
| FM6 | pHash | 9.6% | 0.94 |

**Takeaway:** High Ref ratio alone does not guarantee net savings under VBV; churn and recovered quality differ by matcher.

---

# Exp4: Mario Path Ablation

CRF23 five-clip means, aligned with Exp3/Exp5 Mario intake:

| Strategy | BSP / saving | Rec SSIM |
| --- | ---: | ---: |
| Full pipeline | +30.2% | 0.918 |
| No optical flow | +25.6% | 0.928 |
| No Kalman | +30.2% | 0.918 |
| Template only | +30.9% | 0.921 |

**Takeaway:** Full pipeline matches the reported Mario CRF23 mean; no-flow clearly loses BSP. Kalman/template-only need quality/stability follow-up rather than a bandwidth-only claim.

---

# Exp3: CRF23 Net Savings (Photoreal Examples)

![FC5 CRF23 cumulative](exp3/fc5_00_23p0_cumulative_net_bytes.png)

**Caption:** `fc5_00` under CRF23 learned path — **+8.9%** net BSP (delivered video + MSK1).

**Scope:** Exp3 cache/RTT curves focus on **photoreal** clips (FC5, FM6). Mario omitted — pixel-game templates are assumed **fully known before session start**.

---

# Exp3: FM6 Cumulative Net Bytes

![FM6 CRF23 cumulative](exp3/fm6_04_23p0_cumulative_net_bytes.png)

**Caption:** `fm6_04` — **+16.3%** net BSP under CRF23 heal-only intake.

**Photoreal mean (CRF23):** FC5 **+6.2%** (5 clips), FM6 **+14.1%** (5 clips). Mario **+30.2%** reported in Exp5 only.

---

# Exp3: Synthetic Template-Delay Sensitivity

Exp3 cumulative curves use a **simulation-only** synthetic template model when MSK1 lacks template paths:

- RTT grid: 20 / 80 / 150 ms; delay: 0 / 2 / 5 RTT windows
- Charge 8192 B per template delivery; force Raw while unavailable

**Labeling:** sensitivity bounds on FC5/FM6, not measured client cache policy.

---

# Exp5: BSP vs Masked Area (All Clips)

![BSP vs masked area](exp5/bsp_vs_masked_area.png)

**Caption:** Points only — yellow = pixel art (Mario), blue = photoreal (FC5 circle, FM6 square).

Larger recurring regions create opportunity; Mario pixel bricks show the strongest positive BSP under CRF23.

---

# Exp5: BSP vs Ref Ratio (Photoreal Table)

Mario omitted: pixel templates assumed pre-known (Ref ratio saturates at 1.0).

| Game | Ref ratio | BSP (%) | Masked area (%) | Template reuse |
| --- | ---: | ---: | ---: | ---: |
| FC5 | 0.777 | 6.2 | 10.5 | 0.096 |
| FM6 | 0.949 | 14.1 | 39.9 | 0.490 |

LaTeX: `exp5/bsp_vs_ref_ratio.tex`

---

# Exp5: Per-GOP Savings CDF

![Per-GOP savings CDF](exp5/per_gop_savings_cdf.png)

**Caption:** Per-game CDF; GOP windows outside **[-50%, +50%]** dropped; x-axis clipped to that range.

Shows workload tails without extreme outlier GOPs dominating the plot.

---

# Combined Message

**Exp1 (VBV):** cite for rate-point claims — Mario **+9.3%** mean; FC5/FM6 ~0%.

**Exp4:** feather **4 px**; dominant fill default; inpaint not a bandwidth win; Mario CRF23 ablation now matches Exp5 protocol.

**Exp3 (photoreal):** FC5 **+6.2%**, FM6 **+14.1%** under CRF23; template-delay curves are synthetic sensitivity only.

**Exp5 (generality):** includes Mario **+30.2%** on CRF23 intake. Do not mix Exp1 VBV and Exp3/5 CRF numbers in one sentence.

**Exp2:** online pipeline test — deferred until live transport integration.
