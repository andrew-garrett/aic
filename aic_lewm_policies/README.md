# LeWM Policies

This package contains policy utilities for collecting demonstration datasets in a format compatible with `stable-worldmodel` / `le-wm`.

## CollectDemos

`CollectDemos` wraps another policy and records each commanded step to:

- `pixels` (center camera RGB, HWC uint8), stored **downsampled** by default so collection stays fast (see below)
- `action` (fixed 14-dim vector)
- `proprio`
- `state`
- `reward` (zeros)
- `terminated`
- `episode_idx` (per-row episode id; same info as `ep_len`/`ep_offset`, for tooling parity with releases like `pusht_expert_train`)
- `step_idx` (per-row index within the episode, `0 .. L-1`)
- `ep_len`
- `ep_offset`

**Note on `episode_idx` / `step_idx`:** Official `swm inspect` examples often show these dense columns. They are **not** required for training: `stable_worldmodel.HDF5Dataset` builds episodes from **`ep_len` + `ep_offset`** only. LeWM’s Hydra config should keep `keys_to_load` to model inputs only (`pixels`, `action`, `proprio`, `state`) so these metadata columns are never loaded as model inputs. If you omit them from `keys_to_load`, extra HDF5 datasets are harmless.

Output path:

- `$STABLEWM_HOME/<AIC_DATASET_NAME>.h5`
- defaults:
  - `STABLEWM_HOME=~/.stable-wm`
  - `AIC_DATASET_NAME=aic_cable_train`

### Stored image resolution

LeWM training typically uses `img_size: 224` in Hydra; the dataloader resizes whatever is in `pixels`. To keep **collection** light, we resize the center camera **before** writing to HDF5:

- **`AIC_CAPTURE_PIXEL_MAX`** (default **`512`**) — max of height/width after resize; aspect ratio preserved (e.g. 1024×1152 → ~455×512).
- Set **`AIC_CAPTURE_PIXEL_MAX=0`** to store the **full** camera resolution (slowest / largest files).

Requires **OpenCV** (`cv2`) or **Pillow** for resizing when `AIC_CAPTURE_PIXEL_MAX > 0` (the AIC `pixi` workspace already includes OpenCV).

### Configure delegate (expert) policy

By default, `CollectDemos` delegates control to `aic_lewm_policies.ros.OracleDualInsert`.
`OracleDualInsert` is a ground-truth TF oracle with separate insertion profiles for:

- `SFP_MODULE -> SFP_PORT`
- `SC_PLUG -> SC_PORT`

Set a different delegate with:

```bash
export AIC_DEMO_DELEGATE_POLICY=aic_example_policies.ros.RunACT
```

You can also use:

```bash
export AIC_DEMO_DELEGATE_POLICY=aic_example_policies.ros.CheatCode
```

`CheatCode` relies on ground-truth TF and is not valid for evaluation without those signals.

### Run

```bash
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_lewm_policies.ros.CollectDemos
```

### Validate dataset

```bash
swm inspect aic_cable_train
```

### Inspect HDF5 locally (no `swm` CLI)

From the repo, with `h5py` available:

```bash
python3 aic_lewm_policies/scripts/inspect_swm_h5.py "$STABLEWM_HOME/$AIC_DATASET_NAME.h5" --sample-pixels
```

This prints dataset shapes, `ep_len` / `ep_offset` per episode, and basic numeric stats.
