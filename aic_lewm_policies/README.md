# LeWM Policies

This package contains policy utilities for collecting demonstration datasets in a format compatible with `stable-worldmodel` / `le-wm`.

## CollectDemos

`CollectDemos` wraps another policy and records each commanded step to:

- `pixels` (center camera RGB, HWC uint8)
- `action` (fixed 14-dim vector)
- `proprio`
- `state`
- `reward` (zeros)
- `terminated`
- `ep_len`
- `ep_offset`

Output path:

- `$STABLEWM_HOME/<AIC_DATASET_NAME>.h5`
- defaults:
  - `STABLEWM_HOME=~/.stable-wm`
  - `AIC_DATASET_NAME=aic_cable_train`

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
