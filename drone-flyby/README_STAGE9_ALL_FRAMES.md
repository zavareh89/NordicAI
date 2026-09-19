# Stage 9 update: train final E1/E2 on all labeled frames

## Why

With only ~25 consecutive frames, some legal challenge classes occur in only one or a
few frames. A temporally blocked holdout can therefore remove the only supervised
example of a class from training. That is worse for the final competition model than
losing an internal holdout, because the detector would never be fine-tuned on that
class at all.

The final baseline now defaults to:

```yaml
data:
  split:
    strategy: all_frames
```

The blocked split code is still available as `strategy: blocked_holdout` for diagnostic
experiments. It is not deleted.

## Class-by-frame report

Run:

```bash
python scripts/prepare_dataset.py
```

This now creates:

```text
prepared/class_by_frame.json
prepared/class_by_frame.csv
```

The JSON includes:

- frames containing each class;
- number of frames per class;
- total instances per class;
- singleton classes;
- classes absent from the provided dataset;
- class coverage of the selected split;
- classes that a candidate split would remove from training.

If `blocked_holdout` is selected and it removes any dataset-present class from training,
preparation now fails loudly and tells you to use `all_frames` or choose another
diagnostic split.

## No fake validation claim

`all_frames` means there is no independent held-out validation set.

- YOLO: training validation is disabled; the final `last.pt` checkpoint is selected.
- RF-DETR 1.5.2: the framework requires `valid/`, so `prepared/rfdetr/valid/` mirrors
  `train/` only for framework compatibility. Early stopping and validation-based
  checkpoint selection are disabled, and the final `checkpoint.pth` is selected.
- `metrics.json` explicitly states that held-out metrics were not computed.
- `scripts/evaluate.py` / offline evaluation refuses to report train-mirror metrics as
  held-out performance.

For final integration/scoring, use the official challenge local evaluator. If you want a
rough internal diagnostic, temporarily switch to `blocked_holdout`, prepare again, and
accept that rare-class coverage may be incomplete.

## Commands

```bash
# regenerate exact L0 data, all-frame training exports, split metadata,
# and class-by-frame reports
python scripts/prepare_dataset.py

# inspect rare/singleton classes
cat prepared/class_by_frame.json
# or open prepared/class_by_frame.csv

# train final all-data baselines
python scripts/train_e1.py
python scripts/train_e2.py
```
