# E2 RF-DETR Small — RF-DETR 1.5.2 compatibility fix

Overlay the files in this patch on top of your existing `drone-flyby/` project.

## Why this patch is needed

Your required `torch==2.2.1+cu121` is not compatible with the Transformers-v5 stack used by newer RF-DETR releases. E2 is therefore pinned to:

```text
torch==2.2.1+cu121
torchvision==0.17.1+cu121
rfdetr==1.5.2
transformers==4.56.2
albumentations==1.4.24
```

RF-DETR 1.5.2 differs materially from newer releases:

- no `RFDETRSmall.from_checkpoint()`;
- legacy `.pth` checkpoints (`checkpoint.pth`, `checkpointNNNN.pth`, `checkpoint_best_*.pth`);
- no `augmentation_backend`, `skip_best_epochs`, `log_per_class_metrics`, or `scale_jitter` TrainConfig fields;
- `run_test=True` expects a `test/` dataset folder, which this blocked baseline intentionally does not create;
- the hosted-weight downloader recognizes `rf-detr-small.pth` by bare filename, so downloading into a custom cache needs an explicit wrapper;
- its training resize pipeline has a hard-wired random crop branch before `aug_config`; this patch disables that branch for the tiny-object baseline while retaining RF-DETR's official Albumentations augmentation machinery.

## Install the compatible stack

Inside the existing virtual environment:

```bash
pip uninstall -y rfdetr transformers albumentations
pip install \
  "rfdetr==1.5.2" \
  "transformers==4.56.2" \
  "albumentations==1.4.24"

pip check
```

Do **not** upgrade Torch; keep the challenge requirement.

## Prepare and train E2

```bash
python scripts/prepare_dataset.py
python scripts/download_weights.py --scenario E2
python scripts/check_e2_environment.py
python scripts/train_e2.py
```

The preflight script does not train. It verifies versions, the pretrained checkpoint, train/valid COCO exports, and that every training argument is accepted by RF-DETR 1.5.2's actual `TrainConfig`.

### Resume after interruption

RF-DETR 1.5.2 continuously writes:

```text
experiments/E2_rfdetr_small/logs/rfdetr/checkpoint.pth
```

Resume with the original target epoch count, for example:

```bash
python scripts/train_e2.py \
  --resume experiments/E2_rfdetr_small/logs/rfdetr/checkpoint.pth
```

You can also override resource-sensitive parameters without editing YAML:

```bash
python scripts/train_e2.py --batch-size 1 --grad-accum-steps 4
```

In RF-DETR 1.5.2, the DataLoader batch is `batch_size * grad_accum_steps` and the model then splits it into gradient-accumulation micro-batches. Thus `batch_size=2, grad_accum_steps=4` means an effective batch of 8 with GPU micro-batches of 2.

## Expected RF-DETR warnings

These warnings are normal for RF-DETR Small and are not download/training failures:

```text
Using a different number of positional encodings than DINOv2...
Using patch size 16 instead of 14...
```

RF-DETR then loads the full pretrained RF-DETR checkpoint, which is what E2 fine-tunes.

## Checkpoint selection

RF-DETR 1.5.2 chooses its native best checkpoint by mAP50-95. The patch keeps those files, then evaluates saved candidate checkpoints on the shared blocked validation set and selects the deployment checkpoint using the challenge metric `mAP@0.50`.
