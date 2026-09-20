# ASR benchmark summary

ASR-only comparison; no QA/NLI model is used.

## Environment

```json
{
  "python": "3.11.15",
  "platform": "Linux-5.15.0-191-generic-x86_64-with-glibc2.35",
  "ffmpeg": "/usr/bin/ffmpeg",
  "torch": "2.5.1+cu121",
  "torch_cuda": "12.1",
  "cuda_available": true,
  "gpu_name": "NVIDIA A10",
  "transformers": "5.13.1",
  "huggingface_hub": "1.5.0",
  "numpy": "2.4.6"
}
```

| Model | OK/Fail | P95 inference (s) | P95 RTF | Peak VRAM (MB) | Timestamp ceiling tIoU | Numeric recall | Retrieval H@1 | H@3 | H@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| parakeet_v3 | 39/0 | 5.927 | 0.035 | 2568 | 0.862 | 0.673 | 0.764 | 0.851 | 0.892 |
| medasr | 39/0 | 0.445 | 0.003 | 750 | 0.881 | 0.660 | 0.790 | 0.872 | 0.918 |
| whisper_large_v3 | 39/0 | 49.570 | 0.240 | 13998 | 0.843 | 0.596 | 0.790 | 0.856 | 0.903 |
