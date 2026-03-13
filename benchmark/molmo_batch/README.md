# Molmo Batch Benchmark

This folder contains batch benchmarking utilities to compare Molmo2 with VISTA on the same video QA prompts.

## Files

- `benchmark_molmo2.py`: run Molmo2 over a config of video+prompt test cases
- `benchmark_config.json`: test case list
- `compare_results.py`: generate markdown comparison report from Molmo and VISTA result JSON files

## Setup

Install required packages on your GPU machine:

```bash
pip install transformers==4.57.1 torch pillow einops torchvision accelerate decord2 molmo_utils
```

For smaller GPUs (optional 4-bit mode):

```bash
pip install bitsandbytes
```

## Run Molmo benchmark

```bash
python benchmark/molmo_batch/benchmark_molmo2.py \
  --config benchmark/molmo_batch/benchmark_config.json \
  --video-dir /path/to/videos \
  --max-tokens 512
```

### Optional (small GPU)

```bash
python benchmark/molmo_batch/benchmark_molmo2.py \
  --config benchmark/molmo_batch/benchmark_config.json \
  --video-dir /path/to/videos \
  --load-in-4bit
```

## Compare with VISTA

```bash
python benchmark/molmo_batch/compare_results.py \
  --molmo results/molmo2_results_YYYYMMDD_HHMMSS.json \
  --ours results/vista_results_YYYYMMDD_HHMMSS.json \
  --output results/comparison_report.md
```
