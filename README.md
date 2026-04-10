<h1 align="center">WhisperX (AMD ROCm Fork)</h1>

<p align="center">
  <strong>🔴 AMD/ROCm 7.2 Compatible — Linux Only</strong>
</p>

<img width="1216" align="center" alt="whisperx-arch" src="https://raw.githubusercontent.com/m-bain/whisperX/refs/heads/main/figures/pipeline.png">

This is an AMD/ROCm port of [WhisperX](https://github.com/m-bain/whisperX) — fast automatic speech recognition (70x realtime with large-v2) with word-level timestamps and speaker diarization.

- ⚡️ Batched inference for 70x realtime transcription using whisper large-v2
- 🪶 [faster-whisper](https://github.com/guillaumekln/faster-whisper) backend, requires <8GB GPU memory for large-v2 with beam_size=5
- 🎯 Accurate word-level timestamps using wav2vec2 alignment
- 👯‍♂️ Multispeaker ASR using speaker diarization from [pyannote-audio](https://github.com/pyannote/pyannote-audio)
- 🗣️ VAD preprocessing, reduces hallucination & batching with no WER degradation
- 🔴 **AMD GPU support** via ROCm 7.2

---

## Prerequisites

- **Linux** (Ubuntu 22.04 or 24.04 recommended)
- **AMD GPU** (RDNA2/RDNA3 — e.g. RX 6000/7000 series, or Instinct MI series)
- **ROCm 7.2** installed and working
- **Python 3.10–3.12** (ROCm PyTorch wheels are not available for Python 3.13+)

### Verify ROCm Installation

Before proceeding, ensure ROCm is correctly installed:

```bash
# Check ROCm version
cat /opt/rocm/.info/version

# Verify GPU detection
rocm-smi

# Verify HIP runtime
hipconfig --full
```

If any of these fail, install ROCm 7.2 following the [official AMD guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/).

---

## Setup ⚙️

### Step 1: Create a Conda Environment

```bash
# Create a new conda environment with Python 3.12
conda create -n whisperx python=3.12 -y

# Activate the environment
conda activate whisperx
```

### Step 2: Install System Dependencies

```bash
# Install ffmpeg (required for audio processing)
conda install -c conda-forge ffmpeg -y
```

### Step 3: Install PyTorch with ROCm

Install PyTorch and torchaudio from the official ROCm wheel index:

```bash
# For ROCm 7.2:
pip install torch==2.11.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/rocm7.2

# For ROCm 7.12 Preview:
pip install torch torchaudio \
  --index-url https://download.pytorch.org/whl/rocm7.12
```

**Verify PyTorch detects your AMD GPU:**

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

> **Note:** PyTorch's ROCm backend maps the CUDA API — `torch.cuda.is_available()` returns `True` on ROCm, and `torch.device("cuda")` works correctly. This is by design.

### Step 4: Install CTranslate2 with ROCm Support

The standard `ctranslate2` package from PyPI is **CUDA-only**. You must install the ROCm build from the [CTranslate2 GitHub Releases](https://github.com/OpenNMT/CTranslate2/releases) page.

```bash
# Download the ROCm wheel for your Python version from:
# https://github.com/OpenNMT/CTranslate2/releases
#
# Example (adjust filename for your Python version and ROCm version):
pip install ctranslate2-<version>+rocm<rocm_version>-cp312-cp312-manylinux_2_17_x86_64.whl
```

> **Important:** Make sure the CTranslate2 wheel matches your Python version (cp310/cp311/cp312) and ROCm version.

### Step 5: Install WhisperX

```bash
# Clone this repository
git clone https://github.com/m-bain/whisperX.git
cd whisperX

# Install whisperx and remaining dependencies
pip install -e .
```

### Step 6: (Optional) Speaker Diarization Setup

To enable speaker diarization, you need a Hugging Face access token:

1. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Accept the user agreement for [speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
3. Pass the token via `--hf_token YOUR_TOKEN` when running whisperx

---

## Quick Start

### Verify Installation

```bash
python -c "
import torch
import whisperx
print('PyTorch version:', torch.__version__)
print('ROCm available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
print('WhisperX imported successfully!')
"
```

### Command Line Usage

```bash
# Basic transcription (uses 'small' model by default)
whisperx path/to/audio.wav

# Use a larger model for better accuracy
whisperx path/to/audio.wav --model large-v2 --batch_size 4

# With speaker diarization
whisperx path/to/audio.wav --model large-v2 --diarize --hf_token YOUR_HF_TOKEN

# With word highlighting in SRT output
whisperx path/to/audio.wav --highlight_words True

# Specify language explicitly
whisperx path/to/audio.wav --model large-v2 --language de

# Run on CPU instead of GPU
whisperx path/to/audio.wav --compute_type int8 --device cpu
```

### Python Usage

```python
import os
# Must be set before importing torch on AMD GPUs
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

import whisperx
import gc
import torch
from whisperx.diarize import DiarizationPipeline

device = "cuda"  # Works on ROCm — PyTorch maps CUDA API to HIP
audio_file = "audio.mp3"
batch_size = 16  # reduce if low on GPU mem
compute_type = "float16"  # change to "int8" if low on GPU mem

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("large-v2", device, compute_type=compute_type)
audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
print(result["segments"])  # before alignment

# Cleanup: synchronize GPU before deleting to avoid HIP race conditions
torch.cuda.synchronize()
del model
gc.collect()

# 2. Align whisper output (runs on GPU with flash attention)
model_a, metadata = whisperx.load_align_model(
    language_code=result["language"], device=device
)
result = whisperx.align(
    result["segments"], model_a, metadata, audio, device,
    return_char_alignments=False
)
print(result["segments"])  # after alignment

torch.cuda.synchronize()
del model_a
gc.collect()

# 3. Assign speaker labels (requires HF token)
diarize_model = DiarizationPipeline(token="YOUR_HF_TOKEN", device=device)
diarize_segments = diarize_model(audio)
result = whisperx.assign_word_speakers(diarize_segments, result)
print(result["segments"])  # segments with speaker IDs
```

---

## Troubleshooting

### GPU Not Detected

```bash
# Check if ROCm sees your GPU
rocm-smi

# Check PyTorch
python -c "import torch; print(torch.cuda.is_available())"
```

If `torch.cuda.is_available()` returns `False`:
- Ensure ROCm 7.1 is properly installed
- Ensure your GPU is supported (RDNA2/RDNA3/CDNA)
- Check that the `HSA_OVERRIDE_GFX_VERSION` environment variable is set if needed for your GPU (e.g., `export HSA_OVERRIDE_GFX_VERSION=11.0.0` for RX 7600 XT)

### Out of Memory Errors

Reduce memory usage with these options:
1. Lower batch size: `--batch_size 4`
2. Use a smaller model: `--model base` or `--model small`
3. Use lighter compute type: `--compute_type int8`

### CTranslate2 Errors

If you get `ctranslate2` import errors, ensure you installed the ROCm-specific wheel, **not** the PyPI version:

```bash
pip uninstall ctranslate2
# Then reinstall from the GitHub releases ROCm wheel
```

---

## Technical Details

For details on batching, alignment, VAD effects, and alignment model choices, see the [WhisperX paper](https://www.robots.ox.ac.uk/~vgg/publications/2023/Bain23/bain23.pdf).

### Key Transcription Notes

1. Transcription uses `--without_timestamps True` for single-pass batching
2. VAD-based segment transcription reduces WER and enables accurate batching
3. `--condition_on_prev_text` defaults to `False` to reduce hallucination

---

## Limitations ⚠️

- Words without dictionary characters (e.g., "2014." or "£13.60") cannot be aligned
- Overlapping speech is not handled well
- Diarization is not perfect
- Language-specific wav2vec2 models are required for alignment
- **ROCm support is Linux-only** — no Windows or macOS GPU support

---

## Troubleshooting AMD/ROCm Issues

### 1. "torchcodec is not installed correctly" Warning
You may see a large warning telling you `torchcodec` cannot load `libtorchcodec_core7.so` from pyannote:
- **Cause:** The `torchcodec` library is meant to speed up audio loading, but it lacks AMD ROCm linux wheels.
- **Solution:** You can safely ignore this. The pipeline gracefully falls back to using standard `torchaudio`. If the warning logs bother you, uninstall it: `pip uninstall torchcodec -y`.

### 2. First Run Is Slower Than Subsequent Runs
- **Cause:** On the very first run, ROCm's AOTriton JIT compiler needs to compile flash attention kernels for wav2vec2 alignment. These are cached to `~/.triton/cache/` and reused forever after.
- **Solution:** This is a one-time cost (~1-2 min extra). All subsequent runs will be fast. If you need to reset the cache: `rm -f ~/.cache/whisperx_rocm_kernels_compiled`

### 3. Transcript output is one giant long paragraph
- **Solution:** You didn't give formatting rules to the CLI! To force clean 2-line subtitle boxes, append:
  `--max_line_width 42 --max_line_count 2`

### 4. Whisper Hallucinates or Gets Stuck Repeating the Same Word
- **Cause:** Giving the CLI arguments that strip context or force greedy decoding (like `--beam_size 1` or `--condition_on_previous_text False`).
- **Solution:** Use quality-preserving arguments: 
  `--beam_size 5 --condition_on_previous_text True --compression_ratio_threshold 2.0`

### 5. "xnack 'Off' was requested" Warning
- **Cause:** XNACK is a memory-paging feature for datacenter Instinct GPUs. Consumer RDNA cards don't support it.
- **Solution:** Completely harmless — ignore it.

## Acknowledgements 🙏

This is a ROCm port of [WhisperX](https://github.com/m-bain/whisperX) by Max Bain.

- Original work supported by [VGG (Visual Geometry Group)](https://www.robots.ox.ac.uk/~vgg/) and the University of Oxford
- Built on [OpenAI's Whisper](https://github.com/openai/whisper)
- Alignment code from [PyTorch forced alignment tutorial](https://pytorch.org/tutorials/intermediate/forced_alignment_with_torchaudio_tutorial.html)
- VAD & Diarization: [pyannote-audio](https://github.com/pyannote/pyannote-audio), [silero-vad](https://github.com/snakers4/silero-vad)
- Backend: [faster-whisper](https://github.com/guillaumekln/faster-whisper) and [CTranslate2](https://github.com/OpenNMT/CTranslate2)

## Citation

```bibtex
@article{bain2022whisperx,
  title={WhisperX: Time-Accurate Speech Transcription of Long-Form Audio},
  author={Bain, Max and Huh, Jaesung and Han, Tengda and Zisserman, Andrew},
  journal={INTERSPEECH 2023},
  year={2023}
}
```
