"""Diagnostic script to isolate where the segfault occurs."""
import sys
import os
import torch
import ctranslate2
import numpy as np

print(f"PyTorch: {torch.__version__}")
print(f"CTranslate2: {ctranslate2.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"HSA_OVERRIDE_GFX_VERSION={os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'not set')}")
print()

# --- Test 1: CTranslate2 model load ---
print("=== Test 1: Loading whisper large-v3 model via faster-whisper ===")
try:
    from faster_whisper import WhisperModel
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    print("Model loaded OK")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)

# --- Test 2: Small inference ---
print("\n=== Test 2: Small inference (5s silence) ===")
try:
    # Generate 5 seconds of silence
    audio = np.zeros(16000 * 5, dtype=np.float32)
    segments, info = model.transcribe(audio, beam_size=5, language="en")
    segments = list(segments)  # force evaluation
    print(f"Inference OK, {len(segments)} segments, language={info.language}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# --- Test 3: Batched inference via whisperx ---
print("\n=== Test 3: WhisperX batched inference ===")
try:
    import whisperx
    audio = np.zeros(16000 * 30, dtype=np.float32)  # 30s silence
    wx_model = whisperx.load_model("large-v3", device="cuda", compute_type="float16")
    result = wx_model.transcribe(audio, batch_size=4)
    print(f"WhisperX inference OK, {len(result['segments'])} segments")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n=== All tests passed ===")
del model
if 'wx_model' in dir():
    del wx_model
torch.cuda.empty_cache()
print(f"Peak VRAM used: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
