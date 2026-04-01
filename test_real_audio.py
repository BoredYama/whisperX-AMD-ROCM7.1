"""Test with the actual video file to reproduce & isolate the segfault."""
import sys, os, gc
import numpy as np
import torch

print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(flush=True)

AUDIO_PATH = "/home/sagar/Work/Karen 3 Final.mp4"

# --- Step 1: Load audio ---
print("=== Step 1: Loading audio ===", flush=True)
import whisperx
audio = whisperx.load_audio(AUDIO_PATH)
print(f"  Audio loaded: {len(audio)/16000:.1f}s, dtype={audio.dtype}", flush=True)

# --- Step 2: Test direct faster-whisper (non-batched, small chunk) ---
print("\n=== Step 2: Direct faster-whisper on first 30s ===", flush=True)
try:
    from faster_whisper import WhisperModel
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    chunk = audio[:16000*30]  # first 30 seconds only
    segments, info = model.transcribe(chunk, beam_size=5, language="en")
    segments = list(segments)
    print(f"  OK: {len(segments)} segments", flush=True)
    for s in segments[:3]:
        print(f"    [{s.start:.1f}-{s.end:.1f}] {s.text[:80]}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
except Exception as e:
    print(f"  FAILED: {e}", flush=True)
    import traceback; traceback.print_exc()

print(f"  Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB\n", flush=True)
torch.cuda.reset_peak_memory_stats()

# --- Step 3: WhisperX batch_size=1 (minimal) ---
print("=== Step 3: WhisperX batch_size=1 ===", flush=True)
try:
    wx_model = whisperx.load_model("large-v3", device="cuda", compute_type="float16",
                                    language="en")
    result = wx_model.transcribe(audio, batch_size=1)
    print(f"  OK: {len(result['segments'])} segments", flush=True)
    del wx_model
    gc.collect()
    torch.cuda.empty_cache()
except Exception as e:
    print(f"  FAILED: {e}", flush=True)
    import traceback; traceback.print_exc()

print(f"  Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB\n", flush=True)
torch.cuda.reset_peak_memory_stats()

# --- Step 4: WhisperX batch_size=4 ---
print("=== Step 4: WhisperX batch_size=4 ===", flush=True)
try:
    wx_model = whisperx.load_model("large-v3", device="cuda", compute_type="float16",
                                    language="en")
    result = wx_model.transcribe(audio, batch_size=4)
    print(f"  OK: {len(result['segments'])} segments", flush=True)
    del wx_model
    gc.collect()
    torch.cuda.empty_cache()
except Exception as e:
    print(f"  FAILED: {e}", flush=True)
    import traceback; traceback.print_exc()

print(f"  Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB", flush=True)
print("\n=== Done ===", flush=True)
