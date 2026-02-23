python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

if torch.cuda.is_available():
    x = torch.randn(2, 16000, device="cuda", dtype=torch.float32)
    w = torch.hann_window(400, device="cuda")
    y = torch.stft(x, n_fft=512, hop_length=160, win_length=400, window=w, return_complex=True)
    print("stft ok:", y.shape)
