from pathlib import Path
import urllib.request
root = Path(__file__).resolve().parent
HY = 'https://huggingface.co/tencent/Hunyuan3D-2mini/resolve/main/'
DEPTH = 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/resolve/main/'
files = [
 ('https://huggingface.co/stabilityai/TripoSR/resolve/main/config.yaml', 'models/triposr/config.yaml'),
 ('https://huggingface.co/stabilityai/TripoSR/resolve/main/model.ckpt', 'models/triposr/model.ckpt'),
 ('https://huggingface.co/facebook/dino-vitb16/resolve/main/config.json', 'models/dino/config.json'),
 ('https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx', 'models/rembg/u2net.onnx'),
 # Tách nền sắc hơn u2net, nhất là viền chạm khắc.
 ('https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx',
  'models/rembg/birefnet-general-lite.onnx'),
 # Ước lượng độ sâu từ ảnh để cộng chi tiết nổi vào mặt trước.
 (DEPTH + 'config.json', 'models/depth/config.json'),
 (DEPTH + 'preprocessor_config.json', 'models/depth/preprocessor_config.json'),
 (DEPTH + 'model.safetensors', 'models/depth/model.safetensors'),
 # Hunyuan3D-2mini-Turbo (chỉ hình khối). Mã nguồn: git clone https://github.com/Tencent/Hunyuan3D-2
 (HY + 'hunyuan3d-dit-v2-mini-turbo/config.yaml', 'models/hunyuan3d-2mini/hunyuan3d-dit-v2-mini-turbo/config.yaml'),
 (HY + 'hunyuan3d-dit-v2-mini-turbo/model.fp16.safetensors', 'models/hunyuan3d-2mini/hunyuan3d-dit-v2-mini-turbo/model.fp16.safetensors'),
 (HY + 'hunyuan3d-vae-v2-mini-turbo/config.yaml', 'models/hunyuan3d-2mini/hunyuan3d-vae-v2-mini-turbo/config.yaml'),
 (HY + 'hunyuan3d-vae-v2-mini-turbo/model.fp16.safetensors', 'models/hunyuan3d-2mini/hunyuan3d-vae-v2-mini-turbo/model.fp16.safetensors'),
]
for url, name in files:
    path = root / name
    if path.exists():
        continue
    path.parent.mkdir(parents=True, exist_ok=True)
    print('Downloading', name, flush=True)
    temporary = path.with_suffix(path.suffix + '.part')
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(path)
    print('Saved', path.stat().st_size, flush=True)
