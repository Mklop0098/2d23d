"""Dựng hình khối bằng Hunyuan3D-2mini-Turbo trên CPU (chỉ hình khối, không màu).

Đo trên i5-12400: ảnh mã hoá 518 px, 5 bước -> khoảng 50 giây, RAM khoảng 9 GiB.
Ảnh mã hoá 1022 px (mặc định của Hunyuan) chậm gấp 5 lần và làm lưng tượng thành
một tấm phẳng, nên ở đây cố định 518 px.
"""
import sys
import threading
from pathlib import Path

import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / 'models' / 'hunyuan3d-2mini'
DIT = MODEL_DIR / 'hunyuan3d-dit-v2-mini-turbo'
VAE = MODEL_DIR / 'hunyuan3d-vae-v2-mini-turbo'
ENCODER_SIZE = 518
_pipeline = None
_load_lock = threading.Lock()


def available():
    return all((folder / name).is_file() for folder in (DIT, VAE)
               for name in ('config.yaml', 'model.fp16.safetensors')) and \
        (ROOT / 'Hunyuan3D-2' / 'hy3dgen').is_dir()


def _load():
    global _pipeline
    with _load_lock:
        if _pipeline is not None:
            return _pipeline
        if not available():
            raise ValueError('Thiếu Hunyuan3D-2 hoặc trọng số trong models/hunyuan3d-2mini.')
        sys.path.insert(0, str(ROOT / 'Hunyuan3D-2'))
        from torchvision import transforms
        from hy3dgen.shapegen.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dgen.shapegen.models import ShapeVAE
        pipe = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
            str(DIT / 'model.fp16.safetensors'), str(DIT / 'config.yaml'),
            device='cpu', dtype=torch.float32, use_safetensors=True)
        # Tệp DiT kèm VAE thường; thay bằng VAE turbo để giải mã nhanh (FlashVDM).
        pipe.vae = ShapeVAE.from_single_file(
            str(VAE / 'model.fp16.safetensors'), str(VAE / 'config.yaml'),
            device='cpu', dtype=torch.float32, use_safetensors=True)
        pipe.vae.enable_flashvdm_decoder(enabled=True, adaptive_kv_selection=True,
                                         topk_mode='mean', mc_algo='mc')
        encoder = pipe.conditioner.main_image_encoder
        encoder.transform = transforms.Compose([
            transforms.Resize(ENCODER_SIZE, transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(ENCODER_SIZE),
            transforms.Normalize(mean=encoder.mean, std=encoder.std)])
        encoder.num_patches = (ENCODER_SIZE // 14) ** 2 + 1
        _pipeline = pipe
        return pipe


def generate(rgba_image, octree_resolution=256, seed=1234, max_faces=200000, steps=5):
    """Trả (lưới giảm, lưới đầy đủ) trong hệ Hunyuan: Y lên trên, mặt trước nhìn về +Z.

    Lưới đầy đủ (~500k mặt) giữ chi tiết Hunyuan dựng được; nó chỉ dùng để bake normal map
    và bóng hốc lên lưới giảm, vì tạo UV trên lưới đầy đủ quá chậm.
    """
    pipe = _load()
    with torch.inference_mode():
        mesh = pipe(image=rgba_image, num_inference_steps=steps, octree_resolution=octree_resolution,
                    num_chunks=20000, generator=torch.manual_seed(int(seed)),
                    output_type='trimesh', enable_pbar=False)[0]
    from hy3dgen.shapegen import FloaterRemover, DegenerateFaceRemover, FaceReducer
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    full = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.faces), process=True)
    if len(full.faces) > max_faces:
        mesh = FaceReducer()(full.copy(), max_facenum=max_faces)
    reduced = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.faces), process=True)
    return reduced, full


def to_triposr_frame(mesh, reference, like=None):
    """Đưa lưới Hunyuan về hệ TripoSR (Z lên trên, camera ảnh ở +X).

    Trục: TripoSR (x, y, z) = Hunyuan (z, x, y) — phép hoán vị vòng, không lật gương.
    Tỉ lệ đồng nhất theo chiều cao lưới TripoSR, tâm đặt trùng tâm khung bao TripoSR,
    để tra màu trường triplane của TripoSR đúng chỗ.
    """
    like = mesh if like is None else like  # lưới quyết định phép biến đổi (dùng chung cho cả hai lưới)
    base = np.asarray(like.vertices)[:, [2, 0, 1]]
    ref = np.asarray(reference.vertices)
    scale = np.ptp(ref[:, 2]) / max(np.ptp(base[:, 2]), 1e-6)
    offset = (ref.min(0) + ref.max(0)) / 2 - (base.min(0) + base.max(0)) / 2 * scale
    vertices = np.asarray(mesh.vertices)[:, [2, 0, 1]] * scale + offset
    return trimesh.Trimesh(vertices, np.asarray(mesh.faces), process=False)
