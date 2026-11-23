"""Local CPU inference. All required model files are loaded from this folder."""
import os
import sys
import time
import json
import threading
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['U2NET_HOME'] = str(ROOT / 'models' / 'rembg')
os.environ['TRIPOSR_DINO_CONFIG'] = str(ROOT / 'models' / 'dino' / 'config.json')
sys.path.insert(0, str(ROOT / 'TripoSR'))

import psutil
import torch
import rembg
import trimesh
import math
import numpy as np
from scipy import ndimage
from PIL import Image, ImageOps
from tsr.system import TSR
from tsr.utils import resize_foreground
from tsr.bake_texture import bake_texture, rasterize_position_atlas
import detail
import hunyuan_shape
import surface_detail
from mesh_cleanup import clean_mesh

torch.set_num_threads(6)
_model = None
_background_session = None
_lock = threading.Lock()
# Triplane của TripoSR chỉ có 64 ô mỗi chiều; lưới dày hơn chủ yếu giúp chứa chi tiết từ ảnh.
QUALITY = {'Nhanh': 192, 'Chi tiết': 256, 'Tối đa': 384}
BACKGROUND_MODELS = ('birefnet-general-lite', 'u2net')  # ưu tiên mô hình sắc nét hơn
# Hunyuan dựng hình khối (đoán hông và lưng tốt hơn), TripoSR tô màu phần khuất.
SHAPE_MODELS = ('Hunyuan mini + TripoSR', 'TripoSR')
HUNYUAN_OCTREE = {'Nhanh': 192, 'Chi tiết': 256, 'Tối đa': 320}
HUNYUAN_FACES = {'Nhanh': 60000, 'Chi tiết': 100000, 'Tối đa': 150000}  # xatlas chậm khi lưới lớn
HUNYUAN_CELL = 2 * 0.87 / 128  # Hunyuan giữ được chi tiết mịn hơn triplane 64 của TripoSR


def generate(image_path, remove_background=True, quality='Chi tiết', report=None,
             texture=True, threshold=25.0, relief=1.5, smooth=True, photo_texture=True,
             shape_model=SHAPE_MODELS[0], seed=1234, normal_detail=True, mesh_cleanup=True):
    if not image_path:
        raise ValueError('Hãy chọn ảnh JPG hoặc PNG.')
    if quality not in QUALITY:
        raise ValueError('Mức chất lượng không hợp lệ.')
    if not 5 <= float(threshold) <= 50:
        raise ValueError('Ngưỡng bề mặt phải nằm trong khoảng 5–50.')
    if not 0 <= float(relief) <= 2:
        raise ValueError('Độ nổi chi tiết phải nằm trong khoảng 0–2.')
    if shape_model not in SHAPE_MODELS:
        raise ValueError('Mô hình hình khối không hợp lệ.')
    with _lock:
        return _generate(image_path, remove_background, quality, report or (lambda text: None),
                         texture, float(threshold), float(relief), bool(smooth), bool(photo_texture),
                         shape_model, int(seed), bool(normal_detail), bool(mesh_cleanup))


def _texture_image(values):
    """Mảng atlas (hàng 0 = v=0 theo OpenGL) thành ảnh texture xuất GLB."""
    image = Image.fromarray((np.clip(values[..., :3], 0, 1) * 255).astype(np.uint8))
    return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).convert('RGB')


def _orient_outward(mesh):
    """fix_normals có thể lật cả lưới vào trong khi lưới hở; kiểm tra lại theo tâm vật."""
    mesh.fix_normals()
    outward = np.einsum('ij,ij->i', mesh.vertex_normals, mesh.vertices - mesh.vertices.mean(0))
    if (outward > 0).mean() < 0.5:
        mesh.invert()


def _background_session_for():
    import onnxruntime
    from rembg.sessions import sessions_class
    options = onnxruntime.SessionOptions()
    # BiRefNet cần ~5 GB khi chạy; tắt arena để trả RAM lại ngay sau khi tách nền.
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    for name in BACKGROUND_MODELS:
        if (ROOT / 'models' / 'rembg' / f'{name}.onnx').is_file():
            session_class = next(c for c in sessions_class if c.name() == name)
            return name, session_class(name, options, ['CPUExecutionProvider'])
    raise ValueError('Thiếu mô hình tách nền trong models/rembg.')


def _generate(image_path, remove_background, quality, report, texture, threshold,
              relief, smooth, photo_texture, shape_model, seed, normal_detail, mesh_cleanup):
    global _model, _background_session
    started = time.perf_counter()
    process = psutil.Process()
    peak = [process.memory_info().rss]
    stop = threading.Event()

    def monitor():
        while not stop.wait(0.25):
            peak[0] = max(peak[0], process.memory_info().rss)

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    folder = ROOT / 'results' / (time.strftime('%Y%m%d-%H%M%S') + '-' + uuid4().hex[:8])
    folder.mkdir(parents=True)
    try:
        report('Đang đọc ảnh')
        with Image.open(image_path) as original:
            if original.format not in ('JPEG', 'PNG'):
                raise ValueError('Chỉ hỗ trợ ảnh JPG và PNG.')
            image = ImageOps.exif_transpose(original).convert('RGBA')
        image.thumbnail((2048, 2048))
        image.save(folder / 'original.png')
        if remove_background:
            report('Đang tách nền')
            if _background_session is None:
                _background_session = _background_session_for()
            image = rembg.remove(image, session=_background_session[1])
        if image.getextrema()[3][0] < 255:
            if image.getextrema()[3][1] == 0:
                raise ValueError('Ảnh sau tách nền hoàn toàn trong suốt. Hãy thử tắt tách nền.')
            image = resize_foreground(image, 0.85)
        rgba = image
        alpha = np.asarray(image)[..., 3]
        background = Image.new('RGBA', image.size, (128, 128, 128, 255))
        image = Image.alpha_composite(background, image).convert('RGB')
        image.save(folder / 'input.png')
        if alpha.min() < 255:
            mask = alpha > 127
        else:  # ảnh đã có nền xám sạch
            mask = np.abs(np.asarray(image, np.int16) - 128).sum(-1) > 6
        mask = ndimage.binary_opening(mask, iterations=1)
        use_hunyuan = shape_model == SHAPE_MODELS[0]
        shape_note = None
        if use_hunyuan and not hunyuan_shape.available():
            use_hunyuan, shape_note = False, 'Thiếu Hunyuan, đã dùng TripoSR.'
        report('Đang nạp TripoSR vào RAM')
        if _model is None:
            _model = TSR.from_pretrained(str(ROOT / 'models/triposr'), 'config.yaml', 'model.ckpt')
            _model.eval().to('cpu')
            _model.renderer.set_chunk_size(8192)
        report('Đang dựng hình bằng CPU')
        with torch.inference_mode():
            codes = _model([image], device='cpu')
            report('Đang tạo lưới và màu bề mặt')
            # Với Hunyuan, lưới TripoSR thô chỉ dùng làm mốc kích thước và vị trí.
            mesh = _model.extract_mesh(codes, True, resolution=128 if use_hunyuan else QUALITY[quality],
                                      threshold=threshold,
                                      progress=lambda fraction: report(f'Đang lấy mẫu hình khối {fraction:.0%}'))[0]
        if len(mesh.faces) == 0:
            raise ValueError('Không tạo được bề mặt từ ảnh này. Hãy thử ảnh vật thể rõ hơn.')
        _orient_outward(mesh)
        timings = {}
        cell = 2 * 0.87 / 64
        if use_hunyuan:
            report('Đang dựng hình khối bằng Hunyuan mini (khoảng 1 phút, lần đầu thêm ~20 giây nạp)')
            tick = time.perf_counter()
            if alpha.min() == 255:  # không tách nền: dùng mặt nạ làm kênh alpha
                rgba = image.convert('RGB').copy()
                rgba.putalpha(Image.fromarray((mask * 255).astype(np.uint8)))
            shape, shape_full = hunyuan_shape.generate(
                rgba, octree_resolution=HUNYUAN_OCTREE[quality], seed=seed,
                max_faces=HUNYUAN_FACES[quality])
            if len(shape.faces) == 0:
                raise ValueError('Hunyuan không tạo được bề mặt từ ảnh này.')
            reference = mesh
            mesh = hunyuan_shape.to_triposr_frame(shape, reference, like=shape_full)
            _orient_outward(mesh)
            high = hunyuan_shape.to_triposr_frame(shape_full, reference, like=shape_full)
            _orient_outward(high)
            del shape, shape_full
            cell = HUNYUAN_CELL
            timings['hunyuan_seconds'] = round(time.perf_counter() - tick, 2)
        elif smooth and not mesh_cleanup:
            # Taubin bỏ nốt sần của marching cubes mà không làm co hình khối.
            trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=6)
        cleanup_info = None
        if mesh_cleanup:
            report('Đang làm sạch lưới bằng Open3D')
            tick = time.perf_counter()
            mesh, cleanup_info = clean_mesh(mesh, smooth=smooth, iterations=2 if use_hunyuan else 6)
            _orient_outward(mesh)
            timings['mesh_cleanup_seconds'] = round(time.perf_counter() - tick, 2)
        if not use_hunyuan:
            high = mesh.copy()  # TripoSR: pháp tuyến gốc trước khi cộng chi tiết từ ảnh
        relief_info = None
        relief_map = None
        if relief > 0 or (texture and photo_texture):
            report('Đang ước lượng độ sâu từ ảnh')
            tick = time.perf_counter()
            try:
                disparity = detail.estimate_depth(image, ROOT / 'models' / 'depth')
                relief_map = detail.Relief(mesh, disparity, mask, cell=cell)
            except ValueError as error:
                relief_map = None
                relief_info = {'skipped': str(error)}
            if relief_map is not None and relief > 0:
                report('Đang cộng chi tiết nổi vào bề mặt')
                mesh, mean_offset = detail.add_relief(mesh, relief_map, relief)
                _orient_outward(mesh)
                relief_map.update_depth(mesh)
                relief_info = {'strength': relief, 'mean_offset': round(mean_offset, 5),
                               'depth_fit': [round(x, 5) for x in relief_map.fit],
                               'cell_px': round(relief_map.cell_px, 2)}
            timings['detail_seconds'] = round(time.perf_counter() - tick, 2)
        # Bake while vertices are still in the AI model's coordinate system.
        textured_mesh = clay_mesh = None
        texture_resolution = 2048 if quality != 'Nhanh' else 1024
        if texture:
            report('Đang tạo UV và texture màu')
            tick = time.perf_counter()
            baked = bake_texture(mesh, _model, codes[0], texture_resolution)
            colors = baked['colors']
            if photo_texture and relief_map is not None:
                report('Đang chiếu màu ảnh gốc lên mặt trước')
                fill = None
                if use_hunyuan:  # lưới Hunyuan không trùng trường màu TripoSR: lan màu ảnh
                    fill = (np.asarray(mesh.vertices),
                            detail.spread_photo_colors(mesh, relief_map, image))
                colors = detail.project_photo(colors, baked['positions'], baked['normals'],
                                              relief_map, image, fill=fill)
            timings['texture_seconds'] = round(time.perf_counter() - tick, 2)
            normal_image = clay_image = None
            if normal_detail:
                report('Đang bake normal map và bóng hốc')
                tick = time.perf_counter()
                normal_map, cavity_map, _ = surface_detail.bake(
                    mesh, baked, high, relief_map, strength=relief if relief_map is not None else 0,
                    rasterize=rasterize_position_atlas)
                colors = surface_detail.darken_cavities(
                    colors, cavity_map, surface_detail.seen_texels(baked, relief_map))
                normal_image = _texture_image(normal_map)
                normal_image.save(folder / 'normal.png')
                clay = np.full(cavity_map.shape + (3,), 0.82, np.float32)
                clay_image = _texture_image(surface_detail.darken_cavities(clay, cavity_map))
                timings['normal_seconds'] = round(time.perf_counter() - tick, 2)
            texture_image = _texture_image(colors)
            texture_image.save(folder / 'texture.png')
            uv_visual = dict(uv=baked['uvs'])
            atlas_vertices = mesh.vertices[baked['vmapping']]
            atlas_normals = mesh.vertex_normals[baked['vmapping']]
            material = trimesh.visual.material.PBRMaterial(
                baseColorTexture=texture_image, normalTexture=normal_image,
                metallicFactor=0.0, roughnessFactor=1.0)
            textured_mesh = trimesh.Trimesh(
                vertices=atlas_vertices, faces=baked['indices'], vertex_normals=atlas_normals,
                visual=trimesh.visual.TextureVisuals(material=material, **uv_visual), process=False)
            if clay_image is not None:  # bản đất sét cũng mang normal map để thấy chi tiết
                clay_material = trimesh.visual.material.PBRMaterial(
                    baseColorTexture=clay_image, normalTexture=normal_image,
                    metallicFactor=0.0, roughnessFactor=0.8)
                clay_mesh = trimesh.Trimesh(
                    vertices=atlas_vertices, faces=baked['indices'], vertex_normals=atlas_normals,
                    visual=trimesh.visual.TextureVisuals(material=clay_material, **uv_visual),
                    process=False)
        # TripoSR uses Z-up. GLB and the viewer use Y-up.
        orientation = trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0])
        mesh.apply_transform(orientation)
        colored_mesh = mesh
        if textured_mesh is not None:
            textured_mesh.apply_transform(orientation)
            colored_mesh = textured_mesh
        if clay_mesh is not None:
            clay_mesh.apply_transform(orientation)
        report('Đang lưu GLB và OBJ')
        colored_mesh.export(folder / 'model.glb')
        mesh.export(folder / 'model.obj')
        if clay_mesh is not None:
            clay_mesh.export(folder / 'geometry.glb')
        else:
            geometry = mesh.copy()
            geometry.visual = trimesh.visual.TextureVisuals(
                material=trimesh.visual.material.PBRMaterial(baseColorFactor=[210, 210, 210, 255],
                                                           metallicFactor=0.0, roughnessFactor=0.8))
            geometry.export(folder / 'geometry.glb')
        peak[0] = max(peak[0], process.memory_info().rss)
        metrics = {
            'seconds': round(time.perf_counter() - started, 2),
            'peak_process_ram_gib': round(peak[0] / 1024**3, 2),
            'vertices': len(mesh.vertices), 'faces': len(mesh.faces),
            'resolution': (HUNYUAN_OCTREE if use_hunyuan else QUALITY)[quality], 'device': 'cpu',
            'threshold': threshold, 'texture_resolution': texture_resolution if texture else 0,
            'background_model': _background_session[0] if remove_background and _background_session else None,
            'shape_model': SHAPE_MODELS[0] if use_hunyuan else SHAPE_MODELS[1],
            'seed': seed if use_hunyuan else None, 'shape_note': shape_note,
            'smooth': smooth and (mesh_cleanup or not use_hunyuan), 'relief': relief_info,
            'mesh_cleanup': cleanup_info,
            'normal_detail': bool(texture and normal_detail),
            'photo_texture': bool(texture and photo_texture and relief_map is not None), **timings,
        }
        (folder / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        return folder, metrics
    except Exception as error:
        (folder / 'error.txt').write_text(str(error), encoding='utf-8')
        raise
    finally:
        stop.set()
        watcher.join()


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    options = dict(arg[2:].split('=', 1) for arg in sys.argv[2:] if '=' in arg)
    folder, metrics = generate(sys.argv[1], '--no-remove-bg' not in sys.argv,
                               quality=options.get('quality', 'Chi tiết'),
                               relief=float(options.get('relief', 1.5)),
                               smooth='--no-smooth' not in sys.argv,
                               photo_texture='--no-photo' not in sys.argv,
                               shape_model=options.get('shape', SHAPE_MODELS[0]),
                               seed=int(options.get('seed', 1234)),
                               normal_detail='--no-normal' not in sys.argv,
                               mesh_cleanup='--no-cleanup' not in sys.argv,
                               report=lambda text: print(text, flush=True))
    print(json.dumps({'folder': str(folder), **metrics}), flush=True)
