"""Chi tiết nằm trong texture thay vì trong lưới: normal map và bóng hốc (cavity).

Lưới xuất ra giữ ~100k mặt (tạo UV nhanh), còn texture 2048 có ~4 triệu điểm. Mỗi điểm
nhận hướng bề mặt từ:
1. lưới đầy đủ của mô hình hình khối (Hunyuan ~500k mặt) — chi tiết khắp vật, cả lưng;
2. độ sâu ước lượng từ ảnh (mặt trước) — vân khắc mịn tới mức điểm ảnh.
Bóng hốc lấy từ độ cong của lưới đầy đủ, làm tối nhẹ các rãnh trong màu.
Mọi hàm làm việc trong hệ toạ độ TripoSR (Z lên trên, camera ảnh ở +X).
"""
import numpy as np
import trimesh
from scipy import ndimage, sparse
from scipy.spatial import cKDTree

import detail

PHOTO_SCALE = 0.5  # vân từ ảnh nhân với thanh "Độ nổi chi tiết" (mặc định 1,5 -> 0,75)
MAX_TILT = 0.5


def _normalize(vectors):
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-8)


def cavity(mesh, iterations=10):
    """Độ lõm từng đỉnh trong [0, 1]: 1 là đáy rãnh, 0 là mặt phẳng hoặc gờ lồi.

    So mỗi đỉnh với phiên bản lưới đã làm mượt: đỉnh nằm sâu hơn bề mặt mượt theo hướng
    pháp tuyến là đỉnh ở trong rãnh. Rẻ hơn nhiều so với tính che sáng (AO) bằng tia.
    """
    vertices = np.asarray(mesh.vertices)
    operator = trimesh.smoothing.laplacian_calculation(mesh, equal_weight=True)
    smooth = vertices.copy()
    for _ in range(iterations):
        smooth = operator @ smooth
    depth = np.einsum('ij,ij->i', vertices - smooth, np.asarray(mesh.vertex_normals))
    scale = np.percentile(np.abs(depth), 95) + 1e-9
    return np.clip(-depth / scale, 0, 1).astype(np.float32)


def _atlas_tangents(vertices, faces, uvs):
    """Tiếp tuyến theo U và V cho từng đỉnh atlas (công thức chuẩn từ đạo hàm UV)."""
    p0, p1, p2 = (vertices[faces[:, i]] for i in range(3))
    t0, t1, t2 = (uvs[faces[:, i]] for i in range(3))
    e1, e2 = p1 - p0, p2 - p0
    d1, d2 = t1 - t0, t2 - t0
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    inverse = np.where(np.abs(det) > 1e-12, 1.0 / np.where(det == 0, 1, det), 0.0)[:, None]
    tangent = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * inverse
    bitangent = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) * inverse
    count = len(vertices)
    rows = faces.ravel()
    accumulate = sparse.coo_matrix((np.ones(len(rows)), (rows, np.arange(len(rows)))),
                                   shape=(count, len(rows))).tocsr()
    per_vertex_t = accumulate @ np.repeat(tangent, 3, axis=0)
    per_vertex_b = accumulate @ np.repeat(bitangent, 3, axis=0)
    return _normalize(per_vertex_t), _normalize(per_vertex_b)


def bake(mesh, baked, high, relief, strength=1.0, rasterize=None):
    """Trả (normal map tangent-space RGB [0,1], độ lõm từng texel, mặt nạ texel hợp lệ).

    mesh: lưới xuất (đã có UV qua `baked`); high: lưới đầy đủ chưa giảm (hoặc chính lưới
    trước khi cộng chi tiết với TripoSR); relief: detail.Relief (None nếu không có).
    rasterize: tsr.bake_texture.rasterize_position_atlas.
    """
    import types
    resolution = baked['positions'].shape[0]
    padding = round(max(2, resolution / 256))
    atlas_vertices = np.asarray(mesh.vertices)[baked['vmapping']]
    faces = np.asarray(baked['indices'], np.int64)
    tangent, bitangent = _atlas_tangents(atlas_vertices, faces, np.asarray(baked['uvs'], np.float64))
    identity = np.arange(len(atlas_vertices))

    def raster(values):
        return rasterize(types.SimpleNamespace(vertices=values), identity, baked['indices'],
                         baked['uvs'], resolution, padding)

    tangent_texture = raster(tangent)
    bitangent_texture = raster(bitangent)
    valid = baked['positions'][..., 3] > 0
    points = baked['positions'][valid, :3].astype(np.float64)
    normal = _normalize(baked['normals'][valid, :3].astype(np.float64))
    tangent_t = tangent_texture[valid, :3].astype(np.float64)
    bitangent_t = bitangent_texture[valid, :3].astype(np.float64)

    # 1. Hướng bề mặt và độ lõm từ lưới đầy đủ: nội suy theo 4 đỉnh gần nhất.
    high_vertices = np.asarray(high.vertices)
    distance, index = cKDTree(high_vertices).query(points, k=4, workers=-1)
    weights = 1.0 / np.maximum(distance, 1e-6)
    weights /= weights.sum(1, keepdims=True)
    detailed = _normalize(np.einsum('nk,nkc->nc', weights, np.asarray(high.vertex_normals)[index]))
    concave = np.einsum('nk,nk->n', weights, cavity(high)[index])
    # Lưới đầy đủ có thể lệch hướng ở chỗ hai lưới không trùng (mép mỏng): giữ pháp tuyến thô.
    agree = np.einsum('ij,ij->i', detailed, normal)
    detailed = np.where((agree > 0.2)[:, None], detailed, normal)

    # 2. Vân khắc từ độ sâu ảnh ở mặt nhìn thấy: bề mặt x = h(y, z) có pháp tuyến (1, -h_y, -h_z).
    if relief is not None and strength > 0:
        seen, px, py, _ = relief.weights(points, normal)
        # Gradient làm mượt nhẹ: bậc độ sâu trong ảnh cho độ dốc cực lớn, gây mảng đen gắt.
        grad_y, grad_x = np.gradient(ndimage.gaussian_filter(relief.detail, 1.0))
        per_unit = relief.camera.pixels_per_unit()
        slope_y = detail._sample(grad_x, px, py) * per_unit   # ảnh sang phải = +Y
        slope_z = -detail._sample(grad_y, px, py) * per_unit  # ảnh đi xuống = -Z
        tilt = np.zeros_like(points)
        tilt[:, 1], tilt[:, 2] = slope_y, slope_z
        tilt *= (seen * strength * PHOTO_SCALE)[:, None]
        # Giới hạn độ nghiêng (~27°) để vân khắc không làm bề mặt quay lưng lại đèn.
        length = np.linalg.norm(tilt, axis=1, keepdims=True)
        tilt *= np.minimum(1.0, MAX_TILT / np.maximum(length, 1e-8))
        detailed = _normalize(detailed - tilt)

    # Đổi sang tangent space của lưới xuất. Trục B trực giao hoá nhưng giữ chiều theo +V.
    tangent_t = _normalize(tangent_t - normal * np.einsum('ij,ij->i', tangent_t, normal)[:, None])
    handed = np.sign(np.einsum('ij,ij->i', np.cross(normal, tangent_t), bitangent_t))
    handed[handed == 0] = 1
    bitangent_o = np.cross(normal, tangent_t) * handed[:, None]
    local = np.stack([np.einsum('ij,ij->i', detailed, tangent_t),
                      np.einsum('ij,ij->i', detailed, bitangent_o),
                      np.einsum('ij,ij->i', detailed, normal)], 1)
    local[:, 2] = np.maximum(local[:, 2], 0.05)
    local = _normalize(local)

    normal_map = np.zeros((resolution, resolution, 3), np.float32)
    normal_map[...] = (0.5, 0.5, 1.0)  # texel trống: pháp tuyến phẳng
    normal_map[valid] = local * 0.5 + 0.5
    cavity_map = np.zeros((resolution, resolution), np.float32)
    cavity_map[valid] = concave
    return normal_map, cavity_map, valid


def darken_cavities(colors, cavity_map, seen_map=None, hidden_strength=0.45, seen_strength=0.2):
    """Làm tối rãnh. Mặt nhìn thấy đã có bóng thật trong ảnh nên chỉ tối nhẹ."""
    strength = np.full(cavity_map.shape, hidden_strength, np.float32)
    if seen_map is not None:
        strength = hidden_strength + (seen_strength - hidden_strength) * seen_map
    result = colors.copy()
    result[..., :3] *= (1 - strength * cavity_map)[..., None]
    return result


def seen_texels(baked, relief):
    """Trọng số nhìn thấy cho từng texel (0..1), dùng để giảm bóng hốc ở mặt trước."""
    if relief is None:
        return None
    valid = baked['positions'][..., 3] > 0
    points = baked['positions'][valid, :3].astype(np.float64)
    normal = _normalize(baked['normals'][valid, :3].astype(np.float64))
    weight, *_ = relief.weights(points, normal)
    result = np.zeros(valid.shape, np.float32)
    result[valid] = weight
    return result
