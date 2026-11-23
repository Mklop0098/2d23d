"""Chuyển chi tiết từ ảnh gốc sang lưới TripoSR (chạy CPU, vài giây).

TripoSR chỉ mô tả hình khối trên triplane 64x64, nên hoa văn nhỏ bị mất. Ở đây:
1. Depth Anything V2 Small ước lượng độ sâu tương đối từ ảnh.
2. Độ sâu đó được quy đổi về đơn vị của lưới bằng cách khớp với chính lưới TripoSR.
3. Chỉ phần chi tiết nhỏ (high-pass) được cộng vào các đỉnh nhìn thấy từ camera,
   hình khối tổng thể và mặt sau vẫn do TripoSR quyết định.
Mọi hàm làm việc trong hệ tọa độ gốc của TripoSR (Z-up, camera ảnh đặt ở +X).
"""
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image
from scipy import ndimage

CAMERA_DISTANCE = 1.9  # khoảng cách camera ảnh điều kiện của TripoSR
TAN_HALF_FOV = math.tan(math.radians(20))  # fovy 40°
_depth_model = None


def estimate_depth(image, model_dir, size=770):
    """Độ sâu nghịch đảo tương đối (lớn = gần) ở kích thước của ảnh vào."""
    global _depth_model
    if _depth_model is None:
        from transformers import DepthAnythingForDepthEstimation
        _depth_model = DepthAnythingForDepthEstimation.from_pretrained(str(model_dir)).eval()
    side = max(14, size // 14 * 14)
    x = np.asarray(image.convert('RGB').resize((side, side), Image.BICUBIC), np.float32) / 255
    x = (x - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
    x = torch.from_numpy(x.astype(np.float32)).permute(2, 0, 1)[None]
    with torch.inference_mode():
        depth = _depth_model(pixel_values=x).predicted_depth[0].numpy()
    return np.asarray(Image.fromarray(depth).resize(image.size, Image.BICUBIC), np.float32)


@dataclass
class Camera:
    """Phép chiếu phối cảnh của TripoSR, cộng tỉ lệ/dịch chuyển khớp theo bóng ảnh."""
    scale: float
    center_ndc: tuple
    center_px: tuple

    def project(self, points):
        depth = CAMERA_DISTANCE - points[:, 0]
        u = points[:, 1] / (depth * TAN_HALF_FOV)
        v = points[:, 2] / (depth * TAN_HALF_FOV)
        px = self.center_px[0] + (u - self.center_ndc[0]) * self.scale
        py = self.center_px[1] - (v - self.center_ndc[1]) * self.scale
        return px, py, depth

    def pixels_per_unit(self):
        return self.scale / (CAMERA_DISTANCE * TAN_HALF_FOV)


def fit_camera(vertices, mask):
    """Lưới TripoSR thường nhỏ hơn bóng ảnh một chút; khớp khung bao để hai bên trùng nhau."""
    ys, xs = np.nonzero(mask)
    depth = CAMERA_DISTANCE - vertices[:, 0]
    u = vertices[:, 1] / (depth * TAN_HALF_FOV)
    v = vertices[:, 2] / (depth * TAN_HALF_FOV)
    scale = ((xs.max() - xs.min()) / max(np.ptp(u), 1e-6) +
             (ys.max() - ys.min()) / max(np.ptp(v), 1e-6)) / 2
    return Camera(scale, ((u.min() + u.max()) / 2, (v.min() + v.max()) / 2),
                  ((xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2))


def depth_buffer(mesh, camera, shape):
    """Bộ đệm độ sâu thật (rasterize tam giác bằng OpenGL) theo đúng phép chiếu của ảnh."""
    import moderngl
    height, width = shape
    px, py, depth = camera.project(np.asarray(mesh.vertices))
    near, far = float(depth.min()) - 1e-3, float(depth.max()) + 1e-3
    ndc = np.stack([px / width * 2 - 1, 1 - py / height * 2,
                    (depth - near) / (far - near) * 2 - 1, depth], 1).astype('f4')
    ctx = moderngl.create_context(standalone=True)
    program = ctx.program(
        vertex_shader="""#version 330
            in vec4 p; out float d;
            void main() { d = p.w; gl_Position = vec4(p.xyz, 1.0); }""",
        fragment_shader="""#version 330
            in float d; out vec4 c;
            void main() { c = vec4(d, 0.0, 0.0, 1.0); }""")
    vbo = ctx.buffer(ndc.tobytes())
    ibo = ctx.buffer(np.asarray(mesh.faces, 'i4').tobytes())
    vao = ctx.vertex_array(program, [(vbo, '4f', 'p')], ibo)
    color = ctx.texture((width, height), 1, dtype='f4')
    fbo = ctx.framebuffer([color], ctx.depth_renderbuffer((width, height)))
    fbo.use()
    ctx.enable(moderngl.DEPTH_TEST)
    fbo.clear(1e9, 0, 0, 0, depth=1.0)
    vao.render()
    zbuffer = np.frombuffer(color.read(), 'f4').reshape(height, width)[::-1].copy()
    for resource in (vao, vbo, ibo, program, color, fbo):
        resource.release()
    ctx.release()
    return zbuffer


def visibility(points, camera, zbuffer, tolerance):
    px, py, depth = camera.project(points)
    height, width = zbuffer.shape
    x = np.clip(np.round(px).astype(int), 0, width - 1)
    y = np.clip(np.round(py).astype(int), 0, height - 1)
    return depth <= zbuffer[y, x] + tolerance


def _sample(image, px, py):
    return ndimage.map_coordinates(image, [py, px], order=1, mode='nearest')


def _facing(points, normals):
    to_camera = np.array([CAMERA_DISTANCE, 0, 0]) - points
    to_camera /= np.linalg.norm(to_camera, axis=1, keepdims=True)
    return np.einsum('ij,ij->i', normals, to_camera), to_camera


def _smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


class Relief:
    """Bản đồ chi tiết (đơn vị lưới) và trọng số dùng chung cho hình khối và texture."""

    def __init__(self, mesh, disparity, mask, cell=2 * 0.87 / 64):
        self.mask = mask
        vertices = np.asarray(mesh.vertices)
        self.camera = fit_camera(vertices, mask)
        # cell: chi tiết nhỏ nhất mô hình hình khối tự biểu diễn được (TripoSR: một ô triplane 64).
        self.cell_px = cell * self.camera.pixels_per_unit()
        self.tolerance = cell
        # Mờ dần trong vài pixel cạnh bóng: độ sâu ở viền ảnh không tin được.
        inside = ndimage.distance_transform_edt(mask)
        self.edge = _smoothstep(1.0, max(3.0, self.cell_px * 0.6), inside).astype(np.float32)

        # Quy đổi độ sâu tương đối về khoảng cách thật bằng các đỉnh nhìn thẳng camera.
        px, py, depth = self.camera.project(vertices)
        facing, _ = _facing(vertices, np.asarray(mesh.vertex_normals))
        self.update_depth(mesh)
        seen = visibility(vertices, self.camera, self.zbuffer, self.tolerance)
        good = seen & (facing > 0.5) & (_sample(self.edge, px, py) > 0.99)
        if good.sum() < 200:
            raise ValueError('Không đủ bề mặt nhìn thấy để khớp độ sâu.')
        d, z = _sample(disparity, px[good], py[good]), depth[good]
        keep = np.ones(len(d), bool)
        for _ in range(3):  # bình phương tối thiểu, bỏ dần điểm lệch
            a, b = np.polyfit(d[keep], z[keep], 1)
            residual = np.abs(a * d + b - z)
            keep = residual < 2.5 * np.median(residual[keep]) + 1e-6
        self.fit = (float(a), float(b))
        metric = a * disparity + b
        if a >= 0:  # độ sâu ngược chiều với lưới: không dùng
            metric = np.full_like(disparity, float(np.median(z)))

        # Giữ phần nhỏ hơn khoảng một ô triplane: đúng phần TripoSR không biểu diễn được.
        sigma = max(1.5, self.cell_px * 0.9)
        weight = ndimage.gaussian_filter(mask.astype(np.float32), sigma)
        low = ndimage.gaussian_filter(metric * mask, sigma) / np.maximum(weight, 1e-4)
        detail = (low - metric) * mask  # dương = nhô về phía camera
        limit = 3 * np.std(detail[mask]) + 1e-6
        self.detail = np.clip(detail, -limit, limit).astype(np.float32)

    def update_depth(self, mesh):
        """Gọi lại sau khi lưới đổi (cộng chi tiết) để phép thử che khuất khớp lưới mới."""
        self.zbuffer = depth_buffer(mesh, self.camera, self.mask.shape)

    def weights(self, points, normals):
        px, py, _ = self.camera.project(points)
        facing, to_camera = _facing(points, normals)
        seen = visibility(points, self.camera, self.zbuffer, self.tolerance)
        weight = seen * _smoothstep(0.15, 0.55, facing) * _sample(self.edge, px, py)
        return weight, px, py, to_camera


def add_relief(mesh, relief, strength=1.0):
    vertices = np.asarray(mesh.vertices)
    weight, px, py, to_camera = relief.weights(vertices, np.asarray(mesh.vertex_normals))
    offset = _sample(relief.detail, px, py) * strength
    # Lùi vào ít hơn nhô ra: phần mỏng như hào quang dễ bị thủng sang mặt sau.
    offset = np.clip(offset, -0.6 * relief.tolerance, 1.2 * relief.tolerance) * weight
    # Làm mượt trường độ dời theo lân cận trên lưới để bỏ vết rách ở bậc độ sâu.
    operator = trimesh.smoothing.laplacian_calculation(mesh, equal_weight=True)
    # Lưới càng dày thì cần nhiều lượt hơn để phủ cùng một vùng trên ảnh.
    edge_px = mesh.edges_unique_length.mean() * relief.camera.pixels_per_unit()
    for _ in range(max(1, round(6 / max(edge_px, 1e-3)))):
        offset = 0.5 * offset + 0.5 * (operator @ offset)
    result = mesh.copy()
    result.vertices = vertices + to_camera * offset[:, None]
    # Một lượt làm mượt rất nhẹ để bỏ răng cưa do lấy mẫu ảnh trên lưới.
    trimesh.smoothing.filter_taubin(result, lamb=0.3, nu=-0.31, iterations=1)
    return result, float(np.abs(offset[weight > 0.5]).mean() if (weight > 0.5).any() else 0)


def spread_photo_colors(mesh, relief, photo):
    """Màu từng đỉnh: lấy từ ảnh ở phần nhìn thấy, lan ra phần khuất bằng nghiệm Laplace.

    Dùng khi lưới không trùng trường màu của TripoSR (lưới Hunyuan): màu khuất liền mạch,
    cùng tông với ảnh thay vì màu tra ở chỗ TripoSR coi là khoảng trống.
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    vertices = np.asarray(mesh.vertices)
    weight, px, py, _ = relief.weights(vertices, np.asarray(mesh.vertex_normals))
    rgb = np.asarray(photo.convert('RGB'), np.float32) / 255
    sampled = np.stack([_sample(rgb[..., c], px, py) for c in range(3)], 1)
    known = weight > 0.5
    if known.sum() < 50:
        return np.tile(np.median(rgb[relief.mask], 0), (len(vertices), 1))
    edges = mesh.edges_unique
    n = len(vertices)
    adjacency = sparse.coo_matrix((np.ones(len(edges) * 2), (np.r_[edges[:, 0], edges[:, 1]],
                                   np.r_[edges[:, 1], edges[:, 0]])), shape=(n, n)).tocsr()
    laplacian = sparse.diags(np.asarray(adjacency.sum(1)).ravel()) - adjacency
    unknown = ~known
    # Phần không nối với vùng nhìn thấy kéo nhẹ về màu trung bình để hệ không suy biến.
    mean = sampled[known].mean(0)
    eps = 1e-3
    system = (laplacian[unknown][:, unknown] + eps * sparse.identity(unknown.sum())).tocsc()
    rhs = -laplacian[unknown][:, known] @ sampled[known] + eps * mean
    colors = sampled.copy()
    colors[unknown] = np.column_stack([spsolve(system, rhs[:, c]) for c in range(3)])
    return np.clip(colors, 0, 1).astype(np.float32)


def project_photo(colors, positions, normals, relief, photo, blend=0.85, fill=None):
    """Thay màu nội suy của TripoSR bằng màu ảnh gốc ở các bề mặt nhìn thẳng camera."""
    valid = positions[..., 3] > 0
    points = positions[valid, :3].astype(np.float64)
    normal = normals[valid, :3].astype(np.float64)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-8)
    weight, px, py, _ = relief.weights(points, normal)
    rgb = np.asarray(photo.convert('RGB'), np.float32) / 255
    sampled = np.stack([_sample(rgb[..., c], px, py) for c in range(3)], 1)
    base = colors[valid, :3].astype(np.float32)
    seen, hidden = weight > 0.5, weight < 0.05
    if fill is not None:  # (đỉnh, màu) từ spread_photo_colors: tra đỉnh gần nhất cho từng texel
        from scipy.spatial import cKDTree
        fill_vertices, fill_colors = fill
        base = fill_colors[cKDTree(fill_vertices).query(points, workers=-1)[1]]
    elif seen.sum() > 100 and hidden.sum() > 100:
        # Màu TripoSR ở phần khuất thường tối và loang; cân trung bình/độ lệch theo màu ảnh
        # ở phần nhìn thấy để lưng và hông cùng tông với mặt trước.
        mean_photo, std_photo = sampled[seen].mean(0), sampled[seen].std(0)
        mean_ai, std_ai = base[hidden].mean(0), base[hidden].std(0)
        ratio = np.clip(std_photo / np.maximum(std_ai, 1e-4), 0.3, 1.5)
        base = np.clip((base - mean_ai) * ratio + mean_photo, 0, 1)
    weight = (weight * blend)[:, None]
    result = colors.copy()
    result[valid, :3] = base * (1 - weight) + sampled * weight
    return result
