"""Repair topology before Spot, relief, UV generation, and texture baking."""
import numpy as np
import trimesh
from scipy.spatial import cKDTree

HALFMESH_MODES = ('Tắt', 'Sửa cấu trúc lưới', 'Sửa cấu trúc và vá lỗ nhỏ')


def repair_topology(mesh, mode=HALFMESH_MODES[1], max_hole_edges=12):
    import halfmesh

    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.uint32)
    repaired_vertices, repaired_faces = halfmesh.repair(vertices, faces)
    closed = 0
    if mode == HALFMESH_MODES[2]:
        repaired_vertices, repaired_faces, closed = halfmesh.close_holes(
            repaired_vertices, repaired_faces, max_hole_edges=max_hole_edges)
    if not len(repaired_faces) or not np.isfinite(repaired_vertices).all():
        raise ValueError('halfmesh không tạo được lưới hợp lệ. Hãy kiểm tra ảnh hoặc tắt sửa cấu trúc lưới.')
    colors = None
    if mesh.visual.kind == 'vertex':
        # Repair can split/reorder vertices; newly filled vertices get nearby colors.
        _, nearest = cKDTree(vertices).query(repaired_vertices)
        colors = np.asarray(mesh.visual.vertex_colors)[nearest].copy()
    result = trimesh.Trimesh(vertices=repaired_vertices, faces=repaired_faces,
                             vertex_colors=colors, process=False)
    info = {'mode': mode, 'version': halfmesh.__version__, 'max_hole_edges': max_hole_edges,
            'vertices_before': len(vertices), 'vertices_after': len(result.vertices),
            'faces_before': len(faces), 'faces_after': len(result.faces),
            'holes_closed': int(closed),
            'watertight_before': bool(halfmesh.is_watertight(vertices, faces)),
            'watertight_after': bool(halfmesh.is_watertight(repaired_vertices, repaired_faces)),
            'vertex_color_transfer': 'nearest_vertex' if colors is not None else None}
    return result, info
