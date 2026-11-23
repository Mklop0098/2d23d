"""Feature-preserving smoothing and vertex-sampled surface distance via MeshLab."""
from importlib.metadata import version

import numpy as np
import pymeshlab

SMOOTHING_METHODS = ('Taubin', 'MeshLab giữ cạnh')
MESHLAB_VERSION = version('pymeshlab')


def smooth_with_meshlab(mesh, angle=45.0, iterations=1):
    meshes = pymeshlab.MeshSet()
    meshes.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(mesh.vertices),
                                   face_matrix=np.asarray(mesh.faces, dtype=np.int32)))
    meshes.apply_coord_two_steps_smoothing(
        stepsmoothnum=iterations, normalthr=angle, stepnormalnum=1, stepfitnum=1,
        selected=False)
    filtered = meshes.current_mesh()
    vertices = filtered.vertex_matrix()
    if vertices.shape != mesh.vertices.shape or not np.array_equal(filtered.face_matrix(), mesh.faces):
        raise ValueError('MeshLab thay đổi cấu trúc lưới ngoài dự kiến. Hãy chọn bộ lọc Taubin và tạo lại.')
    if not np.isfinite(vertices).all():
        raise ValueError('MeshLab tạo tọa độ không hợp lệ. Hãy giảm mức làm mượt.')
    result = mesh.copy()
    result.vertices = vertices.copy()
    return result, {'filter': 'apply_coord_two_steps_smoothing', 'angle_degrees': angle,
                    'iterations': iterations, 'normal_steps': 1, 'vertex_fit_steps': 1,
                    'vertices': len(result.vertices), 'faces': len(result.faces)}


def measure_surface_deviation(before, after):
    meshes = pymeshlab.MeshSet()
    for mesh in (before, after):
        meshes.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(mesh.vertices),
                                       face_matrix=np.asarray(mesh.faces, dtype=np.int32)))
    bounds = np.vstack((before.bounds, after.bounds))
    search_distance = float(np.linalg.norm(np.ptp(bounds, axis=0)))
    directions = {}
    for source, target, name in ((0, 1, 'before_to_after'), (1, 0, 'after_to_before')):
        result = meshes.get_hausdorff_distance(
            sampledmesh=source, targetmesh=target, savesample=False,
            samplevert=True, sampleedge=False, sampleface=False,
            samplenum=meshes.mesh(source).vertex_number(),
            maxdist=pymeshlab.PureValue(search_distance))
        directions[name] = {key: float(result[key]) for key in ('min', 'max', 'mean', 'RMS')}
        directions[name]['n_samples'] = int(result['n_samples'])
    maximum = max(direction['max'] for direction in directions.values())
    diagonal = max(float(np.linalg.norm(before.extents)), np.finfo(float).eps)
    return {'sampling': 'all_vertices_to_opposite_triangle_surface',
            'directions': directions, 'max_distance': maximum,
            'reference_bbox_diagonal': diagonal, 'max_percent_bbox_diagonal': 100 * maximum / diagonal}
