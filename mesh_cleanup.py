"""CPU mesh cleanup before UV generation, preserving vertex colors."""
import numpy as np
import trimesh


def clean_mesh(mesh, smooth=True, iterations=2):
    import open3d as o3d

    geometry = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces)))
    has_colors = mesh.visual.kind == 'vertex'
    if has_colors:
        geometry.vertex_colors = o3d.utility.Vector3dVector(
            np.asarray(mesh.visual.vertex_colors)[:, :3] / 255.0)
    before = len(geometry.triangles)
    geometry.remove_duplicated_vertices()
    geometry.remove_duplicated_triangles()
    geometry.remove_degenerate_triangles()
    geometry.remove_unreferenced_vertices()
    if not len(geometry.triangles):
        raise ValueError('Lưới không còn mặt hợp lệ sau khi xử lý Open3D.')
    if smooth:
        geometry = geometry.filter_smooth_taubin(
            number_of_iterations=iterations,
            filter_scope=o3d.geometry.FilterScope.Vertex)
    geometry.compute_vertex_normals()
    colors = None
    if has_colors:
        colors = np.round(np.clip(np.asarray(geometry.vertex_colors), 0, 1) * 255).astype('uint8')
    result = trimesh.Trimesh(
        vertices=np.asarray(geometry.vertices).copy(),
        faces=np.asarray(geometry.triangles).copy(),
        vertex_normals=np.asarray(geometry.vertex_normals).copy(),
        vertex_colors=colors, process=False)
    return result, {'faces_before': before, 'faces_after': len(result.faces),
                    'smooth_iterations': iterations if smooth else 0}
