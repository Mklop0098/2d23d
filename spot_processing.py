"""Apply the pinned Spot core to in-memory meshes before UV and texture baking."""
import numpy as np
import trimesh

from third_party.spot import __version__, correct, detect, mesh as spot_mesh, report

SPOT_MODES = ('Tắt', 'Chỉ phát hiện', 'Sửa điểm nghi lỗi')
SPOT_COMMIT = '1200ac96815a3f78c67ca8c5da0f3c2bcc288fea'


def process_spot(mesh, mode=SPOT_MODES[1], threshold=3.5, max_displacement=0.25, previews=True):
    """Return the mesh, a JSON-compatible report, and highlighted before/after meshes.

    Vertex IDs in the report refer to this pre-UV mesh in the AI coordinate frame.
    A copy preserves the input's colors and leaves the caller's mesh untouched.
    """
    parameters = {'k': 2, 'threshold': threshold, 'iterations': 20,
                  'max_displacement': max_displacement}
    integration = {'mode': mode, 'status': 'completed', 'upstream_commit': SPOT_COMMIT,
                   'coordinate_system': 'TripoSR before export rotation',
                   'displacement_units': 'local neighborhood radius'}
    try:
        # The upstream in-memory constructor omits the validation used by its CLI.
        spot_mesh._validate_manifold(mesh)
    except spot_mesh.NotAManifoldTriangleMesh as error:
        integration.update(status='skipped', reason=str(error))
        document = {'schema_version': report.SCHEMA_VERSION,
                    'tool': {'name': 'spot', 'version': __version__},
                    'source': {'path': None, 'n_vertices': len(mesh.vertices), 'n_faces': len(mesh.faces)},
                    'parameters': parameters, 'integration': integration,
                    'summary': {'n_flagged': 0, 'n_moved': 0, 'max_displacement': 0.0,
                                'max_nonflagged_displacement': 0.0}, 'flagged': []}
        return mesh, document, None, None

    graph = spot_mesh.from_trimesh(mesh)
    detection = detect.detect(graph, k=parameters['k'], threshold=threshold)
    correction = correct.taubin_fix(
        graph, detection.flags, iterations=parameters['iterations'] if mode == SPOT_MODES[2] else 0,
        max_displacement=max_displacement)
    if not np.isfinite(correction.vertices).all():
        raise ValueError('Spot tạo tọa độ không hợp lệ. Hãy thử chế độ Chỉ phát hiện.')
    result = mesh.copy()
    result.vertices = correction.vertices
    document = report.build_report(graph, detection, correction, parameters=parameters)
    document['integration'] = integration
    document['summary']['n_moved'] = int(np.count_nonzero(np.linalg.norm(correction.displacement, axis=1)))
    if not previews:
        return result, document, None, None
    colors = np.tile([185, 190, 198, 255], (len(mesh.vertices), 1)).astype('uint8')
    colors[detection.flags] = [240, 55, 40, 255]
    before = trimesh.Trimesh(vertices=graph.vertices.copy(), faces=graph.faces.copy(),
                             vertex_colors=colors, process=False)
    after = trimesh.Trimesh(vertices=correction.vertices.copy(), faces=graph.faces.copy(),
                            vertex_colors=colors, process=False)
    return result, document, before, after
