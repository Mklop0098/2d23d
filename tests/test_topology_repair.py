import unittest

import numpy as np
import trimesh

from topology_repair import HALFMESH_MODES, repair_topology
from mesh_cleanup import clean_mesh


class TopologyRepairTests(unittest.TestCase):
    def test_open3d_keeps_intentionally_split_vertices(self):
        mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0],
                                          [0, 0, 0], [-1, 0, 0], [0, -1, 0]],
                               faces=[[0, 1, 2], [3, 4, 5]], process=False)
        result, info = clean_mesh(mesh, smooth=False, merge_vertices=False)
        self.assertEqual(len(result.vertices), 6)
        self.assertEqual(len(result.split(only_watertight=False)), 2)
        self.assertFalse(info['merge_vertices'])

    def test_repair_resolves_non_manifold_edge(self):
        mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0],
                                          [0, -1, 0], [0, 0, 1]],
                               faces=[[0, 1, 2], [1, 0, 3], [0, 1, 4]], process=False)
        result, info = repair_topology(mesh)
        counts = np.bincount(result.edges_unique_inverse)
        self.assertLessEqual(counts.max(), 2)
        self.assertTrue(np.isfinite(result.vertex_normals).all())
        self.assertEqual(info['faces_before'], 3)
        self.assertEqual(len(mesh.faces), 3)

    def test_hole_filling_respects_edge_limit(self):
        mesh = trimesh.creation.box()
        mesh.update_faces(mesh.triangles_center[:, 2] < 0.49)
        self.assertFalse(mesh.is_watertight)
        limited, info = repair_topology(mesh, HALFMESH_MODES[2], max_hole_edges=3)
        self.assertEqual(info['holes_closed'], 0)
        self.assertFalse(limited.is_watertight)
        filled, info = repair_topology(mesh, HALFMESH_MODES[2], max_hole_edges=4)
        self.assertEqual(info['holes_closed'], 1)
        self.assertTrue(filled.is_watertight)
        self.assertTrue(info['watertight_after'])

    def test_repair_only_preserves_opening_and_input(self):
        mesh = trimesh.creation.icosphere(subdivisions=2)
        mesh.update_faces(np.arange(len(mesh.faces)) != 0)
        vertices = mesh.vertices.copy()
        faces = mesh.faces.copy()
        result, info = repair_topology(mesh, HALFMESH_MODES[1])
        self.assertFalse(result.is_watertight)
        self.assertEqual(info['holes_closed'], 0)
        np.testing.assert_array_equal(mesh.vertices, vertices)
        np.testing.assert_array_equal(mesh.faces, faces)

    def test_vertex_colors_follow_repaired_vertices(self):
        mesh = trimesh.creation.box()
        colors = np.array([[20 * index, 90, 40, 220] for index in range(len(mesh.vertices))], dtype='uint8')
        mesh.visual.vertex_colors = colors
        mesh.update_faces(mesh.triangles_center[:, 2] < 0.49)
        result, info = repair_topology(mesh, HALFMESH_MODES[2], max_hole_edges=4)
        for vertex, color in zip(mesh.vertices, colors):
            matches = np.all(result.vertices == vertex, axis=1)
            self.assertTrue(matches.any())
            for actual in result.visual.vertex_colors[matches]:
                np.testing.assert_array_equal(actual, color)
        self.assertTrue(np.all(result.visual.vertex_colors[:, 3] == 220))
        self.assertEqual(info['vertex_color_transfer'], 'nearest_vertex')


if __name__ == '__main__':
    unittest.main()
