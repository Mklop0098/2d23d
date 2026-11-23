import unittest

import numpy as np
import trimesh

from spot_processing import SPOT_MODES, process_spot


class SpotProcessingTests(unittest.TestCase):
    def setUp(self):
        self.clean = trimesh.creation.icosphere(subdivisions=3)
        self.mesh = self.clean.copy()
        self.mesh.vertices[0] *= 1.7
        self.mesh.visual.vertex_colors = np.tile([120, 80, 40, 255], (len(self.mesh.vertices), 1))

    def test_detection_preserves_geometry_and_colors(self):
        vertices = self.mesh.vertices.copy()
        result, report, before, after = process_spot(self.mesh)
        self.assertIn(0, [point['vertex_id'] for point in report['flagged']])
        self.assertEqual(report['summary']['n_moved'], 0)
        np.testing.assert_array_equal(result.vertices, vertices)
        np.testing.assert_array_equal(self.mesh.vertices, vertices)
        np.testing.assert_array_equal(result.visual.vertex_colors, self.mesh.visual.vertex_colors)
        np.testing.assert_array_equal(before.vertices, after.vertices)

    def test_fix_moves_only_flagged_vertices_and_obeys_cap(self):
        result, report, _, _ = process_spot(self.mesh, SPOT_MODES[2], max_displacement=0.25)
        flagged = [point['vertex_id'] for point in report['flagged']]
        untouched = np.ones(len(self.mesh.vertices), dtype=bool)
        untouched[flagged] = False
        np.testing.assert_array_equal(result.vertices[untouched], self.mesh.vertices[untouched])
        np.testing.assert_array_equal(result.faces, self.mesh.faces)
        np.testing.assert_array_equal(result.visual.vertex_colors, self.mesh.visual.vertex_colors)
        self.assertLess(np.linalg.norm(result.vertices[0] - self.clean.vertices[0]),
                        np.linalg.norm(self.mesh.vertices[0] - self.clean.vertices[0]))
        for point in report['flagged']:
            neighbors = self.mesh.vertices[self.mesh.vertex_neighbors[point['vertex_id']]]
            radius = np.linalg.norm(neighbors - neighbors.mean(axis=0), axis=1).mean()
            self.assertLessEqual(point['displacement_magnitude'], 0.25 * radius + 1e-12)

    def test_non_manifold_mesh_is_skipped(self):
        mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0],
                                          [0, -1, 0], [0, 0, 1]],
                               faces=[[0, 1, 2], [1, 0, 3], [0, 1, 4]], process=False)
        result, report, before, after = process_spot(mesh, SPOT_MODES[2])
        self.assertIs(result, mesh)
        self.assertEqual(report['integration']['status'], 'skipped')
        self.assertIn('Non-manifold', report['integration']['reason'])
        self.assertIsNone(before)
        self.assertIsNone(after)

    def test_high_mesh_processing_omits_preview_allocations(self):
        result, report, before, after = process_spot(self.mesh, SPOT_MODES[2], previews=False)
        self.assertGreater(report['summary']['n_moved'], 0)
        self.assertIsNone(before)
        self.assertIsNone(after)
        self.assertTrue(np.isfinite(result.vertex_normals).all())


if __name__ == '__main__':
    unittest.main()
