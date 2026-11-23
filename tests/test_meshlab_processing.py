import unittest

import numpy as np
import trimesh

from meshlab_processing import measure_surface_deviation, smooth_with_meshlab


class MeshLabProcessingTests(unittest.TestCase):
    def test_sharp_cube_corners_and_colors_are_preserved(self):
        mesh = trimesh.creation.box()
        mesh.visual.vertex_colors = np.array([[index * 25, 80, 40, 220]
                                              for index in range(len(mesh.vertices))], dtype='uint8')
        result, info = smooth_with_meshlab(mesh, angle=45, iterations=1)
        np.testing.assert_allclose(result.vertices, mesh.vertices, atol=1e-7)
        np.testing.assert_array_equal(result.faces, mesh.faces)
        np.testing.assert_array_equal(result.visual.vertex_colors, mesh.visual.vertex_colors)
        self.assertEqual(info['iterations'], 1)

    def test_filter_reduces_radial_noise_without_mutating_input(self):
        mesh = trimesh.creation.icosphere(subdivisions=3)
        noise = np.random.default_rng(11).normal(0, 0.02, len(mesh.vertices))
        mesh.vertices *= (1 + noise[:, None])
        vertices = mesh.vertices.copy()
        result, _ = smooth_with_meshlab(mesh)
        before = np.std(np.linalg.norm(vertices, axis=1))
        after = np.std(np.linalg.norm(result.vertices, axis=1))
        self.assertLess(after, before)
        np.testing.assert_array_equal(mesh.vertices, vertices)
        self.assertTrue(result.is_watertight)

    def test_bidirectional_distance_detects_translation(self):
        before = trimesh.creation.box()
        after = before.copy()
        after.apply_translation([0, 0, 0.1])
        distance = measure_surface_deviation(before, after)
        for direction in distance['directions'].values():
            self.assertAlmostEqual(direction['max'], 0.1, places=6)
            self.assertEqual(direction['n_samples'], len(before.vertices))
        self.assertAlmostEqual(distance['max_percent_bbox_diagonal'], 100 * 0.1 / np.sqrt(3), places=5)

    def test_identical_mesh_has_zero_surface_deviation(self):
        mesh = trimesh.creation.box()
        distance = measure_surface_deviation(mesh, mesh.copy())
        self.assertLess(distance['max_distance'], 1e-7)


if __name__ == '__main__':
    unittest.main()
