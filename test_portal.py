"""Run with: python -m unittest -v test_portal"""
from types import SimpleNamespace
import unittest

import numpy as np

from portal_effects import HandState, MAX_PORTALS, PortalRenderer, detect_circle, get_gesture


class PortalTests(unittest.TestCase):
    def test_circle_and_incomplete_shapes(self):
        a = np.linspace(0, 2*np.pi, 90)
        circle = np.stack((320+100*np.cos(a), 300+100*np.sin(a)), axis=1)
        self.assertIsNotNone(detect_circle(circle))
        self.assertIsNone(detect_circle(circle[:40]))
        self.assertIsNone(detect_circle([(i, 100) for i in range(90)]))
        self.assertIsNone(detect_circle([(100, 100)]*90))

    def test_gestures(self):
        def hand(extended):
            points = [SimpleNamespace(x=0., y=0.) for _ in range(21)]
            for active, tip, pip in zip(extended, (8, 12, 16, 20), (6, 10, 14, 18)):
                points[pip].y = .2
                points[tip].y = .4 if active else .1
            return points
        self.assertEqual(get_gesture(hand([False]*4)), 'FIST')
        self.assertEqual(get_gesture(hand([True, False, False, False])), 'DRAW')
        self.assertEqual(get_gesture(hand([True]*4)), 'IDLE')

    def test_two_hands_and_fist_preserves_portal(self):
        left, right = HandState(), HandState()
        renderer = PortalRenderer(np.full((80, 120, 3), 150, np.uint8))
        for state, cx in ((left, 200), (right, 480)):
            for i, a in enumerate(np.linspace(0, 2*np.pi, 100)):
                _, circle = state.update('DRAW', (cx+110*np.cos(a), 240+110*np.sin(a)), i/30)
                if circle:
                    renderer.add(circle, i/30)
        self.assertEqual(len(renderer.portals), 2)
        left.update('FIST', (0, 0), 4)
        self.assertFalse(left.trail)
        self.assertIsNone(left.smooth)
        self.assertEqual(len(renderer.portals), 2)
        renderer.clear((left, right))
        self.assertFalse(renderer.portals)
        self.assertFalse(right.trail)
        self.assertEqual(left.cooldown, 0)

    def test_world_mask_and_frame_edges(self):
        renderer = PortalRenderer(np.full((40, 60, 3), 180, np.uint8))
        renderer.add((50, 50, 20), 0)
        frame = np.zeros((100, 100, 3), np.uint8)
        renderer.composite_world(frame, renderer.portals[0], 20)
        self.assertTrue(np.all(frame[50, 50] == 180))
        self.assertTrue(np.all(frame[0, 0] == 0))
        for center in ((0, 0), (99, 99), (-150, 50), (250, 50)):
            renderer.clear()
            renderer.add((*center, 45), 0)
            renderer.render(frame, 1, 1/30)

    def test_limits_opening_and_quality(self):
        renderer = PortalRenderer(np.full((100, 100, 3), 80, np.uint8))
        for _ in range(MAX_PORTALS):
            self.assertTrue(renderer.add((160, 150, 50), 0))
        self.assertFalse(renderer.add((160, 150, 50), 0))
        self.assertEqual(renderer.portals[0].opening(0), 0)
        self.assertEqual(renderer.portals[0].opening(1), 1)
        for quality in ('cinematic', 'smooth'):
            renderer.quality = quality
            for i in range(100):
                renderer.render(np.zeros((300, 320, 3), np.uint8), i/30, 1/30)
            self.assertLessEqual(len(renderer.particles), 1400 if quality == 'cinematic' else 500)
        for radius in range(1, 100):
            renderer.tile(radius)
        self.assertLessEqual(len(renderer.cache), 48)


if __name__ == '__main__':
    unittest.main()
