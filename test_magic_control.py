import math
from types import SimpleNamespace
import unittest

import numpy as np

from magic_control import HandMagic, detect_circle, get_gesture


def loop(radius=140, count=180, clockwise=False):
    angles = np.linspace(0, (-1 if clockwise else 1)*math.tau, count)
    return np.column_stack((500+radius*np.cos(angles), 350+radius*np.sin(angles)))


class MagicControlTests(unittest.TestCase):
    def test_circle_sizes_directions_and_uneven_speed(self):
        for radius in (50, 140, 350, 550):
            for clockwise in (False, True):
                self.assertIsNotNone(detect_circle(loop(radius, clockwise=clockwise)))
        points = loop()
        uneven = np.repeat(points, [12 if i < 50 else 1 for i in range(len(points))], axis=0)
        circle = detect_circle(uneven)
        self.assertLess(math.dist(circle[:2], (500, 350)), 2)
        rng = np.random.default_rng(7)
        self.assertIsNotNone(detect_circle(points+rng.normal(0, 3, points.shape)))

    def test_invalid_shapes(self):
        self.assertIsNone(detect_circle(loop()[:90]))
        self.assertIsNone(detect_circle([(i, 100) for i in range(100)]))
        self.assertIsNone(detect_circle([(100, 100)]*100))
        a = np.linspace(0, math.tau, 180)
        self.assertIsNone(detect_circle(np.column_stack((400+140*np.sin(a), 300+140*np.sin(2*a)))))
        self.assertIsNone(detect_circle(np.vstack((loop()[:90], loop()[:90][::-1]))))

    def cast(self, state, start=0, radius=140):
        for i, point in enumerate(loop(radius)):
            state.update('DRAW', point, (500, 350), start+i/60, 1/60, (1280, 720))
        self.assertIsNotNone(state.magic)

    def test_open_follow_fist_dismiss_redraw(self):
        state = HandMagic()
        self.cast(state)
        for i in range(30):
            state.update('OPEN', (0, 0), (700, 450), 3+i/60, 1/60, (1280, 720))
        self.assertLess(math.dist(state.magic['center'], (700, 450)), 1)
        state.update('FIST', (0, 0), (700, 450), 3.5, 1/60, (1280, 720))
        self.assertIsNotNone(state.magic)
        state.update('FIST', (0, 0), (700, 450), 3.7, 1/60, (1280, 720))
        self.assertIsNone(state.magic)
        self.assertFalse(state.trail)
        self.cast(state, start=3.72, radius=200)

    def test_pose_slip_outlier_and_tracking_loss(self):
        state = HandMagic()
        for i, point in enumerate(loop()[:40]):
            state.update('DRAW', point, (500, 350), i/60, 1/60, (1280, 720))
        original = list(state.trail)
        state.update('IDLE', (0, 0), (0, 0), .7, 1/60, (1280, 720))
        state.update('FIST', (0, 0), (0, 0), .72, 1/60, (1280, 720))
        self.assertEqual(list(state.trail), original)
        state.update('DRAW', (1200, 700), (0, 0), .75, 1/60, (1280, 720))
        self.assertEqual(list(state.trail), original)
        state.lost(.85)
        self.assertEqual(list(state.trail), original)
        state.lost(1.5)
        self.assertFalse(state.trail)

    def test_missing_frames_do_not_confirm_fist(self):
        state = HandMagic()
        self.cast(state)
        state.update('FIST', (0, 0), (0, 0), 3, .03, (1280, 720))
        state.lost(3.1)
        state.update('FIST', (0, 0), (0, 0), 3.3, .03, (1280, 720))
        self.assertIsNotNone(state.magic)

    def test_two_hands_are_independent(self):
        left, right = HandMagic(), HandMagic()
        self.cast(left)
        self.cast(right)
        left.update('FIST', (0, 0), (0, 0), 4, .03, (1280, 720))
        left.update('FIST', (0, 0), (0, 0), 4.2, .03, (1280, 720))
        self.assertIsNone(left.magic)
        self.assertIsNotNone(right.magic)

    def test_gestures_allow_rotation_and_second_finger(self):
        def hand(flags, rotate=False):
            points = [SimpleNamespace(x=.5, y=.8, z=0.) for _ in range(21)]
            for i, (mcp, pip, tip) in enumerate(((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))):
                x = .4+i*.06
                for j, y in ((mcp, .5), (pip, .35), (tip, .1 if flags[i] else .55)):
                    points[j] = SimpleNamespace(x=x, y=y, z=0.)
            if rotate:
                for p in points:
                    x, y = (p.x-.5)*1280, (p.y-.5)*720
                    p.x, p.y = .5-y/1280, .5+x/720
            return points
        for rotated in (False, True):
            self.assertEqual(get_gesture(hand([False]*4, rotated)), 'FIST')
            self.assertEqual(get_gesture(hand([True]*4, rotated)), 'OPEN')
            self.assertEqual(get_gesture(hand([True, False, False, False], rotated)), 'DRAW')
            self.assertEqual(get_gesture(hand([True, True, False, False], rotated)), 'DRAW')


if __name__ == '__main__':
    unittest.main()
