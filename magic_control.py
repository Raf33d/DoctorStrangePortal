"""Camera-independent gesture, circle fitting and magic lifecycle."""
from collections import deque
import math

import numpy as np


def get_gesture(hand, width=1280, height=720):
    # Measure joint angles in camera coordinates, independent of hand rotation.
    points = np.array([(p.x*width, p.y*height, getattr(p, 'z', 0)*width) for p in hand])
    extended = []
    for mcp, pip, tip in ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)):
        a, b = points[mcp]-points[pip], points[tip]-points[pip]
        denominator = np.linalg.norm(a)*np.linalg.norm(b)
        cosine = np.dot(a, b)/denominator if denominator > 1e-6 else 1
        reach = np.linalg.norm(points[tip]-points[0])
        knuckle = np.linalg.norm(points[pip]-points[0])
        extended.append(cosine < -.65 and reach > knuckle*1.03)
    if not any(extended):
        return 'FIST'
    if sum(extended) >= 3:
        return 'OPEN'
    # A slightly extended second finger no longer interrupts index drawing.
    if extended[0]:
        return 'DRAW'
    return 'IDLE'


def detect_circle(points):
    if len(points) < 24:
        return None
    arr = np.asarray(points, dtype=np.float64)
    # Uniform spatial samples prevent slow sections and pauses biasing the fit.
    distances = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    arr = arr[np.r_[True, distances > .5]]
    if len(arr) < 12:
        return None
    cumulative = np.r_[0, np.cumsum(np.linalg.norm(np.diff(arr, axis=0), axis=1))]
    if cumulative[-1] < 200:
        return None
    samples = np.linspace(0, cumulative[-1], 128)
    arr = np.stack([np.interp(samples, cumulative, arr[:, i]) for i in (0, 1)], axis=1)
    origin = arr.mean(axis=0)
    local = arr-origin
    matrix = np.column_stack((2*local, np.ones(len(local))))
    fit, _, rank, _ = np.linalg.lstsq(matrix, np.sum(local*local, axis=1), rcond=None)
    if rank < 3:
        return None
    center = origin+fit[:2]
    radii = np.linalg.norm(arr-center, axis=1)
    radius = float(np.median(radii))
    if not 45 <= radius <= 600 or np.sqrt(np.mean((radii-radius)**2))/radius > .22:
        return None
    if np.linalg.norm(arr[0]-arr[-1]) > radius*.7:
        return None
    angles = np.unwrap(np.arctan2(arr[:, 1]-center[1], arr[:, 0]-center[0]))
    rotation = abs(angles[-1]-angles[0])
    if not 5.25 <= rotation <= 8.5:
        return None
    # Reject scribbles/retracing even if they happen to end near the start.
    if np.sum(np.abs(np.diff(angles))) > rotation*1.4:
        return None
    return round(center[0]), round(center[1]), round(radius)


class HandMagic:
    def __init__(self):
        self.trail = deque(maxlen=600)
        self.smooth = None
        self.magic = None
        self.last_seen = -math.inf
        self.last_draw = -math.inf
        self.last_check = -math.inf
        self.fist_since = None
        self.gesture = 'IDLE'
        self.rejected = 0

    def reset_drawing(self):
        self.trail.clear()
        self.smooth = None
        self.last_draw = -math.inf
        self.last_check = -math.inf
        self.rejected = 0

    def clear(self):
        self.reset_drawing()
        self.magic = None
        self.fist_since = None

    def lost(self, now):
        # Missing frames never count as confirmation of a fist.
        self.fist_since = None
        if now-self.last_seen > .6:
            self.reset_drawing()

    def update(self, gesture, tip, palm, now, dt, frame_size):
        if now-self.last_seen > .3:
            self.fist_since = None
        self.last_seen = now
        self.gesture = gesture
        if gesture == 'FIST':
            if self.fist_since is None:
                self.fist_since = now
            # A single bad pose frame must not erase a circle or drawing.
            if now-self.fist_since >= .18:
                self.clear()
                self.fist_since = now-.18
            return None
        self.fist_since = None
        if self.magic is not None:
            alpha = 1-math.exp(-18*max(0, dt))
            self.magic['center'] = tuple(a+alpha*(b-a) for a, b in zip(self.magic['center'], palm))
            return None
        if gesture != 'DRAW':
            if now-self.last_draw > .6:
                self.reset_drawing()
            return None
        if now-self.last_draw > .6:
            self.reset_drawing()
        self.last_draw = now
        raw = np.asarray(tip, dtype=float)
        if self.smooth is not None:
            delta = np.linalg.norm(raw-self.smooth)
            if delta > max(120, math.hypot(*frame_size)*.22):
                self.rejected += 1
                if self.rejected < 3:
                    return None
                self.reset_drawing()
                self.last_draw = now
            else:
                self.rejected = 0
                speed = delta/max(dt, 1/120)
                alpha = 1-math.exp(-(12+min(speed/40, 30))*max(dt, 1/120))
                raw = self.smooth+alpha*(raw-self.smooth)
        self.smooth = raw
        point = tuple(int(round(v)) for v in raw)
        # Ignore tiny jitter; keep enough history for large, slow circles.
        if not self.trail or math.dist(point, self.trail[-1]) >= 2:
            self.trail.append(point)
        if now-self.last_check >= .08:
            self.last_check = now
            points = list(self.trail)
            # Discard a short lead-in if the user starts by moving into the loop.
            for fraction in (0, .15, .3):
                circle = detect_circle(points[int(len(points)*fraction):])
                if circle:
                    cx, cy, radius = circle
                    self.magic = {'center': (float(cx), float(cy)), 'radius': radius}
                    self.reset_drawing()
                    break
        return point

    def visible(self, now):
        return self.magic is not None and now-self.last_seen <= .2
