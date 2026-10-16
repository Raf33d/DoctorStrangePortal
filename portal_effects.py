"""Camera-independent portal rendering and gesture helpers."""
from collections import deque
from dataclasses import dataclass, field
import math
import random

import cv2
import numpy as np

ORANGE = (0, 100, 255)
GOLD = (30, 195, 255)
HOT = (180, 245, 255)
MAX_PORTALS = 6


def get_gesture(hand):
    def extended(tip, pip):
        wrist = hand[0]
        return math.hypot(hand[tip].x-wrist.x, hand[tip].y-wrist.y) > 1.12 * math.hypot(
            hand[pip].x-wrist.x, hand[pip].y-wrist.y)
    fingers = [extended(t, p) for t, p in ((8, 6), (12, 10), (16, 14), (20, 18))]
    if not any(fingers):
        return "FIST"
    return "DRAW" if fingers == [True, False, False, False] else "IDLE"


def detect_circle(points):
    if len(points) < 35:
        return None
    arr = np.asarray(points, dtype=np.float32)
    center = arr.mean(axis=0)
    distances = np.linalg.norm(arr-center, axis=1)
    radius = float(distances.mean())
    if not 45 <= radius <= 230 or distances.std()/radius > .25:
        return None
    if np.linalg.norm(arr[0]-arr[-1]) > radius*.65:
        return None
    angles = np.unwrap(np.arctan2(arr[:, 1]-center[1], arr[:, 0]-center[0]))
    if not 5.2 <= abs(angles[-1]-angles[0]) <= 8.5:
        return None
    return int(center[0]), int(center[1]), int(radius)


@dataclass
class HandState:
    trail: deque = field(default_factory=lambda: deque(maxlen=180))
    smooth: object = None
    last_seen: float = 0
    cooldown: float = 0

    def reset(self):
        self.trail.clear()
        self.smooth = None

    def update(self, gesture, raw_point, now):
        self.last_seen = now
        if gesture != "DRAW":
            self.reset()
            return None, None
        if self.smooth is None:
            self.smooth = raw_point
        else:
            self.smooth = tuple(s+.45*(p-s) for s, p in zip(self.smooth, raw_point))
        point = tuple(int(v) for v in self.smooth)
        if self.trail and math.dist(point, self.trail[-1]) > 100:
            self.trail.clear()
        # Cooldown starts a new trail instead of accumulating duplicate circles.
        if now < self.cooldown:
            self.trail.clear()
            return point, None
        self.trail.append(point)
        circle = detect_circle(self.trail)
        if circle:
            self.trail.clear()
            self.cooldown = now+2
        return point, circle


@dataclass
class Portal:
    x: int
    y: int
    radius: int
    born: float
    seed: float = field(default_factory=lambda: random.uniform(0, 100))

    def opening(self, now):
        progress = min(1., max(0., (now-self.born)/.75))
        return 1-(1-progress)**3


class PortalRenderer:
    def __init__(self, image, quality="cinematic"):
        if image is None or image.size == 0:
            raise ValueError("Destination image could not be loaded.")
        h, w = image.shape[:2]
        size = min(h, w)
        # Square cover crop preserves the skyline's proportions inside the circle.
        crop = image[(h-size)//2:(h+size)//2, (w-size)//2:(w+size)//2]
        self.destination = cv2.resize(crop, (768, 768), interpolation=cv2.INTER_AREA)
        self.quality = quality
        self.portals = []
        self.particles = []
        self.cache = {}

    def add(self, circle, now):
        if len(self.portals) >= MAX_PORTALS:
            return False
        self.portals.append(Portal(*circle, now))
        return True

    def clear(self, states=()):
        self.portals.clear()
        self.particles.clear()
        for state in states:
            state.reset()
            state.cooldown = 0

    def tile(self, radius):
        radius = max(1, int(radius))
        if radius not in self.cache:
            size = radius*2+1
            yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
            alpha = np.clip((radius-np.sqrt(xx*xx+yy*yy))/2, 0, 1).astype(np.float32)[..., None]
            self.cache[radius] = (cv2.resize(self.destination, (size, size)), alpha)
            # Bound memory while portal opening radii change every frame.
            if len(self.cache) > 48:
                del self.cache[next(iter(self.cache))]
        return self.cache[radius]

    def composite_world(self, frame, portal, radius):
        image, alpha = self.tile(radius)
        h, w = frame.shape[:2]
        left, top = portal.x-radius, portal.y-radius
        x0, y0 = max(0, left), max(0, top)
        x1, y1 = min(w, portal.x+radius+1), min(h, portal.y+radius+1)
        if x0 >= x1 or y0 >= y1:
            return
        sx, sy = x0-left, y0-top
        a = alpha[sy:sy+y1-y0, sx:sx+x1-x0]
        target = frame[y0:y1, x0:x1]
        target[:] = (target*(1-a)+image[sy:sy+y1-y0, sx:sx+x1-x0]*a).astype(np.uint8)

    def fire_ring(self, layer, portal, radius, now, dt):
        count = 150 if self.quality == "cinematic" else 80
        angles = np.linspace(0, math.tau, count, endpoint=False)
        phase = now*2.8+portal.seed
        wobble = 2*np.sin(angles*13+phase*3)+1.5*np.sin(angles*31-phase*2)
        for offset, color, thickness in ((4, ORANGE, 5), (0, GOLD, 3), (-2, HOT, 1)):
            rr = radius+offset+wobble
            points = np.stack((portal.x+rr*np.cos(angles), portal.y+rr*np.sin(angles)), axis=-1)
            cv2.polylines(layer, [points.astype(np.int32)], True, color, thickness, cv2.LINE_AA)
        # Rotating tongues of flame, outside the doorway instead of across the world.
        for i in range(count//2):
            a = i*math.tau/(count//2)+phase
            length = 5+12*(.5+.5*math.sin(i*2.7+phase*5))
            start = (int(portal.x+radius*math.cos(a)), int(portal.y+radius*math.sin(a)))
            end = (int(portal.x+(radius+length)*math.cos(a+.035)),
                   int(portal.y+(radius+length)*math.sin(a+.035)))
            cv2.line(layer, start, end, GOLD if i % 3 else HOT, 1, cv2.LINE_AA)
        rate = 220 if self.quality == "cinematic" else 85
        expected = rate*dt
        amount = int(expected)+(random.random() < expected % 1)
        for _ in range(amount):
            a = random.uniform(0, math.tau)
            speed = random.uniform(35, 150)
            life = random.uniform(.25, .85)
            self.particles.append([portal.x+radius*math.cos(a), portal.y+radius*math.sin(a),
                                   speed*math.cos(a)-45*math.sin(a),
                                   speed*math.sin(a)+45*math.cos(a), life, life])

    def render(self, frame, now, dt, states=()):
        dt = min(.05, max(0., dt))
        layer = np.zeros_like(frame)
        for portal in self.portals:
            radius = max(1, round(portal.radius*portal.opening(now)))
            self.composite_world(frame, portal, radius)
            self.fire_ring(layer, portal, radius, now, dt)
        for state in states:
            if len(state.trail) > 1:
                cv2.polylines(layer, [np.asarray(state.trail, dtype=np.int32)], False, GOLD, 3, cv2.LINE_AA)
        alive = []
        for p in self.particles:
            p[4] -= dt
            if p[4] <= 0:
                continue
            previous = (int(p[0]), int(p[1]))
            p[0] += p[2]*dt
            p[1] += p[3]*dt
            p[3] += 135*dt
            color = tuple(int(c*p[4]/p[5]) for c in GOLD)
            cv2.line(layer, previous, (int(p[0]), int(p[1])), color, 1, cv2.LINE_AA)
            alive.append(p)
        self.particles = alive[-(1400 if self.quality == "cinematic" else 500):]
        # Blur at half resolution to keep cinematic bloom affordable on the CPU.
        h, w = frame.shape[:2]
        small = cv2.resize(layer, (max(1, w//2), max(1, h//2)))
        glow = cv2.resize(cv2.GaussianBlur(small, (21, 21), 0), (w, h))
        cv2.addWeighted(frame, 1, glow, 1.3, 0, dst=frame)
        cv2.add(frame, layer, dst=frame)
        return frame
