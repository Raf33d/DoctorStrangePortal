
import cv2
import mediapipe as mp
import numpy as np
import math
import random
import time
from collections import deque

# ==========================================
# CONFIGURATION
# ==========================================
WIDTH, HEIGHT = 1280, 720
SMOOTHING = 0.45
MAX_TRAIL = 600
MIN_RADIUS = 45
MAX_RADIUS = 600
DRAW_GRACE = 0.4

# BGR colors
ORANGE = (0, 140, 255)
GOLD = (0, 210, 255)
WHITE_GOLD = (160, 245, 255)

# ==========================================
# MEDIAPIPE SETUP
# ==========================================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = {
    "Left": {
        "trail": deque(maxlen=MAX_TRAIL),
        "smooth": None,
        "last_seen": 0,
        "last_draw": 0,
        "cooldown": 0,
        "drawing": False
    },
    "Right": {
        "trail": deque(maxlen=MAX_TRAIL),
        "smooth": None,
        "last_seen": 0,
        "last_draw": 0,
        "cooldown": 0,
        "drawing": False
    }
}

# One summoned magic circle per hand; drawing and palm tracking are independent.
portals = {}
particles = []


def follow_hand(magic, palm, dt):
    """Ease toward the palm at the same speed across different frame rates."""
    alpha = 1 - math.exp(-18 * max(0, dt))
    magic["center"] = tuple(
        current + alpha * (target - current)
        for current, target in zip(magic["center"], palm)
    )


def keep_drawing(state, gesture, now):
    """Keep an unfinished loop through short pose slips; fists still stop."""
    if gesture == "DRAW":
        if now - state["last_draw"] > DRAW_GRACE:
            state["trail"].clear()
            state["smooth"] = None
            state["drawing"] = False
        state["last_draw"] = now
        return True
    if gesture == "FIST" or now - state["last_draw"] > DRAW_GRACE:
        state["drawing"] = False
        state["trail"].clear()
        state["smooth"] = None
    return False

# ==========================================
# HAND GESTURE RECOGNITION
# ==========================================
def finger_extended(hand, tip, pip):
    wrist = hand[0]
    tip_dist = math.hypot(
        hand[tip].x - wrist.x,
        hand[tip].y - wrist.y
    )
    pip_dist = math.hypot(
        hand[pip].x - wrist.x,
        hand[pip].y - wrist.y
    )
    return tip_dist > pip_dist * 1.12


def get_gesture(hand):
    index = finger_extended(hand, 8, 6)
    middle = finger_extended(hand, 12, 10)
    ring = finger_extended(hand, 16, 14)
    pinky = finger_extended(hand, 20, 18)

    if not any([index, middle, ring, pinky]):
        return "FIST"

    if index and not middle and not ring and not pinky:
        return "DRAW"

    return "IDLE"


# ==========================================
# CIRCLE DETECTION
# ==========================================
def detect_circle(points):
    if len(points) < 35:
        return None

    arr = np.array(points, dtype=np.float32)
    center = np.mean(arr, axis=0)

    distances = np.linalg.norm(arr - center, axis=1)
    radius = float(np.mean(distances))

    if not MIN_RADIUS <= radius <= MAX_RADIUS:
        return None

    if np.std(distances) / radius > 0.25:
        return None

    # Start and end should be near each other
    if np.linalg.norm(arr[0] - arr[-1]) > radius * 0.65:
        return None

    angles = np.unwrap(
        np.arctan2(
            arr[:, 1] - center[1],
            arr[:, 0] - center[0]
        )
    )

    rotation = abs(angles[-1] - angles[0])

    if not 5.2 <= rotation <= 8.5:
        return None

    return (
        int(center[0]),
        int(center[1]),
        int(radius)
    )


# ==========================================
# MAGIC PARTICLES
# ==========================================
def spawn_particles(cx, cy, radius, amount=8):
    for _ in range(amount):
        angle = random.uniform(0, 2 * math.pi)
        r = radius + random.uniform(-8, 8)

        particles.append({
            "x": cx + r * math.cos(angle),
            "y": cy + r * math.sin(angle),
            "vx": random.uniform(-2, 2),
            "vy": random.uniform(-3, 1),
            "life": random.randint(12, 28),
            "size": random.randint(1, 3)
        })


def update_particles(layer):
    alive = []

    for p in particles:
        p["x"] += p["vx"]
        p["y"] += p["vy"]
        p["life"] -= 1

        if p["life"] <= 0:
            continue

        x, y = int(p["x"]), int(p["y"])

        if 0 <= x < layer.shape[1] and 0 <= y < layer.shape[0]:
            cv2.circle(
                layer, (x, y),
                p["size"],
                GOLD, -1
            )

        alive.append(p)

    particles[:] = alive[-1500:]


# ==========================================
# REALISTIC MAGIC PORTAL
# ==========================================
def draw_magic_portal(layer, portal, t):
    cx, cy, radius = portal
    center = (cx, cy)

    # Pulsating energy
    pulse = int(3 * math.sin(t * 4))
    r = max(20, radius + pulse)

    # Multiple concentric circles
    for offset, thickness, color in [
        (0, 4, WHITE_GOLD),
        (8, 3, ORANGE),
        (-10, 2, GOLD),
        (-22, 2, ORANGE),
        (-38, 1, GOLD)
    ]:
        rr = r + offset
        if rr > 0:
            cv2.circle(
                layer, center, rr,
                color, thickness,
                cv2.LINE_AA
            )

    # Outer rotating runes
    for i in range(36):
        a = 2 * math.pi * i / 36 + t * 0.65

        inner = r - 21
        outer = r - 9

        p1 = (
            int(cx + inner * math.cos(a)),
            int(cy + inner * math.sin(a))
        )

        p2 = (
            int(cx + outer * math.cos(a)),
            int(cy + outer * math.sin(a))
        )

        cv2.line(
            layer, p1, p2,
            GOLD, 2, cv2.LINE_AA
        )

        # Small rune dots
        if i % 3 == 0:
            cv2.circle(
                layer, p1, 3,
                WHITE_GOLD, -1,
                cv2.LINE_AA
            )

    # Counter-rotating inner geometric pattern
    vertices = []

    for i in range(12):
        a = 2 * math.pi * i / 12 - t * 0.35

        x = int(cx + r * 0.65 * math.cos(a))
        y = int(cy + r * 0.65 * math.sin(a))
        vertices.append((x, y))

    for i in range(12):
        cv2.line(
            layer,
            vertices[i],
            vertices[(i + 4) % 12],
            ORANGE, 1, cv2.LINE_AA
        )

    # Rotating inner energy ring
    for i in range(18):
        a = 2 * math.pi * i / 18 + t * 1.2

        x = int(cx + r * 0.43 * math.cos(a))
        y = int(cy + r * 0.43 * math.sin(a))

        cv2.circle(
            layer, (x, y), 2,
            WHITE_GOLD, -1,
            cv2.LINE_AA
        )

    # Random sparks
    spawn_particles(cx, cy, r, amount=5)


# ==========================================
# WEBCAM
# ==========================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Webcam not found!")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

start_time = time.monotonic()
previous_time = start_time
last_timestamp = -1

# ==========================================
# MAIN APPLICATION
# ==========================================
with HandLandmarker.create_from_options(options) as landmarker:

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        now = time.monotonic()
        dt = min(now - previous_time, 0.1)
        previous_time = now

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = max(
            int((now - start_time) * 1000),
            last_timestamp + 1
        )
        last_timestamp = timestamp

        result = landmarker.detect_for_video(
            mp_image, timestamp
        )

        active_hands = set()
        fingertips = {}
        gestures = {}

        # ==================================
        # PROCESS BOTH HANDS
        # ==================================
        for i, hand in enumerate(result.hand_landmarks):
            label = result.handedness[i][0].category_name

            # Match labels to mirrored webcam
            label = "Left" if label == "Right" else "Right"

            if label in active_hands:
                continue
            active_hands.add(label)
            state = hands[label]

            gesture = get_gesture(hand)
            gestures[label] = gesture

            state["last_seen"] = now

            # Follow the palm even for a fist/open hand, before drawing logic.
            if label in portals:
                palm = (
                    sum(hand[j].x for j in (0, 5, 9, 13, 17)) / 5 * width,
                    sum(hand[j].y for j in (0, 5, 9, 13, 17)) / 5 * height
                )
                follow_hand(portals[label], palm, dt)
                state["trail"].clear()
                state["smooth"] = None
                state["drawing"] = False
                continue

            if not keep_drawing(state, gesture, now):
                continue

            tip = hand[8]
            x = int(tip.x * width)
            y = int(tip.y * height)

            if state["smooth"] is None:
                state["smooth"] = (float(x), float(y))
            else:
                sx, sy = state["smooth"]

                state["smooth"] = (
                    sx + SMOOTHING * (x - sx),
                    sy + SMOOTHING * (y - sy)
                )

            point = (
                int(state["smooth"][0]),
                int(state["smooth"][1])
            )

            state["last_seen"] = now

            # --------------------------------
            # INDEX FINGER = START DRAWING
            # --------------------------------
            if not state["drawing"]:
                state["trail"].clear()
                state["drawing"] = True

            trail = state["trail"]

            if trail:
                distance = math.dist(point, trail[-1])

                if distance > max(100, math.hypot(width, height) * 0.2):
                    trail.clear()

            trail.append(point)
            fingertips[label] = point

            # Detect circle
            if now > state["cooldown"]:
                circle = detect_circle(list(trail))

                if circle:
                    cx, cy, radius = circle
                    portals[label] = {
                        "center": (float(cx), float(cy)),
                        "radius": radius
                    }
                    trail.clear()
                    state["cooldown"] = now + 2.0

        # ==================================
        # HANDLE LOST HANDS
        # ==================================
        for label, state in hands.items():
            if label not in active_hands:
                if now - state["last_seen"] > DRAW_GRACE:
                    state["trail"].clear()
                    state["smooth"] = None
                    state["drawing"] = False

        # ==================================
        # RENDER MAGIC EFFECTS
        # ==================================
        magic_layer = np.zeros_like(frame)

        # Draw finger trails
        for state in hands.values():
            points = list(state["trail"])

            for j in range(1, len(points)):
                cv2.line(
                    magic_layer,
                    points[j - 1],
                    points[j],
                    GOLD, 3, cv2.LINE_AA
                )

        # Draw portals
        elapsed = now - start_time

        for label, magic in portals.items():
            # Hide when tracking is lost; resume on the same hand's return.
            if now - hands[label]["last_seen"] > 0.15:
                continue
            cx, cy = magic["center"]
            draw_magic_portal(
                magic_layer, (round(cx), round(cy), magic["radius"]), elapsed
            )

        # Draw particles
        update_particles(magic_layer)

        # Strong cinematic glow
        glow = cv2.GaussianBlur(
            magic_layer, (41, 41), 0
        )

        frame = cv2.addWeighted(
            frame, 1.0, glow, 1.1, 0
        )

        frame = cv2.addWeighted(
            frame, 1.0, magic_layer, 1.0, 0
        )

        # ==================================
        # MAGIC FINGERTIPS
        # ==================================
        for label, point in fingertips.items():
            cv2.circle(
                frame, point, 12,
                ORANGE, -1, cv2.LINE_AA
            )

            cv2.circle(
                frame, point, 5,
                WHITE_GOLD, -1, cv2.LINE_AA
            )

            cv2.putText(
                frame,
                f"{label} MAGIC",
                (point[0] + 15, point[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, GOLD, 2
            )

        # ==================================
        # DISPLAY INFORMATION
        # ==================================
        cv2.putText(
            frame,
            "DOCTOR STRANGE - HAND MAGIC",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, GOLD, 2
        )

        cv2.putText(
            frame,
            "Draw circle: SUMMON | Move hand: CARRY MAGIC",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 2
        )

        cv2.putText(
            frame,
            "C: Clear | Q: Quit",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 2
        )

        cv2.putText(
            frame,
            f"Summoned Magic Circles: {len(portals)} / 2",
            (20, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, GOLD, 2
        )

        for j, (label, gesture) in enumerate(gestures.items()):
            cv2.putText(
                frame,
                f"{label}: {gesture}",
                (20, 170 + j * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, GOLD, 2
            )

        cv2.imshow("Doctor Strange Portal", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("c"):
            portals.clear()
            particles.clear()

            for state in hands.values():
                state["trail"].clear()
                state["smooth"] = None
                state["drawing"] = False
                state["last_draw"] = 0

cap.release()
cv2.destroyAllWindows()
