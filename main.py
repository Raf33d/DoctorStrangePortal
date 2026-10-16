
import cv2
import mediapipe as mp
import numpy as np
import math
import random
import time
from magic_control import HandMagic, get_gesture

# ==========================================
# CONFIGURATION
# ==========================================
WIDTH, HEIGHT = 1280, 720

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

hands = {label: HandMagic() for label in ("Left", "Right")}
particles = []

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

            gesture = get_gesture(hand, width, height)
            gestures[label] = gesture
            palm = (
                sum(hand[j].x for j in (0, 5, 9, 13, 17)) / 5 * width,
                sum(hand[j].y for j in (0, 5, 9, 13, 17)) / 5 * height
            )
            point = state.update(
                gesture, (hand[8].x * width, hand[8].y * height), palm,
                now, dt, (width, height)
            )
            if point is not None:
                fingertips[label] = point

        for label, state in hands.items():
            if label not in active_hands:
                state.lost(now)

        # ==================================
        # RENDER MAGIC EFFECTS
        # ==================================
        magic_layer = np.zeros_like(frame)

        # Draw finger trails
        for state in hands.values():
            points = list(state.trail)

            for j in range(1, len(points)):
                cv2.line(
                    magic_layer,
                    points[j - 1],
                    points[j],
                    GOLD, 3, cv2.LINE_AA
                )

        # Draw portals
        elapsed = now - start_time

        for state in hands.values():
            if not state.visible(now):
                continue
            magic = state.magic
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
            "Draw: SUMMON | Open hand: HOLD | Fist: DISMISS",
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
            f"Summoned Magic Circles: {sum(s.magic is not None for s in hands.values())} / 2",
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
            particles.clear()
            for state in hands.values():
                state.clear()

cap.release()
cv2.destroyAllWindows()
