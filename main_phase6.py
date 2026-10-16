"""Phase 6: two-hand, fiery New York webcam portals."""
import argparse
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np

from portal_effects import GOLD, HOT, MAX_PORTALS, HandState, PortalRenderer, get_gesture

ROOT = Path(__file__).resolve().parent


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--quality', choices=('cinematic', 'smooth'), default='cinematic')
    parser.add_argument('--destination', type=Path, default=ROOT/'assets'/'new_york.jpg')
    parser.add_argument('--demo', action='store_true', help='Show portals without using a webcam/model')
    parser.add_argument('--preview', type=Path, help='Save a camera-free preview and exit')
    return parser.parse_args()


def draw_hud(frame, renderer, fps, gestures, demo=False):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (14, 14), (min(w-1, 690), 151), (18, 15, 14), -1)
    lines = [
        ('PHASE 6 / NEW YORK PORTAL', GOLD),
        ('Index finger: draw a circle | Fist: stop drawing', (235, 235, 235)),
        ('C: close all | V: quality | S: screenshot | Q / Esc: quit', (235, 235, 235)),
        (f'{renderer.quality.upper()}  |  ' + ('DEMO' if demo else f'{fps:.0f} FPS')
         + f'  |  Portals {len(renderer.portals)}/{MAX_PORTALS}', GOLD),
    ]
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (27, 42+i*29), cv2.FONT_HERSHEY_SIMPLEX, .57, color, 1, cv2.LINE_AA)
    status = 'DEMO MODE - camera disabled' if demo else '  |  '.join(f'{k}: {v}' for k, v in gestures.items())
    if not status:
        status = 'Show your hand, extend only your index finger, then draw a circle.'
    cv2.putText(frame, status, (25, h-25), cv2.FONT_HERSHEY_SIMPLEX, .55, HOT, 1, cv2.LINE_AA)
    if len(renderer.portals) >= MAX_PORTALS:
        cv2.putText(frame, 'Portal limit reached. Press C to start again.', (25, h-55),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, GOLD, 1, cv2.LINE_AA)


def demo_background():
    frame = np.zeros((720, 1280, 3), np.uint8)
    frame[:] = (28, 21, 17)
    for x in range(0, 1280, 80):
        cv2.line(frame, (x, 160), (x, 720), (42, 33, 25), 1)
    for y in range(160, 720, 80):
        cv2.line(frame, (0, y), (1280, y), (42, 33, 25), 1)
    return frame


def main():
    args = arguments()
    image = cv2.imread(str(args.destination))
    if image is None:
        raise RuntimeError(f'Destination image missing or invalid: {args.destination}. See README.md.')
    renderer = PortalRenderer(image, args.quality)
    states = {name: HandState() for name in ('Left', 'Right')}
    demo = args.demo or args.preview is not None
    cap = None
    landmarker = None
    try:
        if not demo:
            model = ROOT/'models'/'hand_landmarker.task'
            if not model.is_file():
                raise RuntimeError(f'Hand model missing: {model}. See README.md for the download command.')
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=2, min_hand_detection_confidence=.5,
                min_hand_presence_confidence=.5, min_tracking_confidence=.5)
            landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
            cap = cv2.VideoCapture(args.camera)
            if not cap.isOpened():
                raise RuntimeError('Webcam unavailable. Close other camera apps or try --camera 1.')
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        start = previous = time.monotonic()
        last_timestamp = -1
        fps = 0.
        if demo:
            renderer.add((470, 390, 190), start)
            renderer.add((920, 410, 130), start+.12)
        preview_index = 0
        while True:
            now = time.monotonic()
            dt = now-previous
            previous = now
            gestures = {}
            if demo:
                frame = demo_background()
                if args.preview:
                    # Deterministic simulation time for reproducible visual QA.
                    now = start+preview_index/30
                    dt = 1/30
                    preview_index += 1
            else:
                success, frame = cap.read()
                if not success:
                    raise RuntimeError('Webcam stopped delivering frames. Check its connection and restart.')
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp = max(int((now-start)*1000), last_timestamp+1)
                last_timestamp = timestamp
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp)
                seen = set()
                for hand, handedness in zip(result.hand_landmarks, result.handedness):
                    # MediaPipe labels assume a mirrored selfie image, as supplied above.
                    label = handedness[0].category_name
                    if label in seen or label not in states:
                        continue
                    seen.add(label)
                    gesture = get_gesture(hand)
                    gestures[label] = gesture
                    point, circle = states[label].update(
                        gesture, (hand[8].x*w, hand[8].y*h), now)
                    if circle:
                        renderer.add(circle, now)
                    if point:
                        cv2.circle(frame, point, 7, GOLD, -1, cv2.LINE_AA)
                for label, state in states.items():
                    if label not in seen and now-state.last_seen > .15:
                        state.reset()
            renderer.render(frame, now, dt, states.values())
            if dt > 0:
                fps = .9*fps+.1/dt if fps else 1/dt
            draw_hud(frame, renderer, fps, gestures, demo)
            if args.preview:
                if preview_index < 45:
                    continue
                args.preview.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(args.preview), frame):
                    raise RuntimeError(f'Unable to save preview: {args.preview}')
                print(f'Preview saved: {args.preview.resolve()}')
                break
            cv2.imshow('Doctor Strange - Interdimensional Portal', frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if cv2.getWindowProperty('Doctor Strange - Interdimensional Portal', cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord('c'):
                renderer.clear(states.values())
            elif key == ord('v'):
                renderer.quality = 'smooth' if renderer.quality == 'cinematic' else 'cinematic'
            elif key == ord('s'):
                folder = ROOT/'captures'
                folder.mkdir(exist_ok=True)
                output = folder/f'portal_{time.time_ns()}.png'
                if cv2.imwrite(str(output), frame):
                    print(f'Screenshot saved: {output}')
                else:
                    print(f'Could not save screenshot: {output}')
    finally:
        if cap is not None:
            cap.release()
        if landmarker is not None:
            landmarker.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
