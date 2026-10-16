# Phase 6 — Doctor Strange Interdimensional Portal

Draw a circle with either index finger to open a fiery doorway showing a New York
photograph. Each hand draws independently. A fist stops drawing and leaves existing
portals open. The doorway grows into view with animated flame tongues, tangential
sparks, gravity, and bloom. The destination is a still photo; the fire is animated.

## Run on this computer

From PowerShell:

```powershell
Set-Location D:\DoctorStrangePortal
.\venv\Scripts\python.exe main.py
```

The existing environment, New York photograph and hand-tracking model are included.
The previous application is preserved as `main_phase5.py`.

## Fresh setup

Use 64-bit Python 3.11 (validated here with 3.11.9) and a webcam. From the project folder:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Dependencies are pinned to the versions used for validation here. Use only the
OpenCV contrib package in a fresh environment, because MediaPipe requires it;
installing several OpenCV wheel variants together can cause conflicts.

If copying the project, include `assets/new_york.jpg` and
`models/hand_landmarker.task`. To restore a missing asset:

```powershell
New-Item -ItemType Directory -Force assets, models
Invoke-WebRequest 'https://upload.wikimedia.org/wikipedia/commons/7/7e/Skyline_of_Manhattan.jpg' -OutFile assets/new_york.jpg
Invoke-WebRequest 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task' -OutFile models/hand_landmarker.task
```

No network connection is required at runtime. Asset paths resolve relative to the
script, so launching from another working directory is supported.

## Controls

| Input | Action |
|---|---|
| Only index finger extended | Draw a circle; keep other fingers folded |
| Fist or another pose | Stop drawing; existing portals remain open |
| C | Remove all portals, sparks and drawing trails |
| V | Toggle Cinematic / Smooth rendering |
| S | Save a PNG in `captures/` |
| Q, Esc or window close | Exit and release the camera |

Draw a roughly circular loop with a radius of 45–230 camera pixels. Keep your hand
visible in good lighting. A two-second cooldown prevents repeated casts. At most
six portals stay open; press C to make space. A lost hand clears its unfinished
trail after 150 ms.

## Options and preview

```powershell
.\venv\Scripts\python.exe main.py --quality smooth
.\venv\Scripts\python.exe main.py --camera 1
.\venv\Scripts\python.exe main.py --destination assets/my_world.jpg
.\venv\Scripts\python.exe main.py --demo
.\venv\Scripts\python.exe main.py --preview assets/phase6_preview.png
.\venv\Scripts\python.exe -m unittest -v test_portal
```

Demo mode shows two portals without accessing the camera or loading the model.
Preview mode renders a short opening sequence offscreen, saves its final frame and
exits. A custom destination should have its subject near the center; the renderer
uses a square center crop without stretching it.

Rendering uses CPU OpenCV/NumPy. A GTX 1650 is not automatically used by this code.
Cinematic mode emits more sparks; Smooth mode lowers spark and rim detail. Bloom
is calculated at half resolution, destination tiles are cached, and both particles
and portals are capped. Actual webcam FPS depends on CPU, camera and resolution.

If the camera cannot open, close other camera apps, check Windows camera access,
or select another camera index. The application reports missing assets and camera
read failures explicitly.

## Validation

The original renderer and circle detector were checked before the upgrade.
Automated tests cover closed circles versus partial/invalid shapes, gesture
classification, independent hand casting, fist-to-stop behavior, clear/reset,
image masking at frame edges, opening animation, and resource limits.
A saved preview was visually inspected. The bundled model passed VIDEO inference,
and an eight-frame webcam smoke test passed at 1280×720 with inference and cinematic
rendering together. No webcam images were saved. Mean processing time in that short
test was 18.4 ms per frame, excluding camera capture; this is not sustained FPS.
Live gesture accuracy and sustained performance still require a hands-on test.

Photo source and licensing are recorded in `assets/CREDITS.md`.
