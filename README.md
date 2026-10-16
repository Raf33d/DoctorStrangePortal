# Doctor Strange — Magic Effects

The magic-only version in `main.py` has glowing finger trails,
rotating geometric magic circles, orange sparks and cinematic glow. Summoned circles
now smoothly follow the palm of the hand that created them. There are no
New York images or destination doorways in the active application.

## Run

```powershell
Set-Location D:\DoctorStrangePortal
.\venv\Scripts\python.exe main.py
```

Extend only your index finger and draw a circle to create the magic effect.
Both hands work independently, with one magic circle per hand. After summoning,
move your hand to carry the circle; it follows your palm even when you make a fist
or open your hand. Drawing stops once that hand has a circle. The circle hides when
the hand leaves view and resumes following when that hand returns.
Press C to clear everything and draw new circles; press Q to quit.

Large circles are supported up to a 600-pixel radius, with a longer drawing
history for slower loops. Brief pose or tracking slips of up to 0.4 seconds
preserve the unfinished drawing. A fist still stops drawing immediately.
Keep your fingertip inside the camera frame throughout the loop.

The existing virtual environment and `models/hand_landmarker.task` are required.
For a fresh environment, use Python 3.11 and install `requirements.txt`.

The restored source passed a Python syntax check. Its circle detector and renderer
also passed camera-free checks before the Phase 6 upgrade.

## Saved versions

- `main_phase5.py`: original magic-only source with stationary circles.
- `main_phase6.py`: saved New York doorway experiment.
- `README_phase6.md`: instructions for that experiment; run `main_phase6.py`
  instead of `main.py` when following those instructions.
