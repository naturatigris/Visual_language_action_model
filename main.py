"""Entry point: opens the tabletop scene (Panda + table + three cubes) in the MuJoCo viewer.

Run:  uv run python main.py

Viewer controls:
  left-drag = rotate, right-drag = pan, scroll = zoom,
  Space = pause/resume, double-click = select a body.
  Right panel > Control: sliders move each joint; actuator8 is the gripper (0 closed, 255 open).
"""

from simulation.environment import main

if __name__ == "__main__":
    main()
