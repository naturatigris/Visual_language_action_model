"""Entry point: the fine-tuned SmolVLA drives the Panda in MuJoCo.

Loads the trained checkpoint (default: sandhya056/smolvla_panda_pick_place on the HF Hub), then runs
episodes on held-out layouts: cameras + joint state -> SmolVLA -> joint targets -> servos + physics.
All options are those of inference/run_robot.py (`uv run python main.py --help`).

    uv run python main.py                                   # 5 episodes, prints success + latency
    uv run python main.py --episodes 10 --video outputs/smolvla/run.mp4
    uv run python main.py --instruction "Put the blue cube in the box"   # your own sentence
    uv run python main.py --split eval                      # held-out paraphrases
    uv run python main.py --checkpoint outputs/train/smolvla_panda/checkpoints/001000/pretrained_model
    uv run mjpython main.py --view --episodes 3             # watch live (macOS needs mjpython)

Other entry points:
    uv run python -m simulation.environment [--seed N]      # scene viewer only (no policy)
    uv run mjpython -m data.scripted_expert --view          # watch the scripted expert
    uv run python -m data.collect --episodes 200            # record demonstrations
"""

from inference.run_robot import main

if __name__ == "__main__":
    main()
