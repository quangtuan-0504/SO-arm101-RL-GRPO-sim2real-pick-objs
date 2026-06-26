"""Build the VLM (Qwen-VL) prompt for predicting SO-101 control angles.

This module produces a (system_prompt, user_prompt) pair plus a ready-to-send
``messages`` structure for Qwen-VL. Each joint's angle limits are read
DYNAMICALLY from ``Simulation`` so the prompt always matches the simulation and
never drifts from the actual parameters.

Quick start
-----------
    from sim.simulation import Simulation
    from vlm.prompt_builder import build_messages
    import yaml

    cfg = yaml.safe_load(open("config.yaml"))
    sim = Simulation(cfg)
    messages = build_messages(
        sim,
        target_desc="the red cube (cube_red)",
        wrist_img_b64=...,   # camera 1 image as base64 (data URL)
        stand_img_b64=...,   # camera 2 image as base64 (data URL)
    )
    # messages -> feed into the OpenAI-compatible Qwen-VL client
"""
from __future__ import annotations
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import json

from sim.constants import JOINT_NAMES

# Meaning + positive direction of each joint (so the VLM reasons about the
# correct direction).
JOINT_DESCRIPTIONS = {
    "shoulder_pan":  "Rotate the whole arm about the vertical axis (z). "
                     "Positive turns the body one way, negative the other. "
                     "Used to aim the arm horizontally toward the object.",
    "shoulder_lift": "Raise/lower the shoulder (the upper-arm segment). "
                     "Positive raises the shoulder, negative lowers it toward "
                     "the floor.",
    "elbow_flex":    "Flex/extend the elbow. Adjusts reach (near/far) and the "
                     "height of the gripper.",
    "wrist_flex":    "Flex the wrist up/down. Used to set the gripper's "
                     "approach direction (top-down grasp vs. side grasp).",
    "wrist_roll":    "Rotate the wrist about the gripper axis. Used to align "
                     "the two jaws with the length/orientation of the object.",
    "gripper":       "Gripper opening. LARGE value = open wide, SMALL value "
                     "(near the lower limit) = closed/clamped tight.",
}


def _joint_table_from_spec(joints: list) -> str:
    """Angle-limit table (radians) + home from a spec list [{name,lo,hi,home}]."""
    lines = [
        "| # | Joint name | Limit (radians) | Home | Description |",
        "|---|------------|-----------------|------|-------------|",
    ]
    for i, j in enumerate(joints, 1):
        name = j["name"]
        lines.append(
            f"| {i} | `{name}` | [{j['lo']:.3f}, {j['hi']:.3f}] | "
            f"{j['home']:.2f} | {JOINT_DESCRIPTIONS[name]} |"
        )
    return "\n".join(lines)


def _output_schema_example_from_spec(joints: list) -> str:
    """A FORMAT-ONLY example JSON output.

    IMPORTANT: the angle values here are deliberately NOT the home pose and NOT
    a real solution. They are placeholder numbers whose only job is to show the
    JSON shape. The model must never copy them — it must compute its own angles
    from the images. Using the home pose here caused the model to copy it
    verbatim and never move the arm.
    """
    # Placeholder angles: midpoint between home and the upper limit, rounded
    # oddly, so they look clearly "made up" and differ per joint. They are NOT a
    # valid action — only an illustration of the field shape.
    angles = {}
    for j in joints:
        lo, hi, home = float(j["lo"]), float(j["hi"]), float(j["home"])
        placeholder = home + 0.137 * (hi - home if hi > home else home - lo)
        angles[j["name"]] = round(placeholder, 3)
    example = {
        "reasoning": "A brief description of where the object and the basket "
                     "are in the 2 images, where the gripper is relative to "
                     "them, and the next move.",
        "target_object": "cube_red",
        "phase": "approach",
        "gripper_action": "open",
        "joint_angles_rad": angles,
        "done": False,
        "confidence": 0.0,
    }
    return json.dumps(example, ensure_ascii=False, indent=2)


def _current_pose_block(current_angles: dict | None) -> list[str]:
    """Lines showing the CURRENT target angle of each joint (the real anchor)."""
    if not current_angles:
        return []
    rows = ["| Joint name | Current angle (rad) |",
            "|------------|---------------------|"]
    for name in JOINT_NAMES:
        if name in current_angles:
            rows.append(f"| `{name}` | {float(current_angles[name]):.3f} |")
    return [
        "# CURRENT ARM POSE (where the joints are RIGHT NOW)",
        "These are the joint angles the arm is currently holding. Your output "
        "is the NEXT absolute target. To make the arm move you MUST output "
        "angles that DIFFER from these. Returning these same numbers (or the "
        "home pose, or the example values) keeps the arm frozen and makes no "
        "progress.",
        "",
        "\n".join(rows),
        "",
    ]


def _joint_table(sim) -> str:
    """Angle-limit table (radians) + home pose read dynamically from sim."""
    home = sim.config.get("arm", {}).get("home_pose", {})
    joints = []
    for name in JOINT_NAMES:
        lo, hi = sim.get_ctrl_range(name)
        joints.append({"name": name, "lo": lo, "hi": hi,
                       "home": float(home.get(name, 0.0))})
    return _joint_table_from_spec(joints)


def _output_schema_example(sim) -> str:
    """A valid example JSON output, using the home pose as illustration."""
    home = sim.config.get("arm", {}).get("home_pose", {})
    joints = [{"name": n, "home": float(home.get(n, 0.0))} for n in JOINT_NAMES]
    return _output_schema_example_from_spec(joints)


def build_system_prompt() -> str:
    """System prompt: the VLM's role + invariant rules."""
    return "\n".join([
        "You are a vision-based controller for a 6-DOF SO-101 robot arm in a "
        "MuJoCo simulation environment. Your task is to OBSERVE two camera "
        "images describing the current state, then PREDICT the target rotation "
        "angles (radians) for the 6 joints so the arm GRASPS the requested "
        "object and then CARRIES AND DROPS it INTO THE BASKET.",
        "",
        "The whole task is a pick-and-place cycle:",
        "approach the object -> align -> grasp -> lift -> move above the "
        "basket -> open the gripper to drop the object into the basket. Only "
        "report done=true once the object is inside the basket.",
        "",
        "Mandatory principles:",
        "1. Reason ONLY about what is VISIBLE in the 2 images; do not fabricate "
        "information that cannot be observed.",
        "2. Every output angle must be in radians and MUST BE WITHIN the "
        "allowed limit of each joint. Never exceed the bounds.",
        "3. Control in CLOSED LOOP: each time, propose only one safe, "
        "moderately small move that gets closer to the current phase's target.",
        "4. Only CLOSE the gripper once it is aligned with and close enough to "
        "the object in the image.",
        "5. After the object has been grasped, KEEP the gripper closed "
        "(gripper_action=\"hold\") throughout lifting and moving; only OPEN the "
        "gripper once it is ABOVE the basket's mouth to drop the object into "
        "the correct basket.",
        "6. The ONLY output is a single valid JSON object, with no other text, "
        "no markdown, no ``` blocks.",
        "7. You MUST actually move the arm: the `joint_angles_rad` you output is "
        "the NEXT target pose and MUST DIFFER from the current pose. Never copy "
        "the home pose or the values from the format example — those are "
        "references only. Outputting the current/home angles freezes the arm "
        "and is wrong; compute fresh angles from the images that advance the "
        "task. Only `gripper` may stay the same while a grasp is held.",
    ])


def build_user_prompt(sim, target_desc: str, current_angles: dict | None = None) -> str:
    """User prompt: concrete goal + mechanism description + output format (from sim)."""
    home = sim.config.get("arm", {}).get("home_pose", {})
    joints = []
    for name in JOINT_NAMES:
        lo, hi = sim.get_ctrl_range(name)
        joints.append({"name": name, "lo": lo, "hi": hi,
                       "home": float(home.get(name, 0.0))})
    return build_user_prompt_from_spec(joints, target_desc, current_angles)


def build_user_prompt_from_spec(joints: list, target_desc: str,
                                current_angles: dict | None = None) -> str:
    """User prompt from a static joint spec list [{name,lo,hi,home}] (no sim needed)."""
    joints_csv = ", ".join(f'"{j["name"]}"' for j in joints)
    return "\n".join([
        "# GOAL",
        f"Control the SO-101 arm to GRASP **{target_desc}** then CARRY and DROP "
        "it INTO THE BASKET. The task is complete once the object rests inside "
        "the basket.",
        "",
        *_current_pose_block(current_angles),
        "# TWO-CAMERA CONTEXT",
        "You receive 2 images in this order:",
        "- Image 1 — WRIST CAMERA (wrist_cam): rigidly mounted on the gripper, "
        "a close-up view of the grasping area. Use it to precisely align the "
        "gripper relative to the object (left/right, front/back, whether it is "
        "aligned, whether it is close enough to close the gripper).",
        "- Image 2 — STAND CAMERA (stand_cam): fixed to the side, a wide "
        "horizontal overview. Use it to estimate the arm's overall pose, the "
        "gripper height, and the rough distance to the object AND to the "
        "basket.",
        "",
        "You are NOT given the coordinates of the object or the basket. You "
        "must LOCATE them in space entirely from what you observe in the 2 "
        "images (left/right, far/near, high/low positions of the object, the "
        "basket, and the gripper).",
        "",
        "# ACTUATORS (6 position-controlled joints, units in RADIANS)",
        "Each joint is a position motor: you set an absolute TARGET ANGLE, and "
        "the internal PID controller drives the joint to that angle. The "
        "\"home\" pose is a safe rest pose for reference.",
        "",
        _joint_table_from_spec(joints),
        "",
        "Gripper convention (`gripper`): the LARGER the value the more OPEN, "
        "the SMALLER (toward the lower limit) the more tightly CLOSED.",
        "",
        "# PICK & PLACE-INTO-BASKET PROCEDURE (phases)",
        '- "approach": move the gripper closer to the object, gripper OPEN.',
        '- "align": fine-tune shoulder_pan / wrist_roll / wrist_flex so the '
        "gripper is aligned with and surrounds the object.",
        '- "grasp": the gripper is around the object -> CLOSE the gripper '
        '(gripper_action="close").',
        '- "lift": raise shoulder_lift / elbow_flex to lift the object off the '
        'floor, KEEPING the gripper closed (gripper_action="hold").',
        '- "carry": rotate shoulder_pan and adjust reach to bring the object '
        'ABOVE the basket mouth, still KEEPING the gripper closed '
        '(gripper_action="hold").',
        '- "release": the gripper is above the basket mouth and high enough -> '
        'OPEN the gripper (gripper_action="open") to drop the object into the '
        "basket. Once the object is in the basket, set done=true.",
        "Pick the correct current phase based on what you see in the images. Do "
        "not skip phases; only move to the next phase when the current phase's "
        "condition is met.",
        "",
        "# OUTPUT FORMAT (JSON only, nothing else)",
        "Return exactly the following structure:",
        '- "reasoning": (string) brief reasoning from the 2 images: object '
        "position, basket position, gripper position, and the next step.",
        '- "target_object": (string) the name/identifier of the object being '
        "targeted.",
        '- "phase": one of ["approach","align","grasp","lift","carry",'
        '"release"].',
        '- "gripper_action": "open" | "close" | "hold".',
        '- "joint_angles_rad": (object) target radian angles for ALL 6 joints: '
        f"{joints_csv}.",
        '- "done": (bool) true IF AND ONLY IF the object has been dropped and '
        "rests inside the basket.",
        '- "confidence": (float 0..1) the level of certainty.',
        "",
        "Example of the JSON SHAPE ONLY (the numbers below are placeholders — "
        "DO NOT copy them, compute your own angles from the images):",
        _output_schema_example_from_spec(joints),
        "",
        "Now analyze the 2 images and output the control JSON for the next "
        "step.",
    ])


def build_messages(sim, target_desc: str,
                   wrist_img_b64: str, stand_img_b64: str,
                   current_angles: dict | None = None) -> list:
    """Build OpenAI-compatible (Qwen-VL) messages with 2 images + prompt.

    ``wrist_img_b64`` / ``stand_img_b64`` are data URLs, e.g.:
    ``"data:image/png;base64,<...>"``.
    """
    return _assemble_messages(
        build_user_prompt(sim, target_desc, current_angles),
        wrist_img_b64, stand_img_b64)


def build_messages_from_spec(joints: list, target_desc: str,
                             wrist_img_b64: str, stand_img_b64: str,
                             current_angles: dict | None = None) -> list:
    """Like ``build_messages`` but uses a static joint spec (for the GPU server).

    ``joints`` = [{name, lo, hi, home}] (read from ``dataset/robot_spec.json``).
    ``current_angles`` = {joint_name: rad} the arm's CURRENT target pose, shown
    to the model as the anchor it must move away from.
    """
    return _assemble_messages(
        build_user_prompt_from_spec(joints, target_desc, current_angles),
        wrist_img_b64, stand_img_b64)


def _assemble_messages(user_prompt: str, wrist_img_b64: str,
                       stand_img_b64: str) -> list:
    return [
        {"role": "system", "content": build_system_prompt()},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "text", "text": "Image 1 — WRIST CAMERA (wrist_cam):"},
                {"type": "image_url", "image_url": {"url": wrist_img_b64}},
                {"type": "text", "text": "Image 2 — STAND CAMERA (stand_cam):"},
                {"type": "image_url", "image_url": {"url": stand_img_b64}},
            ],
        },
    ]

if __name__ == "__main__":
    # Local import: only needed when running a test on Mac (with mujoco).
    import yaml
    from sim.simulation import Simulation

    cfg = yaml.safe_load(open("config.yaml"))
    sim = Simulation(cfg)

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(sim, target_desc="the red cube (cube_red)")

    print("=== SYSTEM PROMPT ===")
    print(system_prompt)
    print("\n=== USER PROMPT ===")
    print(user_prompt)
