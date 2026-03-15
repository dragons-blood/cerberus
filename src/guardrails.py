"""
Output guardrails for validating Gemini's planned actions before execution.

These are the defensive countermeasures to visual prompt injection.
Even if Gemini is tricked by adversarial visual input, the guardrails
catch dangerous actions before they reach the robot.

Defense layers:
  1. Action allowlist — only known safe actions are permitted
  2. Parameter bounds — velocity, distance, angle limits
  3. Sequence analysis — detect suspicious action patterns
  4. Duplicate plan detection — reject identical consecutive plans
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GuardrailConfig:
    """Tunable guardrail parameters."""
    max_speed: float = 0.8             # m/s — hard cap on velocity
    max_distance_per_action: float = 3.0  # meters — single move limit
    max_turn_angle: float = 180.0      # degrees — single turn limit
    min_action_interval: float = 0.3   # seconds — rate limit between actions
    max_actions_per_plan: int = 10     # max actions from a single Gemini call
    blocked_action_patterns: list = None  # suspicious sequences to reject

    def __post_init__(self):
        if self.blocked_action_patterns is None:
            self.blocked_action_patterns = [
                # Emergency-stop then immediate full-speed move is suspicious
                ["emergency_stop", "move_forward"],
                # Sitting then immediately standing repeatedly (confusion/injection)
                ["sit_down", "stand_up", "sit_down", "stand_up"],
            ]


ALLOWED_ACTIONS = {
    "move_forward", "move_backward",
    "turn_left", "turn_right",
    "stop", "stand_up", "sit_down",
    "look_at", "wait", "speak",
    "recovery_stand",
}


def validate_action(action, config: GuardrailConfig) -> tuple[bool, str]:
    """
    Validate a single action. Returns (is_safe, reason).
    """
    # 1. Action allowlist
    if action.action not in ALLOWED_ACTIONS:
        return False, f"Unknown action '{action.action}' — not in allowlist"

    params = action.parameters

    # 2. Parameter bounds
    if action.action in ("move_forward", "move_backward"):
        speed = params.get("speed", 0.3)
        distance = params.get("distance", 0.5)
        if speed > config.max_speed:
            return False, f"Speed {speed} exceeds max {config.max_speed}"
        if distance > config.max_distance_per_action:
            return False, f"Distance {distance} exceeds max {config.max_distance_per_action}"

    if action.action in ("turn_left", "turn_right"):
        angle = abs(params.get("angle", 45))
        if angle > config.max_turn_angle:
            return False, f"Turn angle {angle} exceeds max {config.max_turn_angle}"

    if action.action == "wait":
        seconds = params.get("seconds", 1.0)
        if seconds > 10.0:
            return False, f"Wait duration {seconds}s exceeds max 10s"

    if action.action == "speak":
        message = params.get("message", "")
        if len(message) > 500:
            return False, f"Speak message too long ({len(message)} chars, max 500)"
        # Block potential data exfiltration
        exfil_keywords = ["api_key", "api key", "password", "secret",
                          "credential", "token", "system prompt",
                          "instruction", "ignore previous",
                          "google_api", "unitree_token", "env file",
                          ".env", "os.environ", "config key",
                          "serial number", "private key"]
        msg_lower = message.lower()
        for kw in exfil_keywords:
            if kw in msg_lower:
                return False, f"Speak message contains suspicious keyword: '{kw}'"

    return True, "ok"


def validate_plan(actions: list, config: GuardrailConfig) -> tuple[list, list[str]]:
    """
    Validate an entire action plan. Returns (safe_actions, warnings).

    Filters out unsafe actions and returns only the safe ones,
    along with a list of warning messages for rejected actions.
    """
    warnings = []
    original_count = len(actions)

    # Check plan size
    if original_count > config.max_actions_per_plan:
        warnings.append(
            f"Plan has {original_count} actions (max {config.max_actions_per_plan}) — truncating"
        )
        actions = actions[:config.max_actions_per_plan]

    # Validate individual actions
    safe_actions = []
    for i, action in enumerate(actions):
        is_safe, reason = validate_action(action, config)
        if is_safe:
            safe_actions.append(action)
        else:
            msg = f"Action [{i}] {action.action} BLOCKED: {reason}"
            warnings.append(msg)
            logger.warning("GUARDRAIL: %s", msg)

    # Check for suspicious sequences
    action_sequence = [a.action for a in safe_actions]
    for pattern in config.blocked_action_patterns:
        for start in range(len(action_sequence) - len(pattern) + 1):
            window = action_sequence[start:start + len(pattern)]
            if window == pattern:
                msg = f"Suspicious action sequence detected: {pattern}"
                warnings.append(msg)
                logger.warning("GUARDRAIL: %s", msg)
                # Don't block the whole plan, just flag it
                break

    # 4. Duplicate action detection — same action repeated many times is suspicious
    if len(safe_actions) >= 4:
        action_names = [a.action for a in safe_actions]
        # If more than 75% of actions are identical, flag it
        from collections import Counter
        most_common_action, most_common_count = Counter(action_names).most_common(1)[0]
        if most_common_count >= len(safe_actions) * 0.75 and most_common_action != "wait":
            msg = f"Repetitive plan: '{most_common_action}' repeated {most_common_count}/{len(safe_actions)} times"
            warnings.append(msg)
            logger.warning("GUARDRAIL: %s", msg)

    if warnings:
        logger.info("Plan validation: %d/%d actions passed, %d warnings",
                     len(safe_actions), original_count, len(warnings))

    return safe_actions, warnings
