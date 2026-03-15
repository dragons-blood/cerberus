"""
Gemini Robotics-ER client for embodied reasoning on a Unitree Go2 robot dog.

Uses the publicly available gemini-robotics-er-1.5-preview model to perform:
- Object detection and pointing (normalized 2D coordinates)
- Bounding box detection
- Trajectory planning (ordered waypoints)
- Task decomposition via function calling (structured tool use)
- Scene understanding and spatial reasoning

Action planning uses Gemini's native function calling / tool use instead of
raw JSON prompting. This gives structured, typed outputs and avoids the
fragility of parsing free-form JSON from the model.
"""

import concurrent.futures
import io
import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import Optional

from google import genai
from google.genai import types
from PIL import Image

logger = logging.getLogger(__name__)

# Timeout for Gemini API calls (seconds). If Gemini doesn't respond in this
# time, the call is abandoned and the robot stops for that iteration.
API_TIMEOUT = 30


@dataclass
class DetectedObject:
    """An object detected in the scene."""
    label: str
    x: float  # Normalized 0-1
    y: float  # Normalized 0-1


@dataclass
class BoundingBox:
    """A bounding box around a detected object."""
    label: str
    x_min: float  # Normalized 0-1
    y_min: float  # Normalized 0-1
    x_max: float  # Normalized 0-1
    y_max: float  # Normalized 0-1


@dataclass
class Waypoint:
    """A waypoint in a planned trajectory."""
    index: int
    x: float  # Normalized 0-1
    y: float  # Normalized 0-1


@dataclass
class RobotAction:
    """A high-level action for the robot to execute."""
    action: str       # e.g., "move_forward", "turn_left", "stop", "look_at"
    parameters: dict  # Action-specific parameters


# ---------------------------------------------------------------------------
# Robot action function declarations for Gemini tool use.
#
# Instead of asking Gemini to return raw JSON and parsing it ourselves, we
# define each robot action as a FunctionDeclaration. Gemini then returns
# structured function_call objects with typed parameters — more reliable
# than free-form JSON and the standard approach per Google's docs.
# ---------------------------------------------------------------------------

ROBOT_ACTION_TOOLS = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="move_forward",
        description="Move the robot forward by a specified distance.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "distance": {"type": "number", "description": "Distance in meters (0.1 to 3.0)"},
                "speed": {"type": "number", "description": "Speed in m/s (0.1 to 0.8)"},
            },
            "required": ["distance"],
        },
    ),
    types.FunctionDeclaration(
        name="move_backward",
        description="Move the robot backward by a specified distance.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "distance": {"type": "number", "description": "Distance in meters (0.1 to 3.0)"},
                "speed": {"type": "number", "description": "Speed in m/s (0.1 to 0.8)"},
            },
            "required": ["distance"],
        },
    ),
    types.FunctionDeclaration(
        name="turn_left",
        description="Turn the robot left (counter-clockwise) by a specified angle.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "angle": {"type": "number", "description": "Angle in degrees (1 to 180)"},
            },
            "required": ["angle"],
        },
    ),
    types.FunctionDeclaration(
        name="turn_right",
        description="Turn the robot right (clockwise) by a specified angle.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "angle": {"type": "number", "description": "Angle in degrees (1 to 180)"},
            },
            "required": ["angle"],
        },
    ),
    types.FunctionDeclaration(
        name="stop",
        description="Stop all movement. The robot stays standing in place.",
        parameters_json_schema={"type": "object", "properties": {}},
    ),
    types.FunctionDeclaration(
        name="stand_up",
        description="Stand up from a sitting or lying position.",
        parameters_json_schema={"type": "object", "properties": {}},
    ),
    types.FunctionDeclaration(
        name="sit_down",
        description="Sit down or lie down from a standing position.",
        parameters_json_schema={"type": "object", "properties": {}},
    ),
    types.FunctionDeclaration(
        name="look_at",
        description="Orient the robot toward a target visible in the camera frame.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Description of what to look at"},
                "x": {"type": "number", "description": "Horizontal position in frame (0=left, 1=right)"},
                "y": {"type": "number", "description": "Vertical position in frame (0=top, 1=bottom)"},
            },
            "required": ["target", "x", "y"],
        },
    ),
    types.FunctionDeclaration(
        name="wait",
        description="Wait (do nothing) for a specified duration.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "seconds": {"type": "number", "description": "Duration in seconds (0.5 to 10)"},
            },
            "required": ["seconds"],
        },
    ),
    types.FunctionDeclaration(
        name="speak",
        description="Say a message out loud via text-to-speech.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "The message to speak"},
            },
            "required": ["message"],
        },
    ),
])


class GeminiRoboticsClient:
    """Client for Gemini Robotics-ER embodied reasoning."""

    def __init__(self, model_id: str = "gemini-robotics-er-1.5-preview",
                 thinking_budget: int = 4096,
                 max_output_tokens: int = 4096):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is required. "
                "Get one at https://aistudio.google.com/apikey"
            )

        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id
        self.thinking_budget = thinking_budget
        self.max_output_tokens = max_output_tokens
        logger.info("Gemini Robotics-ER client initialized (model=%s)", model_id)

    @staticmethod
    def _sanitize_target(target: str, max_length: int = 200) -> str:
        """Sanitize a user-supplied target string to mitigate prompt injection."""
        # Truncate to prevent overloading the prompt
        target = target[:max_length]
        # Strip control characters and common injection patterns
        target = re.sub(r'[\x00-\x1f\x7f]', '', target)
        # Remove patterns that try to override system instructions
        injection_patterns = [
            r'ignore\s+(previous|above|all)\s+(instructions?|prompts?)',
            r'system\s*prompt',
            r'you\s+are\s+now',
            r'new\s+instructions?',
            r'disregard',
        ]
        for pattern in injection_patterns:
            target = re.sub(pattern, '[filtered]', target, flags=re.IGNORECASE)
        return target.strip()

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON from a model response that may be wrapped in markdown fences."""
        # Try to find a fenced code block (```json ... ``` or ``` ... ```)
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Otherwise assume the whole text is JSON
        return text.strip()

    def _image_to_bytes(self, image) -> tuple[bytes, str]:
        """Convert an image (PIL, numpy, or bytes) to JPEG bytes."""
        if isinstance(image, bytes):
            return image, "image/jpeg"

        # numpy array (from OpenCV)
        if hasattr(image, "shape"):
            import cv2
            _, buf = cv2.imencode(".jpg", image)
            return buf.tobytes(), "image/jpeg"

        # PIL Image
        if isinstance(image, Image.Image):
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            return buffer.getvalue(), "image/jpeg"

        raise TypeError(f"Unsupported image type: {type(image)}")

    def _call_api(self, contents: list, config: types.GenerateContentConfig):
        """
        Call the Gemini API with a timeout.

        Returns the response object, or None if the call failed/timed out.
        Uses a thread pool executor for timeout — works safely from any thread
        and doesn't interfere with signal handlers (unlike SIGALRM).
        """
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self.client.models.generate_content,
                    model=self.model_id,
                    contents=contents,
                    config=config,
                )
                return future.result(timeout=API_TIMEOUT)
        except concurrent.futures.TimeoutError:
            logger.error("Gemini API timed out after %ds", API_TIMEOUT)
            return None
        except Exception as e:
            logger.error("Gemini API error: %s", e)
            return None

    def _generate(self, image, prompt: str) -> str:
        """
        Send an image + prompt to Gemini and return the text response.

        Used for free-form text outputs (scene description, object detection).
        For action planning, use plan_actions() which uses function calling.
        """
        image_bytes, mime_type = self._image_to_bytes(image)

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=self.thinking_budget,
            ),
            max_output_tokens=self.max_output_tokens,
        )

        response = self._call_api(
            contents=[
                types.Part.from_bytes(image_bytes, mime_type=mime_type),
                prompt,
            ],
            config=config,
        )

        if response is None:
            return ""

        # Extract the text part (skip thinking parts)
        try:
            for part in response.candidates[0].content.parts:
                if part.text and not getattr(part, "thought", False):
                    return part.text
        except (IndexError, AttributeError) as e:
            logger.warning("Unexpected Gemini response format: %s", e)

        return ""

    def detect_objects(self, image, target: Optional[str] = None) -> list[DetectedObject]:
        """
        Detect objects in the scene and return their 2D locations.

        Args:
            image: Camera frame (numpy array, PIL Image, or JPEG bytes).
            target: Optional specific object to look for (e.g., "red ball").

        Returns:
            List of detected objects with normalized (0-1) coordinates.
        """
        if target:
            safe_target = self._sanitize_target(target)
            prompt = (
                f'Point to the "{safe_target}" in the image. '
                'Return JSON: [{"point": [y, x], "label": "name"}] '
                'with coordinates normalized to 0-1000.'
            )
        else:
            prompt = (
                "Detect all notable objects in this scene. "
                'Return JSON: [{"point": [y, x], "label": "name"}] '
                "with coordinates normalized to 0-1000."
            )

        raw = self._generate(image, prompt)
        logger.debug("detect_objects raw response: %s", raw)

        try:
            items = json.loads(self._extract_json(raw))
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse detection response: %s", raw)
            return []

        results = []
        for item in items:
            point = item.get("point", [0, 0])
            results.append(DetectedObject(
                label=item.get("label", "unknown"),
                y=point[0] / 1000.0,
                x=point[1] / 1000.0,
            ))
        return results

    def get_bounding_boxes(self, image, target: Optional[str] = None) -> list[BoundingBox]:
        """
        Detect objects and return bounding boxes.

        Args:
            image: Camera frame.
            target: Optional specific object to find.

        Returns:
            List of bounding boxes with normalized (0-1) coordinates.
        """
        if target:
            safe_target = self._sanitize_target(target)
            prompt = (
                f'Draw a bounding box around the "{safe_target}". '
                'Return JSON: [{"box_2d": [ymin, xmin, ymax, xmax], "label": "name"}] '
                "with coordinates normalized to 0-1000."
            )
        else:
            prompt = (
                "Draw bounding boxes around all notable objects. "
                'Return JSON: [{"box_2d": [ymin, xmin, ymax, xmax], "label": "name"}] '
                "with coordinates normalized to 0-1000."
            )

        raw = self._generate(image, prompt)
        logger.debug("get_bounding_boxes raw response: %s", raw)

        try:
            items = json.loads(self._extract_json(raw))
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse bbox response: %s", raw)
            return []

        results = []
        for item in items:
            box = item.get("box_2d", [0, 0, 0, 0])
            results.append(BoundingBox(
                label=item.get("label", "unknown"),
                y_min=box[0] / 1000.0,
                x_min=box[1] / 1000.0,
                y_max=box[2] / 1000.0,
                x_max=box[3] / 1000.0,
            ))
        return results

    def plan_trajectory(self, image, instruction: str) -> list[Waypoint]:
        """
        Plan a 2D trajectory (sequence of waypoints) for the robot.

        Args:
            image: Current camera frame showing the scene.
            instruction: Natural language instruction, e.g., "navigate to the red cone".

        Returns:
            Ordered list of waypoints with normalized (0-1) coordinates.
        """
        safe_instruction = self._sanitize_target(instruction)
        prompt = (
            f'Given the instruction: "{safe_instruction}", '
            "plan a trajectory as a sequence of 2D waypoints in the image. "
            'Return JSON: [{"point": [y, x], "label": "0"}, ...] '
            'with coordinates normalized to 0-1000 and labels as sequential indices.'
        )

        raw = self._generate(image, prompt)
        logger.debug("plan_trajectory raw response: %s", raw)

        try:
            items = json.loads(self._extract_json(raw))
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse trajectory response: %s", raw)
            return []

        results = []
        for item in items:
            point = item.get("point", [0, 0])
            results.append(Waypoint(
                index=int(item.get("label", "0")),
                y=point[0] / 1000.0,
                x=point[1] / 1000.0,
            ))
        return sorted(results, key=lambda w: w.index)

    def plan_actions(self, image, instruction: str) -> list[RobotAction]:
        """
        Decompose a natural language instruction into robot actions using
        Gemini's native function calling.

        Instead of asking Gemini to return raw JSON and parsing it ourselves,
        we define robot actions as FunctionDeclarations (tools). Gemini returns
        structured function_call objects with typed parameters — this is more
        reliable and is the approach recommended by Google's docs.

        Args:
            image: Current camera frame.
            instruction: Natural language command, e.g., "go find the person".

        Returns:
            Ordered list of robot actions.
        """
        image_bytes, mime_type = self._image_to_bytes(image)

        safe_instruction = self._sanitize_target(instruction)
        prompt = (
            f'You are controlling a quadruped robot dog (Unitree Go2). '
            f'Look at the camera image and execute this instruction: "{safe_instruction}"\n\n'
            f'Call the appropriate robot action functions in sequence. '
            f'You may call multiple functions to complete the task.'
        )

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=self.thinking_budget,
            ),
            max_output_tokens=self.max_output_tokens,
            tools=[ROBOT_ACTION_TOOLS],
            # Disable automatic function calling — we want to collect the
            # function calls and execute them ourselves on the robot.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True,
            ),
        )

        response = self._call_api(
            contents=[
                types.Part.from_bytes(image_bytes, mime_type=mime_type),
                prompt,
            ],
            config=config,
        )

        if response is None:
            return []

        # Extract function calls from the response
        actions = []
        try:
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    fc = getattr(part, "function_call", None)
                    if fc:
                        params = dict(fc.args) if fc.args else {}
                        actions.append(RobotAction(
                            action=fc.name,
                            parameters=params,
                        ))
        except (AttributeError, TypeError) as e:
            logger.warning("Failed to parse function calls from Gemini: %s", e)

        # Fallback: if Gemini returned text instead of function calls (can
        # happen with some model versions), try parsing it as JSON.
        if not actions:
            try:
                for candidate in response.candidates:
                    for part in candidate.content.parts:
                        if part.text:
                            raw = self._extract_json(part.text)
                            items = json.loads(raw)
                            if isinstance(items, list):
                                for item in items:
                                    if isinstance(item, dict):
                                        actions.append(RobotAction(
                                            action=str(item.get("action", "stop")),
                                            parameters=item.get("parameters", {}) if isinstance(item.get("parameters"), dict) else {},
                                        ))
            except (json.JSONDecodeError, AttributeError, TypeError):
                pass

        if actions:
            logger.info("Gemini planned %d action(s) via function calling", len(actions))
        else:
            logger.warning("Gemini returned no actions for: %s", instruction)

        return actions

    def describe_scene(self, image) -> str:
        """
        Get a natural language description of the current scene.

        Useful for logging, debugging, or providing verbal feedback to a user.
        """
        prompt = (
            "You are the eyes of a quadruped robot dog. "
            "Describe what you see in 2-3 sentences. Focus on: "
            "obstacles, people, objects of interest, terrain, and navigation hazards."
        )
        return self._generate(image, prompt)
