"""
gesture_control.py
------------------
Handles MediaPipe hand landmark detection and gesture recognition.
Translates detected gestures into robot movement commands.
"""

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


# ─────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────

@dataclass
class GestureResult:
    """Holds the outcome of a single-frame gesture analysis."""
    command: str          # Human-readable robot command
    gesture_name: str     # Internal gesture label
    confidence: float     # 0.0 – 1.0 confidence score (reserved for future ML upgrade)
    landmarks_detected: bool


# ─────────────────────────────────────────────
#  Landmark index constants (MediaPipe Hands)
# ─────────────────────────────────────────────

# Fingertip landmark IDs
THUMB_TIP   = 4
INDEX_TIP   = 8
MIDDLE_TIP  = 12
RING_TIP    = 16
PINKY_TIP   = 20

# Second-knuckle (PIP) landmark IDs – used to judge finger extension
INDEX_PIP   = 6
MIDDLE_PIP  = 10
RING_PIP    = 14
PINKY_PIP   = 18

# MCP (knuckle) base landmarks
INDEX_MCP   = 5
PINKY_MCP   = 17
WRIST       = 0
THUMB_IP    = 3   # Thumb interphalangeal joint


# ─────────────────────────────────────────────
#  Command labels
# ─────────────────────────────────────────────

COMMAND_FORWARD  = "⬆️  MOVE FORWARD"
COMMAND_BACKWARD = "⬇️  MOVE BACKWARD"
COMMAND_LEFT     = "⬅️  MOVE LEFT"
COMMAND_RIGHT    = "➡️  MOVE RIGHT"
COMMAND_STOP     = "🛑  STOP"
COMMAND_NONE     = "⏳  Waiting for gesture…"


# ─────────────────────────────────────────────
#  GestureDetector class
# ─────────────────────────────────────────────

class GestureDetector:
    """
    Wraps MediaPipe Hands and exposes a simple `process_frame()` API.

    Usage
    -----
    detector = GestureDetector()
    result, annotated_frame = detector.process_frame(bgr_frame)
    """

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
    ):
        self.mp_hands    = mp.solutions.hands
        self.mp_drawing  = mp.solutions.drawing_utils
        self.mp_styles   = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # Custom drawing specs for a cleaner look
        self._landmark_style = self.mp_drawing.DrawingSpec(
            color=(0, 255, 180), thickness=2, circle_radius=4
        )
        self._connection_style = self.mp_drawing.DrawingSpec(
            color=(255, 255, 0), thickness=2
        )

    # ── Public API ────────────────────────────

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[GestureResult, np.ndarray]:
        """
        Detect hands in *frame*, draw landmarks, and classify the gesture.

        Parameters
        ----------
        frame : np.ndarray
            BGR image from OpenCV.

        Returns
        -------
        (GestureResult, annotated_frame)
        """
        # Flip so it feels like a mirror
        frame = cv2.flip(frame, 1)

        # MediaPipe works in RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True

        annotated = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if not results.multi_hand_landmarks:
            return GestureResult(
                command=COMMAND_NONE,
                gesture_name="none",
                confidence=0.0,
                landmarks_detected=False,
            ), annotated

        # Use the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw skeleton on frame
        self._draw_landmarks(annotated, hand_landmarks)

        # Classify gesture from landmark positions
        gesture_name, command = self._classify(hand_landmarks, annotated.shape)

        return GestureResult(
            command=command,
            gesture_name=gesture_name,
            confidence=1.0,
            landmarks_detected=True,
        ), annotated

    def release(self):
        """Free MediaPipe resources."""
        self.hands.close()

    # ── Drawing helpers ───────────────────────

    def _draw_landmarks(self, frame: np.ndarray, hand_landmarks) -> None:
        """Draw hand skeleton and landmark dots on *frame* in-place."""
        self.mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self._landmark_style,
            self._connection_style,
        )

    # ── Gesture classification ────────────────

    def _classify(
        self, hand_landmarks, frame_shape: Tuple[int, int, int]
    ) -> Tuple[str, str]:
        """
        Determine which gesture is being made.

        Returns (gesture_name, command_string).

        Gesture priority order:
          1. Open palm  → STOP
          2. Thumb up   → FORWARD
          3. Thumb down → BACKWARD
          4. Point left → LEFT
          5. Point right→ RIGHT
          6. Fallback   → NONE
        """
        h, w, _ = frame_shape
        lm = hand_landmarks.landmark  # list of NormalizedLandmark

        def px(idx):
            """Return (x_pixels, y_pixels) for landmark *idx*."""
            return int(lm[idx].x * w), int(lm[idx].y * h)

        # ── 1. Open palm ──────────────────────────────────────────────────
        if self._is_open_palm(lm):
            return "open_palm", COMMAND_STOP

        # ── 2. Thumb up / down ────────────────────────────────────────────
        thumb_state = self._thumb_direction(lm)
        if thumb_state == "up":
            return "thumb_up", COMMAND_FORWARD
        if thumb_state == "down":
            return "thumb_up_down", COMMAND_BACKWARD

        # ── 3. Index finger pointing left / right ─────────────────────────
        direction = self._pointing_direction(lm)
        if direction == "left":
            return "point_left", COMMAND_LEFT
        if direction == "right":
            return "point_right", COMMAND_RIGHT

        return "unknown", COMMAND_NONE

    # ── Individual gesture detectors ─────────

    def _is_open_palm(self, lm) -> bool:
        """
        Open palm: all four fingers extended.
        A finger is extended when its tip y-coord is ABOVE its PIP y-coord
        (smaller y = higher on screen).
        """
        finger_pairs = [
            (INDEX_TIP,  INDEX_PIP),
            (MIDDLE_TIP, MIDDLE_PIP),
            (RING_TIP,   RING_PIP),
            (PINKY_TIP,  PINKY_PIP),
        ]
        extended = sum(
            1 for tip, pip in finger_pairs if lm[tip].y < lm[pip].y
        )
        return extended >= 4

    def _thumb_direction(self, lm) -> Optional[str]:
        """
        Detect thumb-up / thumb-down.

        Logic:
          - Other four fingers must be curled (tips below their PIPs).
          - Thumb tip above wrist  → "up"
          - Thumb tip below wrist  → "down"
        """
        # Check that index→pinky are curled
        fingers_curled = all(
            lm[tip].y > lm[pip].y
            for tip, pip in [
                (INDEX_TIP,  INDEX_PIP),
                (MIDDLE_TIP, MIDDLE_PIP),
                (RING_TIP,   RING_PIP),
                (PINKY_TIP,  PINKY_PIP),
            ]
        )
        if not fingers_curled:
            return None

        # Compare thumb tip vs wrist height
        if lm[THUMB_TIP].y < lm[WRIST].y - 0.05:
            return "up"
        if lm[THUMB_TIP].y > lm[WRIST].y + 0.05:
            return "down"
        return None

    def _pointing_direction(self, lm) -> Optional[str]:
        """
        Detect index-finger pointing left or right.

        Logic:
          - Index finger extended horizontally (tip x differs from MCP x).
          - Middle, ring, pinky are curled.
          - Direction determined by whether tip is left or right of MCP.
        """
        # Middle, ring, pinky must be curled
        side_fingers_curled = all(
            lm[tip].y > lm[pip].y
            for tip, pip in [
                (MIDDLE_TIP, MIDDLE_PIP),
                (RING_TIP,   RING_PIP),
                (PINKY_TIP,  PINKY_PIP),
            ]
        )
        if not side_fingers_curled:
            return None

        # Index finger must be more horizontal than vertical
        dx = lm[INDEX_TIP].x - lm[INDEX_MCP].x
        dy = abs(lm[INDEX_TIP].y - lm[INDEX_MCP].y)

        if abs(dx) < 0.08 or dy > abs(dx) * 0.9:
            return None   # Not horizontal enough

        # NOTE: frame is already flipped → left in frame = left for user
        return "left" if dx < 0 else "right"


# ─────────────────────────────────────────────
#  Overlay utilities (used by app.py)
# ─────────────────────────────────────────────

# Command → accent colour (BGR)
COMMAND_COLORS = {
    COMMAND_FORWARD:  (0,   200, 80),
    COMMAND_BACKWARD: (0,   100, 255),
    COMMAND_LEFT:     (255, 180, 0),
    COMMAND_RIGHT:    (255, 180, 0),
    COMMAND_STOP:     (0,   0,   230),
    COMMAND_NONE:     (180, 180, 180),
}


def draw_command_overlay(frame: np.ndarray, result: GestureResult) -> np.ndarray:
    """
    Render a semi-transparent status banner at the bottom of *frame*
    showing the current robot command.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Dark banner
    banner_h = 60
    cv2.rectangle(overlay, (0, h - banner_h), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    color = COMMAND_COLORS.get(result.command, (200, 200, 200))

    # Clean text (strip emoji for OpenCV)
    clean_text = result.command.encode("ascii", "ignore").decode().strip()

    cv2.putText(
        frame,
        clean_text,
        (20, h - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2,
        cv2.LINE_AA,
    )

    # Landmark indicator dot
    if result.landmarks_detected:
        cv2.circle(frame, (w - 30, h - 30), 10, (0, 255, 120), -1)
    else:
        cv2.circle(frame, (w - 30, h - 30), 10, (60, 60, 60), -1)

    return frame