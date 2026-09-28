#!/usr/bin/env python3

"""
tracker_reid.py

ShopMate human-only multi-person tracker with a stable logical customer ID.

Key behaviour:
- The detector feeding this module should already be PERSON-only.
- Customer selection creates one logical customer identity: CUSTOMER_ID = 1.
- The logical customer ID never changes when the underlying short-term tracker
  creates a new track ID.
- While the selected person remains on the same short-term track, that track is
  followed directly.
- If the selected customer disappears and later reappears, this prototype uses
  a conservative HSV appearance descriptor to reacquire them. Reacquisition
  must be a strong and unambiguous appearance match for two consecutive frames.
- It never chooses an arbitrary nearest person merely because the selected
  track disappeared.
- The existing reid_proxy.onnx is NOT trusted for identity decisions because
  those proxy weights are not a trained person-ReID model.

For a production system, replace the HSV appearance descriptor with a trained
person-ReID model (OSNet / FastReID-class model) while keeping the same API.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


BBox = Tuple[float, float, float, float]

# Stable customer identity shown to the user.
CUSTOMER_ID = 1

# Conservative appearance reacquisition settings.
REACQUIRE_MIN_SIM = 0.82
REACQUIRE_SINGLE_MIN_SIM = 0.86
REACQUIRE_MARGIN = 0.08
REACQUIRE_CONFIRM_FRAMES = 2
APPEARANCE_UPDATE_ALPHA = 0.04


@dataclass
class Detection:
    bbox: BBox
    confidence: float


@dataclass
class Track:
    track_id: int
    bbox: BBox
    embedding: Optional[np.ndarray] = None
    age: int = 0
    misses: int = 0
    hits: int = 1
    is_target: bool = False
    matched_this_frame: bool = True

    @property
    def centroid(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class TrackingResult:
    tracks: List[Track]
    # Stable logical customer ID. This remains CUSTOMER_ID even if the
    # underlying tracker ID changes after disappearance/reappearance.
    target_track_id: Optional[int]
    target_bbox: Optional[BBox]
    target_lost: bool
    frame_index: int


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    if inter <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def centroid_dist(a: BBox, b: BBox) -> float:
    ax = (a[0] + a[2]) / 2.0
    ay = (a[1] + a[3]) / 2.0
    bx = (b[0] + b[2]) / 2.0
    by = (b[1] + b[3]) / 2.0
    return math.hypot(ax - bx, ay - by)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


class IouCentroidTracker:
    """
    Lightweight short-term tracker.

    Track IDs are temporary implementation details. CUSTOMER_ID is stable and
    separate from these short-term track IDs.
    """

    def __init__(
        self,
        iou_threshold: float = 0.30,
        max_centroid_dist: float = 120.0,
        max_misses: int = 8,
    ):
        self.iou_threshold = iou_threshold
        self.max_centroid_dist = max_centroid_dist
        self.max_misses = max_misses

        self._next_id = itertools.count(1)
        self.tracks: List[Track] = []

    def update(self, detections: Sequence[Detection]) -> List[Track]:
        unmatched_tracks = list(self.tracks)
        unmatched_dets = list(detections)
        matches = []

        # Pass 1: IoU matching.
        pairs = []
        for t in unmatched_tracks:
            for d in unmatched_dets:
                score = iou(t.bbox, d.bbox)
                if score >= self.iou_threshold:
                    pairs.append((score, t, d))

        pairs.sort(key=lambda p: p[0], reverse=True)

        used_tracks = set()
        used_dets = set()

        for _, t, d in pairs:
            if id(t) in used_tracks or id(d) in used_dets:
                continue
            matches.append((t, d))
            used_tracks.add(id(t))
            used_dets.add(id(d))

        # Pass 2: centroid fallback.
        remaining_tracks = [
            t for t in unmatched_tracks
            if id(t) not in used_tracks
        ]
        remaining_dets = [
            d for d in unmatched_dets
            if id(d) not in used_dets
        ]

        pairs = []
        for t in remaining_tracks:
            for d in remaining_dets:
                dist = centroid_dist(t.bbox, d.bbox)
                if dist <= self.max_centroid_dist:
                    pairs.append((dist, t, d))

        pairs.sort(key=lambda p: p[0])

        for _, t, d in pairs:
            if id(t) in used_tracks or id(d) in used_dets:
                continue
            matches.append((t, d))
            used_tracks.add(id(t))
            used_dets.add(id(d))

        # Apply matches.
        matched_track_objects = set()

        for t, d in matches:
            t.bbox = d.bbox
            t.age += 1
            t.hits += 1
            t.misses = 0
            t.matched_this_frame = True
            matched_track_objects.add(id(t))

        # Coast unmatched tracks briefly.
        still_alive = []
        for t in self.tracks:
            if id(t) not in matched_track_objects:
                t.age += 1
                t.misses += 1
                t.matched_this_frame = False

            if t.misses <= self.max_misses:
                still_alive.append(t)

        self.tracks = still_alive

        # Create tracks for unmatched detections.
        matched_det_ids = {id(d) for _, d in matches}

        for d in detections:
            if id(d) not in matched_det_ids:
                self.tracks.append(
                    Track(
                        track_id=next(self._next_id),
                        bbox=d.bbox
                    )
                )

        return self.tracks


def _crop(frame: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(0, min(w, int(round(x2))))
    y2 = max(0, min(h, int(round(y2))))

    if x2 - x1 < 12 or y2 - y1 < 24:
        return None

    return frame[y1:y2, x1:x2]


def appearance_descriptor(
    frame: np.ndarray,
    bbox: BBox
) -> Optional[np.ndarray]:
    """
    Clothing/appearance descriptor for prototype reacquisition.

    Uses separate upper/lower-body HSV histograms plus coarse grayscale shape.
    This is intentionally independent of reid_proxy.onnx.
    """

    crop = _crop(frame, bbox)

    if crop is None or crop.size == 0:
        return None

    h, w = crop.shape[:2]

    # Ignore small borders/background around detector box.
    x_pad = int(w * 0.12)
    y_top = int(h * 0.08)
    y_bottom = int(h * 0.95)

    body = crop[
        y_top:y_bottom,
        x_pad:max(x_pad + 1, w - x_pad)
    ]

    if body.size == 0:
        return None

    bh = body.shape[0]
    upper = body[:max(1, int(bh * 0.55))]
    lower = body[int(bh * 0.45):]

    features = []

    for region in (upper, lower):
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # H/S histogram: clothing colour with less dependence on brightness.
        hist = cv2.calcHist(
            [hsv],
            [0, 1],
            None,
            [18, 8],
            [0, 180, 0, 256]
        )
        hist = cv2.normalize(
            hist,
            None,
            alpha=1.0,
            norm_type=cv2.NORM_L1
        ).flatten()

        features.append(hist.astype(np.float32))

    # Coarse grayscale appearance/shape.
    gray = cv2.cvtColor(body, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (8, 16), interpolation=cv2.INTER_AREA)
    gray = gray.astype(np.float32) / 255.0
    gray = gray.flatten()

    # Zero-center coarse gray so similarity is not dominated by brightness.
    gray -= float(np.mean(gray))
    n = float(np.linalg.norm(gray))
    if n > 1e-6:
        gray /= n

    features.append(gray.astype(np.float32))

    desc = np.concatenate(features).astype(np.float32)

    n = float(np.linalg.norm(desc))
    if n <= 1e-8:
        return None

    return desc / n


class Pipeline:
    """
    Person tracker + stable selected-customer identity.

    The user sees one stable CUSTOMER_ID. Underlying tracker IDs may change;
    conservative appearance reacquisition maps a new short-term track back to
    the same logical customer.
    """

    def __init__(
        self,
        embed_fn: Callable[[np.ndarray, BBox], np.ndarray],
        detect_fn: Optional[Callable[[np.ndarray], List[Detection]]] = None,
        match_threshold: float = 0.5,
        target_loss_grace_frames: int = 2,
    ):
        self.embed_fn = embed_fn
        self.detect_fn = detect_fn

        # Kept for command-line/API compatibility. Not used for identity
        # decisions while reid_proxy.onnx is untrained.
        self.match_threshold = match_threshold
        self.target_loss_grace_frames = target_loss_grace_frames

        self.tracker = IouCentroidTracker()

        # Compatibility flag used by dashboard:
        # non-None means "a customer has been enrolled".
        self.target_embedding: Optional[np.ndarray] = None

        # Underlying current short-term tracker ID.
        self._locked_track_id: Optional[int] = None

        # Stable appearance of enrolled customer.
        self._target_appearance: Optional[np.ndarray] = None

        self._frames_since_target_seen = 0
        self._frame_index = 0

        # Reacquisition confirmation.
        self._candidate_track_id: Optional[int] = None
        self._candidate_count = 0

    @property
    def target_track_id(self) -> Optional[int]:
        # Stable public/logical identity.
        return CUSTOMER_ID if self.target_embedding is not None else None

    def clear_target(self) -> None:
        self.target_embedding = None
        self._locked_track_id = None
        self._target_appearance = None
        self._frames_since_target_seen = 0
        self._candidate_track_id = None
        self._candidate_count = 0

        for t in self.tracker.tracks:
            t.is_target = False

    def enroll_target(self, frame: np.ndarray, bbox: BBox) -> None:
        """
        Enroll the exact selected current track.
        """

        # Existing proxy embedding is retained only as an enrollment flag.
        try:
            self.target_embedding = self.embed_fn(frame, bbox)
        except Exception:
            # Still create a valid enrollment flag if the proxy embedding fails.
            self.target_embedding = np.ones((1,), dtype=np.float32)

        self._target_appearance = appearance_descriptor(frame, bbox)

        best_track = None
        best_iou = -1.0

        for t in self.tracker.tracks:
            score = iou(t.bbox, bbox)
            if score > best_iou:
                best_iou = score
                best_track = t

        if best_track is None or best_iou <= 0.0:
            best_track = None
            best_dist = float("inf")

            for t in self.tracker.tracks:
                d = centroid_dist(t.bbox, bbox)
                if d < best_dist:
                    best_dist = d
                    best_track = t

            if best_track is not None and best_dist > 120.0:
                best_track = None

        self._locked_track_id = (
            best_track.track_id
            if best_track is not None
            else None
        )

        self._frames_since_target_seen = 0
        self._candidate_track_id = None
        self._candidate_count = 0

        for t in self.tracker.tracks:
            t.is_target = (
                best_track is not None
                and
                t.track_id == best_track.track_id
            )

    def process_frame_raw(self, frame: np.ndarray) -> TrackingResult:
        if self.detect_fn is None:
            raise RuntimeError(
                "detect_fn not supplied; use process_frame(detections, frame)"
            )

        detections = self.detect_fn(frame)
        return self.process_frame(detections, frame)

    def _find_locked_visible(
        self,
        tracks: Sequence[Track]
    ) -> Optional[Track]:
        if self._locked_track_id is None:
            return None

        for t in tracks:
            if (
                t.track_id == self._locked_track_id
                and
                t.matched_this_frame
            ):
                return t

        return None

    def _try_reacquire(
        self,
        tracks: Sequence[Track],
        frame: np.ndarray
    ) -> Optional[Track]:
        if self._target_appearance is None:
            return None

        scored = []

        for t in tracks:
            if not t.matched_this_frame:
                continue

            # Never use stale/coasting boxes.
            desc = appearance_descriptor(frame, t.bbox)
            if desc is None:
                continue

            sim = cosine_sim(
                desc,
                self._target_appearance
            )

            scored.append((sim, t, desc))

        if not scored:
            self._candidate_track_id = None
            self._candidate_count = 0
            return None

        scored.sort(key=lambda x: x[0], reverse=True)

        best_sim, best_track, best_desc = scored[0]
        second_sim = scored[1][0] if len(scored) > 1 else -1.0

        # Single-person case requires an even stronger similarity.
        if len(scored) == 1:
            acceptable = best_sim >= REACQUIRE_SINGLE_MIN_SIM
        else:
            acceptable = (
                best_sim >= REACQUIRE_MIN_SIM
                and
                (best_sim - second_sim) >= REACQUIRE_MARGIN
            )

        if not acceptable:
            self._candidate_track_id = None
            self._candidate_count = 0
            return None

        if self._candidate_track_id == best_track.track_id:
            self._candidate_count += 1
        else:
            self._candidate_track_id = best_track.track_id
            self._candidate_count = 1

        if self._candidate_count < REACQUIRE_CONFIRM_FRAMES:
            return None

        # Confirm reacquisition.
        self._locked_track_id = best_track.track_id
        self._candidate_track_id = None
        self._candidate_count = 0

        # Very slow appearance update after successful match.
        merged = (
            (1.0 - APPEARANCE_UPDATE_ALPHA) * self._target_appearance
            +
            APPEARANCE_UPDATE_ALPHA * best_desc
        )

        n = float(np.linalg.norm(merged))
        if n > 1e-8:
            self._target_appearance = (
                merged / n
            ).astype(np.float32)

        return best_track

    def process_frame(
        self,
        detections: Sequence[Detection],
        frame: np.ndarray
    ) -> TrackingResult:

        self._frame_index += 1
        tracks = self.tracker.update(detections)

        for t in tracks:
            t.is_target = False
            t.embedding = None

        # No customer selected.
        if self.target_embedding is None:
            return TrackingResult(
                tracks=list(tracks),
                target_track_id=None,
                target_bbox=None,
                target_lost=False,
                frame_index=self._frame_index,
            )

        target = self._find_locked_visible(tracks)

        if target is None:
            target = self._try_reacquire(
                tracks,
                frame
            )

        if target is not None:
            target.is_target = True
            self._frames_since_target_seen = 0

            current_desc = appearance_descriptor(
                frame,
                target.bbox
            )

            if (
                current_desc is not None
                and
                self._target_appearance is not None
            ):
                # Only drift slowly to avoid absorbing another person's look
                # during a brief tracker error.
                if cosine_sim(
                    current_desc,
                    self._target_appearance
                ) >= 0.78:
                    merged = (
                        (1.0 - APPEARANCE_UPDATE_ALPHA)
                        * self._target_appearance
                        +
                        APPEARANCE_UPDATE_ALPHA
                        * current_desc
                    )

                    n = float(np.linalg.norm(merged))
                    if n > 1e-8:
                        self._target_appearance = (
                            merged / n
                        ).astype(np.float32)

            return TrackingResult(
                tracks=list(tracks),
                target_track_id=CUSTOMER_ID,
                target_bbox=target.bbox,
                target_lost=False,
                frame_index=self._frame_index,
            )

        self._frames_since_target_seen += 1

        # Safety-first: no currently verified target means LOST to steering.
        return TrackingResult(
            tracks=list(tracks),
            target_track_id=CUSTOMER_ID,
            target_bbox=None,
            target_lost=True,
            frame_index=self._frame_index,
        )


def _self_test() -> None:
    print("Running tracker_reid.py stable-customer-ID self-test...")

    def fake_embed_fn(frame, bbox):
        return np.ones((8,), dtype=np.float32)

    pipeline = Pipeline(
        embed_fn=fake_embed_fn,
        target_loss_grace_frames=2
    )

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Give target and decoy visibly different colours for the appearance test.
    frame[100:320, 80:180] = (0, 0, 255)      # red target
    frame[100:320, 400:500] = (255, 0, 0)     # blue decoy

    b_target = (80, 100, 180, 320)
    b_decoy = (400, 100, 500, 320)

    r0 = pipeline.process_frame(
        [
            Detection(b_target, 0.95),
            Detection(b_decoy, 0.95),
        ],
        frame
    )

    selected = min(
        r0.tracks,
        key=lambda t: centroid_dist(
            t.bbox,
            b_target
        )
    )

    pipeline.enroll_target(
        frame,
        selected.bbox
    )

    r1 = pipeline.process_frame(
        [
            Detection((85, 100, 185, 320), 0.95),
            Detection((395, 100, 495, 320), 0.95),
        ],
        frame
    )

    assert r1.target_track_id == CUSTOMER_ID
    assert any(t.is_target for t in r1.tracks)

    print("  target remains logically CUSTOMER_ID=1")

    # Remove target.
    r2 = pipeline.process_frame(
        [Detection((395, 100, 495, 320), 0.95)],
        frame
    )

    assert r2.target_lost
    assert not any(t.is_target for t in r2.tracks)

    print("  absent customer is not replaced by another person")
    print("Self-test PASSED.")


if __name__ == "__main__":
    _self_test()
