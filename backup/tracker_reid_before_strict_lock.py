"""
tracker_reid.py

IOU/centroid multi-person tracker with per-frame re-ID target matching, for the
human-following robot pipeline. Drop-in module: sits between the YOLOv8n
detector (imgsz=416, dynamic-shape ONNX export) and the Pi-owned steering
logic.

Design notes (per project decisions to date):
  - Detector: yolov8n.onnx, dynamic input shape, run at imgsz=416.
  - Re-ID cost (~0.7ms/crop) is negligible next to the detector floor
    (~91ms @ 416), so re-ID runs on EVERY tracked person EVERY frame rather
    than "new track only + periodic drift-check". This replaces the interval
    scheduling in the original Option B design.
  - reid_proxy.onnx has random weights -> embeddings are for TIMING/PLUMBING
    validation only. Do not trust match decisions from it. Swap in a real
    re-ID model later via the `embed_fn` injection point below; nothing else
    in this file needs to change.
  - Re-ID only runs on tracks matched to a real detection in the CURRENT
    frame, never on "coasting" tracks that missed a detection this frame --
    a coasting track's bbox is stale, so cropping it would sample whatever
    is now at that old location rather than the track's original occupant.
    An earlier version of this file got this wrong (self-test caught it:
    a stale track and a legitimately-new track could both look like plausible
    target matches after an ID switch); fixed via `Track.matched_this_frame`.
  - This module owns tracking + target identity. It does NOT own steering
    (attraction/repulsion vectors from ultrasonics) or ESP32 comms -- those
    stay Pi-side / downstream, per the standing execution-only ESP32
    constraint. This module's job ends at producing a clean per-frame
    TrackingResult for the steering layer to consume.

Integration point:
  Call `Pipeline.process_frame(boxes, frame)` once per frame, where `boxes`
  is the list of Detection objects from the existing YOLO postprocessing in
  benchmark_pipeline_onnx.py (class-filtered to "person" already). This file
  does not re-implement YOLO decoding/NMS since that already exists and is
  validated on the Pi; it starts from detector output.

  If you want this file to own detector inference too, wire a real
  `detect_fn(frame) -> List[Detection]` and call `Pipeline.process_frame_raw
  (frame)` instead -- stub included below, disabled until a detect_fn is
  supplied, so this stays a pure drop-in either way.

Self-test:
  Run this file directly (`python3 tracker_reid.py`) to exercise the tracker
  and re-ID target-recovery logic against synthetic detections/embeddings --
  no ONNX Runtime or model files required for this. It validates the
  ALGORITHM (assignment, ID continuity, target recovery after an ID switch,
  target-loss signaling), not the ONNX inference path itself. The ONNX
  plumbing (session load, crop preprocessing, actual embed_fn/detect_fn) still
  needs a real smoke test on the Pi against the actual .onnx files, since
  those aren't available in this sandbox.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Data types
# --------------------------------------------------------------------------

BBox = Tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass
class Detection:
    bbox: BBox
    confidence: float


@dataclass
class Track:
    track_id: int
    bbox: BBox
    embedding: Optional[np.ndarray] = None
    age: int = 0             # frames since creation
    misses: int = 0          # consecutive frames without a matched detection
    hits: int = 1            # total frames matched
    is_target: bool = False
    matched_this_frame: bool = True  # False for "coasting" tracks with no detection this frame

    @property
    def centroid(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class TrackingResult:
    tracks: List[Track]
    target_track_id: Optional[int]
    target_bbox: Optional[BBox]
    target_lost: bool
    frame_index: int


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
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
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# --------------------------------------------------------------------------
# Tracker: IOU-primary with centroid-distance fallback (greedy assignment)
# --------------------------------------------------------------------------

class IouCentroidTracker:
    """
    Greedy frame-to-frame association. Tries IOU first (best for stable,
    non-occluded boxes); falls back to centroid distance for fast motion /
    partial overlap where IOU degrades. Simpler than Hungarian matching and
    fast enough given track counts here are small (a handful of people).
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
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
        matches: List[Tuple[Track, Detection]] = []

        # Pass 1: IOU-based greedy matching
        pairs = []
        for t in unmatched_tracks:
            for d in unmatched_dets:
                score = iou(t.bbox, d.bbox)
                if score >= self.iou_threshold:
                    pairs.append((score, t, d))
        pairs.sort(key=lambda p: p[0], reverse=True)
        used_tracks, used_dets = set(), set()
        for score, t, d in pairs:
            if id(t) in used_tracks or id(d) in used_dets:
                continue
            matches.append((t, d))
            used_tracks.add(id(t))
            used_dets.add(id(d))

        remaining_tracks = [t for t in unmatched_tracks if id(t) not in used_tracks]
        remaining_dets = [d for d in unmatched_dets if id(d) not in used_dets]

        # Pass 2: centroid-distance greedy matching on leftovers
        pairs = []
        for t in remaining_tracks:
            for d in remaining_dets:
                dist = centroid_dist(t.bbox, d.bbox)
                if dist <= self.max_centroid_dist:
                    pairs.append((dist, t, d))
        pairs.sort(key=lambda p: p[0])
        for dist, t, d in pairs:
            if id(t) in used_tracks or id(d) in used_dets:
                continue
            matches.append((t, d))
            used_tracks.add(id(t))
            used_dets.add(id(d))

        # Apply matches
        matched_track_ids = set()
        for t, d in matches:
            t.bbox = d.bbox
            t.age += 1
            t.hits += 1
            t.misses = 0
            t.matched_this_frame = True
            matched_track_ids.add(id(t))

        # Age out unmatched tracks. NOTE: these tracks are kept alive (in case
        # the person reappears in ~the same spot next frame) but their bbox is
        # now stale -- do not run re-id against a stale bbox, since it will
        # crop whatever is actually at that location in the current frame,
        # not the track's original occupant. Pipeline.process_frame relies on
        # matched_this_frame to skip these.
        still_alive = []
        for t in self.tracks:
            if id(t) not in matched_track_ids:
                t.age += 1
                t.misses += 1
                t.matched_this_frame = False
            if t.misses <= self.max_misses:
                still_alive.append(t)
        self.tracks = still_alive

        # New tracks for leftover detections
        for t, d in matches:
            pass  # already applied above
        matched_det_ids = {id(d) for _, d in matches}
        for d in detections:
            if id(d) not in matched_det_ids:
                new_track = Track(track_id=next(self._next_id), bbox=d.bbox)
                self.tracks.append(new_track)

        return self.tracks


# --------------------------------------------------------------------------
# Pipeline: tracker + per-frame re-ID + target identity management
# --------------------------------------------------------------------------

class Pipeline:
    """
    embed_fn(frame, bbox) -> np.ndarray
        Runs re-ID on one crop. Swap the proxy net for real weights here --
        no other code in this file changes.
    detect_fn(frame) -> List[Detection]
        Optional. Only needed if you want this module to own detection too
        (process_frame_raw). Otherwise call process_frame(detections, frame).
    match_threshold:
        Minimum cosine similarity to accept a track as the target. With the
        proxy net this number is meaningless (random weights) -- tune this
        once real re-ID weights are in place.
    """

    def __init__(
        self,
        embed_fn: Callable[[np.ndarray, BBox], np.ndarray],
        detect_fn: Optional[Callable[[np.ndarray], List[Detection]]] = None,
        match_threshold: float = 0.5,
        target_loss_grace_frames: int = 5,
    ):
        self.embed_fn = embed_fn
        self.detect_fn = detect_fn
        self.match_threshold = match_threshold
        self.target_loss_grace_frames = target_loss_grace_frames

        self.tracker = IouCentroidTracker()
        self.target_embedding: Optional[np.ndarray] = None
        self.target_track_id: Optional[int] = None
        self._frames_since_target_seen = 0
        self._frame_index = 0

    def enroll_target(self, frame: np.ndarray, bbox: BBox) -> None:
        """Call once (enrollment UI flow) to lock the target identity."""
        self.target_embedding = self.embed_fn(frame, bbox)
        self._frames_since_target_seen = 0

    def process_frame_raw(self, frame: np.ndarray) -> TrackingResult:
        if self.detect_fn is None:
            raise RuntimeError("detect_fn not supplied; use process_frame(detections, frame) instead")
        detections = self.detect_fn(frame)
        return self.process_frame(detections, frame)

    def process_frame(self, detections: Sequence[Detection], frame: np.ndarray) -> TrackingResult:
        self._frame_index += 1
        tracks = self.tracker.update(detections)

        # Re-ID on every tracked person, every frame (adopted simplification --
        # negligible cost next to detector floor; see module docstring). Only
        # tracks matched to a real detection THIS frame get re-embedded and
        # are eligible for target matching -- a coasting/unmatched track's
        # bbox is stale, so cropping it would sample the wrong image region.
        for t in tracks:
            t.is_target = False
            if t.matched_this_frame:
                t.embedding = self.embed_fn(frame, t.bbox)
            else:
                t.embedding = None

        target_track_id: Optional[int] = None
        target_bbox: Optional[BBox] = None

        if self.target_embedding is not None and tracks:
            best_track, best_score = None, -1.0
            for t in tracks:
                if t.embedding is None:
                    continue
                score = cosine_sim(t.embedding, self.target_embedding)
                if score > best_score:
                    best_track, best_score = t, score
            if best_track is not None and best_score >= self.match_threshold:
                best_track.is_target = True
                target_track_id = best_track.track_id
                target_bbox = best_track.bbox
                self._frames_since_target_seen = 0
                # Slowly drift the stored embedding toward the current match so
                # re-id tolerates gradual appearance change (lighting, angle).
                # Left as a simple EMA; tune alpha once real weights are in.
                alpha = 0.1
                self.target_embedding = (
                    (1 - alpha) * self.target_embedding + alpha * best_track.embedding
                )

        target_lost = False
        if self.target_embedding is not None and target_track_id is None:
            self._frames_since_target_seen += 1
            target_lost = self._frames_since_target_seen >= self.target_loss_grace_frames

        return TrackingResult(
            tracks=tracks,
            target_track_id=target_track_id,
            target_bbox=target_bbox,
            target_lost=target_lost,
            frame_index=self._frame_index,
        )


# --------------------------------------------------------------------------
# Self-test (synthetic, no ONNX Runtime / model files required)
# --------------------------------------------------------------------------

def _self_test() -> None:
    print("Running tracker_reid.py self-test (synthetic detections/embeddings)...")

    rng = np.random.default_rng(0)

    # Fixed per-"person" base embeddings, simulating a real re-ID model's
    # output: same person -> similar vector + small noise each frame.
    person_bases = {
        "target": rng.normal(size=32),
        "decoy_a": rng.normal(size=32),
        "decoy_b": rng.normal(size=32),
    }

    def fake_embed_fn(frame, bbox):
        # In this synthetic test we stash which "person" a bbox belongs to
        # in a closure-captured dict (see frame_labels below) rather than
        # deriving it from bbox, since bbox alone doesn't carry identity.
        label = frame_labels[bbox]
        base = person_bases[label]
        noisy = base + rng.normal(scale=0.05, size=32)
        return noisy

    pipeline = Pipeline(embed_fn=fake_embed_fn, match_threshold=0.9)

    # Frame 0: three people, enroll "target"
    frame_labels = {}
    target_box_f0 = (100, 100, 160, 260)
    decoy_a_box_f0 = (300, 100, 360, 260)
    decoy_b_box_f0 = (500, 100, 560, 260)
    frame_labels[target_box_f0] = "target"
    frame_labels[decoy_a_box_f0] = "decoy_a"
    frame_labels[decoy_b_box_f0] = "decoy_b"

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    pipeline.enroll_target(dummy_frame, target_box_f0)

    dets_f0 = [Detection(target_box_f0, 0.9), Detection(decoy_a_box_f0, 0.9), Detection(decoy_b_box_f0, 0.9)]
    result0 = pipeline.process_frame(dets_f0, dummy_frame)
    assert result0.target_track_id is not None, "target should be matched on frame 0"
    target_id_f0 = result0.target_track_id
    print(f"  frame 0: 3 tracks created, target locked as track_id={target_id_f0}")

    # Frame 1: everyone moves slightly, tracker should keep same IDs
    target_box_f1 = (105, 100, 165, 260)
    decoy_a_box_f1 = (305, 100, 365, 260)
    decoy_b_box_f1 = (505, 100, 565, 260)
    frame_labels[target_box_f1] = "target"
    frame_labels[decoy_a_box_f1] = "decoy_a"
    frame_labels[decoy_b_box_f1] = "decoy_b"
    dets_f1 = [Detection(target_box_f1, 0.9), Detection(decoy_a_box_f1, 0.9), Detection(decoy_b_box_f1, 0.9)]
    result1 = pipeline.process_frame(dets_f1, dummy_frame)
    assert result1.target_track_id == target_id_f0, "IOU tracking should preserve target track_id across small motion"
    print(f"  frame 1: target track_id preserved ({result1.target_track_id}) across motion")

    # Frame 2: simulate occlusion causing a hard track-ID switch -- target
    # reappears in a place that does NOT overlap its previous box (as if a
    # brief occlusion caused the tracker to drop and recreate the track), at
    # the SAME time a lookalike decoy is also present. Re-ID (not IOU) must
    # be what recovers the correct target identity.
    target_box_f2 = (250, 100, 310, 260)  # jumped, no IOU continuity
    decoy_a_box_f2 = (305, 100, 365, 260)  # decoy stayed put
    frame_labels[target_box_f2] = "target"
    frame_labels[decoy_a_box_f2] = "decoy_a"
    dets_f2 = [Detection(target_box_f2, 0.9), Detection(decoy_a_box_f2, 0.9)]
    result2 = pipeline.process_frame(dets_f2, dummy_frame)
    assert result2.target_track_id is not None, "target should be re-identified via embedding, not just IOU"
    assert result2.target_bbox == target_box_f2, "recovered target should map to the correct (jumped) bbox"
    new_target_id = result2.target_track_id
    assert new_target_id != target_id_f0, "this frame's jump should have produced a NEW track id (sanity check on test setup)"
    print(f"  frame 2: simulated occlusion/ID-switch -> re-id correctly recovered target as NEW track_id={new_target_id}")

    # Frame 3..7: target genuinely leaves the frame -> target_lost should
    # trigger after target_loss_grace_frames consecutive misses.
    lost_flagged_frame = None
    for i in range(3, 3 + pipeline.target_loss_grace_frames + 1):
        decoy_only_box = (305 + i, 100, 365 + i, 260)
        frame_labels[decoy_only_box] = "decoy_a"
        dets = [Detection(decoy_only_box, 0.9)]
        result = pipeline.process_frame(dets, dummy_frame)
        if result.target_lost and lost_flagged_frame is None:
            lost_flagged_frame = i
    assert lost_flagged_frame is not None, "target_lost should trigger after grace period with no target match"
    print(f"  frames 3-{3 + pipeline.target_loss_grace_frames}: target absent -> target_lost correctly triggered at frame index {lost_flagged_frame}")

    print("Self-test PASSED: assignment, ID continuity, re-id-based recovery after track-ID switch, "
          "and target-loss signaling all behave correctly.")


if __name__ == "__main__":
    _self_test()