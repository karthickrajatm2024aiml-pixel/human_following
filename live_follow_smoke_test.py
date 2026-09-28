#!/usr/bin/env python3
"""
live_follow_smoke_test.py

Wires tracker_reid.Pipeline to the REAL yolov8n.onnx (imgsz=416, dynamic
shape) and reid_proxy.onnx models, reusing preprocess()/postprocess_person_
boxes() imported directly from benchmark_pipeline_onnx.py -- so detection
here uses the exact same code path that was benchmarked (no drift between
"what we measured" and "what actually runs").

This is a SMOKE TEST, not a benchmark: it prints per-frame tracker/target
state so you can visually confirm detection + tracking + re-id target-lock
behave sanely before wiring this into steering. Run it, stand in frame,
then walk around / turn partially / step behind something briefly and watch
whether target_id stays sane and target_lost behaves as expected.

Run on the Pi, in the same directory as yolov8n.onnx, reid_proxy.onnx,
benchmark_pipeline_onnx.py, and tracker_reid.py:

    python3 live_follow_smoke_test.py --source 0 --frames 200

NOTE: reid_proxy.onnx has RANDOM weights. The tracker/target-lock PLUMBING
is real and worth watching, but if the target track ID flips to the wrong
person, that is expected with the proxy net -- it is not evidence of a bug
in this integration. Re-run this same script once real re-id weights are
in place to judge actual identity-matching accuracy.
"""

import argparse
import sys
import time

import cv2
import numpy as np

from benchmark_pipeline_onnx import preprocess, postprocess_person_boxes
from tracker_reid import Pipeline, Detection

IMGSZ = 416            # locked-in decision (see project handoff)
REID_CROP_SIZE = 128   # matches benchmark_pipeline_onnx.py's default --crop-size


def build_detect_fn(det_sess, det_input_name, conf_thres=0.4, iou_thres=0.45):
    def detect_fn(frame):
        blob, r, px, py = preprocess(frame, IMGSZ)
        output = det_sess.run(None, {det_input_name: blob})[0]
        boxes, scores = postprocess_person_boxes(
            output, r, px, py, frame.shape, conf_thres=conf_thres, iou_thres=iou_thres
        )
        return [Detection(tuple(float(v) for v in b), float(s)) for b, s in zip(boxes, scores)]
    return detect_fn


def build_embed_fn(reid_sess, reid_input_name, crop_size=REID_CROP_SIZE):
    def embed_fn(frame, bbox):
        x1, y1, x2, y2 = (int(v) for v in bbox)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
        if x2 <= x1 or y2 <= y1:
            # Degenerate box (can happen right at frame edges). Return an
            # all-zero embedding rather than crashing -- cosine_sim against
            # zero is 0.0, so it just fails to match anything, which is the
            # safe behavior here rather than raising mid-loop.
            return np.zeros((512,), dtype=np.float32)
        crop = cv2.resize(frame[y1:y2, x1:x2], (crop_size, crop_size))
        tensor = crop.transpose(2, 0, 1).astype(np.float32)[None] / 255.0
        out = reid_sess.run(None, {reid_input_name: tensor})[0]
        return out.flatten()
    return embed_fn


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="Camera index or video file path")
    ap.add_argument("--frames", type=int, default=200, help="Number of frames to run")
    ap.add_argument("--yolo-onnx", default="yolov8n.onnx")
    ap.add_argument("--reid-onnx", default="reid_proxy.onnx")
    ap.add_argument("--match-threshold", type=float, default=0.5,
                     help="Cosine sim threshold to accept a track as target (meaningless with the proxy net -- tune once real weights are in)")
    args = ap.parse_args()

    try:
        import onnxruntime as ort
    except ImportError:
        print("[error] onnxruntime not installed. Run: pip install onnxruntime")
        sys.exit(1)

    print(f"[load] {args.yolo_onnx}")
    det_sess = ort.InferenceSession(args.yolo_onnx, providers=["CPUExecutionProvider"])
    det_input_name = det_sess.get_inputs()[0].name

    print(f"[load] {args.reid_onnx} (NOTE: random weights, timing/plumbing only)")
    reid_sess = ort.InferenceSession(args.reid_onnx, providers=["CPUExecutionProvider"])
    reid_input_name = reid_sess.get_inputs()[0].name

    detect_fn = build_detect_fn(det_sess, det_input_name)
    embed_fn = build_embed_fn(reid_sess, reid_input_name)
    pipeline = Pipeline(embed_fn=embed_fn, detect_fn=detect_fn, match_threshold=args.match_threshold)

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[error] could not open source: {args.source}")
        sys.exit(1)

    ok, frame = cap.read()
    if not ok:
        print("[error] could not read first frame")
        sys.exit(1)

    first_dets = detect_fn(frame)
    if not first_dets:
        print("[error] no person detected in first frame -- stand in view of the camera and re-run")
        sys.exit(1)

    target_det = max(first_dets, key=lambda d: d.confidence)
    pipeline.enroll_target(frame, target_det.bbox)
    print(f"[enroll] target locked at bbox={tuple(round(v, 1) for v in target_det.bbox)} "
          f"conf={target_det.confidence:.2f}")
    print("[run] stand still a moment, then walk around / turn / step behind something briefly...")

    for i in range(args.frames):
        ok, frame = cap.read()
        if not ok:
            print("[warn] frame read failed, stopping")
            break
        t0 = time.time()
        result = pipeline.process_frame_raw(frame)
        dt_ms = (time.time() - t0) * 1000.0
        tb = tuple(round(v, 0) for v in result.target_bbox) if result.target_bbox else None
        print(f"frame={i:4d}  dt={dt_ms:6.1f}ms  n_tracks={len(result.tracks):2d}  "
              f"target_id={result.target_track_id}  target_bbox={tb}  target_lost={result.target_lost}")

    cap.release()
    print("[done]")


if __name__ == "__main__":
    main()
