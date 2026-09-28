#!/usr/bin/env python3
"""
Pi 5 CPU benchmark for the human-following pipeline (Option B) -- ONNX RUNTIME
VERSION. No PyTorch import anywhere in this script.

WHY THIS VERSION EXISTS
------------------------
Generic PyPI PyTorch wheels for aarch64 are frequently NOT well-tested against
Raspberry Pi's specific Broadcom SoC / ARM configuration, and can crash with
"Bus error" the instant `import torch` runs -- before any model even loads.
ONNX Runtime has much more reliable ARM64 support and is the standard choice
for edge/embedded inference for exactly this reason. This script needs only:

    pip install onnxruntime opencv-python-headless numpy

and two .onnx files (provided alongside this script):
    yolov8n.onnx    -- person detector (exported from Ultralytics YOLOv8n)
    reid_proxy.onnx -- compute-matched proxy for a re-id embedding network
                       (RANDOM WEIGHTS -- for TIMING ONLY, not real identity
                       matching; see note at the bottom of this file)

WHAT THIS MEASURES
-------------------
  1. Pure detector fps  (how fast can we find all people in a frame)
  2. Per-crop re-ID embedding latency (cost of ONE identity check)
  3. Combined fps if re-ID ran on every detected person, every frame
     (the "worst case" reference point -- Option A cost)

USAGE
-----
  python3 benchmark_pipeline_onnx.py --source 0 --duration 20
  python3 benchmark_pipeline_onnx.py --source /path/to/video.mp4 --duration 20
  python3 benchmark_pipeline_onnx.py --source synthetic --duration 15 --num-people 2
"""

import argparse
import time
import sys
import numpy as np
import cv2

def log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------
# YOLOv8n ONNX pre/post-processing (no ultralytics dependency needed at
# inference time -- just onnxruntime + opencv + numpy)
# --------------------------------------------------------------------------
def letterbox(img, new_shape=640):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
    top = (new_shape - nh) // 2
    left = (new_shape - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


def preprocess(img, imgsz=640):
    canvas, r, pad_x, pad_y = letterbox(img, imgsz)
    blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, 0)
    return np.ascontiguousarray(blob), r, pad_x, pad_y


def postprocess_person_boxes(output, r, pad_x, pad_y, orig_shape, conf_thres=0.4, iou_thres=0.45):
    # output: (1, 84, 8400) -> (8400, 84):  4 box coords + 80 class scores
    preds = output[0].transpose(1, 0)
    boxes_xywh = preds[:, :4]
    class_scores = preds[:, 4:]
    person_scores = class_scores[:, 0]  # COCO class 0 = person
    mask = person_scores > conf_thres
    boxes_xywh = boxes_xywh[mask]
    person_scores = person_scores[mask]
    if len(boxes_xywh) == 0:
        return np.empty((0, 4)), np.empty((0,))

    # xywh (center) -> xyxy, undo letterbox
    cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    x1 = (cx - w / 2 - pad_x) / r
    y1 = (cy - h / 2 - pad_y) / r
    x2 = (cx + w / 2 - pad_x) / r
    y2 = (cy + h / 2 - pad_y) / r
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_shape[1])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_shape[0])

    # NMS via OpenCV
    idxs = cv2.dnn.NMSBoxes(
        boxes.tolist(), person_scores.tolist(), conf_thres, iou_thres
    )
    if len(idxs) == 0:
        return np.empty((0, 4)), np.empty((0,))
    idxs = np.array(idxs).flatten()
    return boxes[idxs], person_scores[idxs]


# --------------------------------------------------------------------------
# Frame sources
# --------------------------------------------------------------------------
def frame_source_camera_or_file(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log(f"[error] Could not open video source: {source}")
        sys.exit(1)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
    cap.release()


def frame_source_synthetic(width=1280, height=720):
    rng = np.random.default_rng(0)
    while True:
        yield rng.integers(0, 255, (height, width, 3), dtype=np.uint8)


# --------------------------------------------------------------------------
# Main benchmark
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="Camera index, video file path, or 'synthetic'")
    ap.add_argument("--duration", type=float, default=20.0, help="Benchmark duration in seconds")
    ap.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    ap.add_argument("--num-people", type=int, default=2, help="Synthetic-mode: number of simulated people per frame")
    ap.add_argument("--crop-size", type=int, default=128, help="Re-ID input crop size (square)")
    ap.add_argument("--yolo-onnx", default="yolov8n.onnx", help="Path to YOLO ONNX model")
    ap.add_argument("--reid-onnx", default="reid_proxy.onnx", help="Path to re-id ONNX model")
    ap.add_argument("--threads", type=int, default=0, help="ONNX Runtime intra-op threads (0 = auto/all cores)")
    args = ap.parse_args()

    log("=" * 70)
    log("Human-following pipeline benchmark -- ONNX Runtime (no torch)")
    log("=" * 70)

    try:
        import onnxruntime as ort
    except ImportError:
        log("[error] onnxruntime not installed. Run: pip install onnxruntime")
        sys.exit(1)

    sess_opts = ort.SessionOptions()
    if args.threads > 0:
        sess_opts.intra_op_num_threads = args.threads

    log(f"[detector] Loading {args.yolo_onnx} ...")
    det_sess = ort.InferenceSession(args.yolo_onnx, sess_options=sess_opts, providers=["CPUExecutionProvider"])
    det_input_name = det_sess.get_inputs()[0].name

    log(f"[reid] Loading {args.reid_onnx} ...")
    reid_sess = ort.InferenceSession(args.reid_onnx, sess_options=sess_opts, providers=["CPUExecutionProvider"])
    reid_input_name = reid_sess.get_inputs()[0].name
    log("[reid] NOTE: reid_proxy.onnx has RANDOM weights -- timing only, not real identity matching.")

    # ---- Warmup ----
    log("[warmup] Running warmup iterations...")
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(3):
        blob, r, px, py = preprocess(dummy, args.imgsz)
        det_sess.run(None, {det_input_name: blob})
    dummy_crop = np.zeros((1, 3, args.crop_size, args.crop_size), dtype=np.float32)
    for _ in range(3):
        reid_sess.run(None, {reid_input_name: dummy_crop})

    # ---- Benchmark loop ----
    synthetic = (args.source == "synthetic")
    if synthetic:
        source_iter = frame_source_synthetic()
    else:
        src = int(args.source) if args.source.isdigit() else args.source
        source_iter = frame_source_camera_or_file(src)

    det_times, reid_times_per_crop, combined_frame_times = [], [], []
    start = time.time()
    last_report = start
    frame_count = 0

    log(f"[bench] Running for {args.duration:.0f}s... (Ctrl+C to stop early)")
    try:
        while True:
            now = time.time()
            if now - start >= args.duration:
                break

            frame = next(source_iter)
            frame_start = time.time()

            # --- Detection ---
            t0 = time.time()
            blob, r, px, py = preprocess(frame, args.imgsz)
            output = det_sess.run(None, {det_input_name: blob})[0]
            det_time = time.time() - t0
            det_times.append(det_time)

            if synthetic:
                h, w = frame.shape[:2]
                n = args.num_people
                boxes = np.array([
                    [w * i / (n + 1), h * 0.2, w * i / (n + 1) + 150, h * 0.8]
                    for i in range(1, n + 1)
                ])
            else:
                boxes, scores = postprocess_person_boxes(output, r, px, py, frame.shape)

            # --- Re-ID stage ---
            for (x1, y1, x2, y2) in boxes.astype(int):
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = cv2.resize(frame[y1:y2, x1:x2], (args.crop_size, args.crop_size))
                tensor = crop.transpose(2, 0, 1).astype(np.float32)[None] / 255.0
                t1 = time.time()
                reid_sess.run(None, {reid_input_name: tensor})
                rt = time.time() - t1
                reid_times_per_crop.append(rt)

            combined_frame_times.append(time.time() - frame_start)
            frame_count += 1

            if time.time() - last_report >= 2.0:
                rd = np.mean(det_times[-30:]) if det_times else 0
                rc = np.mean(combined_frame_times[-30:]) if combined_frame_times else 0
                log(f"  [{time.time()-start:5.1f}s] frames={frame_count:4d}  "
                    f"detector_fps={1/rd if rd else 0:5.1f}  combined_fps={1/rc if rc else 0:5.1f}  "
                    f"people_this_frame={len(boxes)}")
                last_report = time.time()

    except KeyboardInterrupt:
        log("\n[bench] Stopped early by user.")

    # ---- Summary ----
    log("=" * 70)
    log("RESULTS")
    log("=" * 70)
    if det_times:
        avg_det = np.mean(det_times)
        log(f"Detector only:        {1/avg_det:6.2f} fps   ({avg_det*1000:6.1f} ms/frame)")
    if reid_times_per_crop:
        avg_reid = np.mean(reid_times_per_crop)
        log(f"Re-ID per crop:       {avg_reid*1000:6.1f} ms/crop   (proxy net, timing only)")
    if combined_frame_times:
        avg_c = np.mean(combined_frame_times)
        log(f"Combined (worst case, re-id on ALL people EVERY frame):")
        log(f"                      {1/avg_c:6.2f} fps   ({avg_c*1000:6.1f} ms/frame)")
    log("-" * 70)
    if det_times and reid_times_per_crop:
        log("How to use this for Option B tuning:")
        log(f"  - Per-frame FLOOR (detector + tracker, no re-id): ~{avg_det*1000:.0f} ms/frame")
        log(f"  - Each re-id check costs ~{avg_reid*1000:.0f} ms extra, one-time, when it runs.")
        log(f"  - Example: re-id every 15 frames adds ~{avg_reid*1000/15:.1f} ms/frame average overhead.")
    log("=" * 70)
    log("Run this on the Pi 5 itself for numbers that matter.")


if __name__ == "__main__":
    main()
