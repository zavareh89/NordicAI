"""Run from challenge root: python -m pytest tests/test_endpoint_replay.py -q.
Tests use synthetic images/annotations and a local HTTP server, never a detector.
"""
import base64
import copy
import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import threading

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import replay_training_endpoint as replay


@pytest.fixture
def protocol():
    try:
        import dtos
    except ImportError:
        pytest.skip("Official dtos.py must be on PYTHONPATH or in challenge root")
    return dtos


def examples():
    frames = [{"frame": i, "width": 3840, "height": 2160,
               "annotations": [{"object_id": "hangar", "bbox": [100,100,200,200]}]} for i in (0,7)]
    preds = {f["frame"]: [{**a, "confidence": .9} for a in f["annotations"]] for f in frames}
    return frames, preds


def test_perfect_and_empty():
    frames, preds = examples()
    m,_ = replay.coco_metrics(frames,preds,["hangar","tank"])
    assert m["map50"] == pytest.approx(1)
    assert m["map50_95"] == pytest.approx(1)
    assert m["classes_without_ground_truth"] == ["tank"]
    empty,_ = replay.coco_metrics(frames,{},["hangar","tank"])
    assert empty["map50"] == 0
    assert replay.operating_metrics(frames,{},.05)["fn"] == 2


def test_missing_frame_retained():
    frames,preds = examples()
    preds.pop(7)
    m,_ = replay.coco_metrics(frames,preds,["hangar"])
    assert m["map50"] == pytest.approx(51/101)
    assert replay.operating_metrics(frames,preds,0)["recall"] == .5


def test_high_confidence_false_positive_and_duplicate():
    frames,preds = examples()
    preds[0].insert(0,{"object_id":"hangar","confidence":.99,"bbox":[500,500,600,600]})
    preds[0].append(copy.deepcopy(preds[0][1]))
    m,_ = replay.coco_metrics(frames,preds,["hangar"])
    assert 0 < m["map50"] < 1
    op = replay.operating_metrics(frames,preds,0)
    assert (op["tp"],op["fp"],op["fn"]) == (2,2,0)


def test_class_agnostic_diagnostic():
    frames,preds = examples()
    for annotations in preds.values():
        annotations[0]["object_id"] = "tank"
    m,_ = replay.coco_metrics(frames,preds,["hangar","tank"])
    assert m["map50"] == 0
    assert replay.operating_metrics(frames,preds,0)["recall"] == 0
    assert replay.operating_metrics(frames,preds,0,True)["recall"] == 1


def test_threshold_only_changes_operating_point():
    frames,preds = examples()
    assert replay.operating_metrics(frames,preds,.95)["recall"] == 0
    assert replay.coco_metrics(frames,preds,["hangar"])[0]["map50"] == pytest.approx(1)


@pytest.mark.parametrize("change", ["frame","request_id","pixel_box","nan","unknown_class","extra_key"])
def test_invalid_contract(protocol, change):
    payload = {"frame":7,"request_id":"req","original_width":3840,"original_height":2160}
    body = {"frame":7,"request_id":"req","annotations":[{"object_id":"hangar","bbox":[.1,.1,.2,.2],"confidence":.9}],"requested_view":None}
    if change == "frame": body["frame"] = 8
    if change == "request_id": body["request_id"] = "wrong"
    if change == "pixel_box": body["annotations"][0]["bbox"] = [100,100,200,200]
    if change == "nan": body["annotations"][0]["confidence"] = float("nan")
    if change == "unknown_class": body["annotations"][0]["object_id"] = "invalid"
    if change == "extra_key": body["unexpected"] = True
    with pytest.raises(ValueError): replay.validate_predictions(body,payload,protocol)


def test_matches_official_ap50(protocol,monkeypatch):
    try:
        import local_evaluator as official
    except ImportError:
        pytest.skip("Official local_evaluator.py not on PYTHONPATH")
    frames,preds = examples()
    preds[0].insert(0,{"object_id":"hangar","confidence":.99,"bbox":[500,500,600,600]})
    preds.pop(7)
    monkeypatch.setattr(official,"frame_numbers",lambda scene:[f["frame"] for f in frames])
    monkeypatch.setattr(official,"load_annotations",lambda frame,scene: next(f["annotations"] for f in frames if f["frame"]==frame))
    actual,per_class = official.score("synthetic",preds)
    ours,_ = replay.coco_metrics(frames,preds,protocol.OBJECT_CLASSES)
    assert ours["map50"] == pytest.approx(actual,abs=1e-12)
    assert ours["per_class"]["hangar"]["ap50"] == pytest.approx(per_class["hangar"],abs=1e-12)


@pytest.mark.parametrize("mode",["perfect","failure"])
def test_real_http_replay(tmp_path,protocol,mode):
    images = tmp_path/"images"
    images.mkdir()
    frames = []
    for fid in [0,7]:
        path = images/f"frame_{fid:06d}.png"
        cv2.imwrite(str(path),np.zeros((540,960,3),dtype=np.uint8))
        frames.append({"frame_id":fid,"level0_image":f"/old/machine/{path.name}","source_width":3840,"source_height":2160,
                       "target_width":960,"target_height":540,"annotations":[{"frame_id":fid,"class_name":"hangar","bbox_xyxy":[25,25,50,50]}]})
    manifest,split = tmp_path/"manifest.json",tmp_path/"split.json"
    replay.write_json(manifest,{"frames":frames})
    replay.write_json(split,{"train_frame_ids":[0,7],"val_frame_ids":[]})
    received = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            protocol.DroneFlybyPredictRequestDto.model_validate(payload)
            assert base64.b64decode(payload["view"]["image"]) == (images/f"frame_{payload['frame']:06d}.png").read_bytes()
            received.append(payload["frame"])
            if mode == "failure" and payload["frame"] == 7:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"simulated error")
                return
            body={"frame":payload["frame"],"request_id":payload["request_id"],"requested_view":None,
                  "annotations":[{"object_id":"hangar","confidence":.9,"bbox":[100/3840,100/2160,200/3840,200/2160]}]}
            data=json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    out=tmp_path/"results"
    try:
        result=subprocess.run([sys.executable,str(ROOT/"scripts/replay_training_endpoint.py"),
            "--url",f"http://127.0.0.1:{server.server_port}/predict","--challenge-root",str(Path(protocol.__file__).parent),
            "--manifest",str(manifest),"--split",str(split),"--images-dir",str(images),"--output",str(out),"--overlays"],
            capture_output=True,text=True,timeout=30)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert result.returncode == (0 if mode == "perfect" else 2),result.stdout+result.stderr
    assert received == [0,7]
    metrics=replay.read_json(out/"metrics.json")
    assert metrics["coco"]["map50"] == pytest.approx(1 if mode == "perfect" else 51/101)
    assert metrics["failed_frames"] == ([] if mode == "perfect" else [7])
    assert len(list((out/"overlays").glob("*.png"))) == 2
    assert (out/"per_class.csv").exists()
    assert metrics["operating_point"]["fn"] == (0 if mode == "perfect" else 1)
