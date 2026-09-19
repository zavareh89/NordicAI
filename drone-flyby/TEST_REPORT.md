# Validation report for this delivery

Date: 2026-09-19

## Executed

Only the lightweight test suite was executed, following the explicit delivery constraint not to run training, downloads, servers, or evaluator jobs in this environment.

Command:

```bash
pytest -q
```

Latest result before packaging:

```text
19 passed, 1 skipped
```

The skipped test is `tests/test_official_dto_optional.py`; this standalone artifact intentionally does not duplicate the official challenge `dtos.py`. Once the project files are overlaid into the official `drone-flyby/` root, that test detects and validates the real DTO module.

The executed tests use no detector weight downloads and no real model training/inference. They cover annotation parsing, exact `INTER_AREA` Level-0 resizing, generic box scaling, orphan/missing data validation, blocked temporal splitting, YOLO/COCO export geometry, detector-adapter output contracts using stubs, confidence/top-K/NMS-independent postprocessing, Level-0/global coordinate conversion, response `requested_view=None`, configuration invariants, module imports, RF-DETR resolution constraints, Ultralytics scale-argument conversion, and checkpoint-candidate discovery.

## Deliberately not executed here

- dependency installation;
- pretrained-weight downloads;
- YOLO26-S or RF-DETR-Small real checkpoint loading;
- real detector inference;
- smoke training;
- API server startup / HTTP request;
- official local-evaluator replay;
- GPU latency benchmarking.

Those are provided as reproducible commands in `README.md` but were not run because the delivery request explicitly restricted execution to tests. No success claim is made for those unexecuted operations.
