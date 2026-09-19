from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import orjson
from fastapi import FastAPI, Request
from fastapi.responses import ORJSONResponse

from survival_policy import PolicyConfig, SurvivalController, load_default_config

LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.WARNING))
logger = logging.getLogger("survival-agent")


def _load_config() -> PolicyConfig:
    config_path = os.getenv("SURVIVAL_POLICY_CONFIG")
    if config_path:
        return PolicyConfig.from_json(Path(config_path))
    return load_default_config()


CONFIG = _load_config()
CONTROLLER = SurvivalController(CONFIG)

# Keep the serving app minimal during competition.
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


def _noop_actions(payload: Any) -> list[dict[str, Any]]:
    """Return one safe no-op action for every visible herbivore."""
    if not isinstance(payload, dict):
        return []

    actions: list[dict[str, Any]] = []
    for status in payload.get("agent_status") or ():
        if not isinstance(status, dict) or "agent_id" not in status:
            continue
        actions.append(
            {
                "agent_id": int(status["agent_id"]),
                "move_distance": 0.0,
                "move_direction": 0.0,
                "turn_angle": 0.0,
                "spawn_agent": False,
            }
        )
    return actions


def _prediction_response(actions: list[dict[str, Any]]) -> ORJSONResponse:
    """Return the exact competition response contract.

    The challenge endpoint expects a JSON object with an ``actions`` field,
    not a bare list. ORJSONResponse serializes the Python object directly and
    avoids returning a Python bytes representation such as ``b'[...]'``.
    """
    return ORJSONResponse(content={"actions": actions})


@app.get("/")
async def health() -> ORJSONResponse:
    return ORJSONResponse(
        content={
            "status": "ok",
            "config_hash": CONFIG.stable_hash,
        }
    )


@app.post("/predict")
async def predict(request: Request) -> ORJSONResponse:
    """Low-overhead competition endpoint with the official response wrapper."""
    payload: Any = None

    try:
        # Fast path: raw body -> orjson -> existing controller.
        payload = orjson.loads(await request.body())
        actions = CONTROLLER.decide_step(payload)

        # IMPORTANT: official contract is {"actions": [...]}.
        return _prediction_response(actions)

    except Exception as exc:
        logger.error("predict failed: %s: %s", type(exc).__name__, exc)
        return _prediction_response(_noop_actions(payload))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "agent_server_fast_contract_fixed:app",
        host=os.getenv("SURVIVAL_AGENT_HOST", "0.0.0.0"),
        port=int(os.getenv("SURVIVAL_AGENT_PORT", "9052")),
        workers=1,  # REQUIRED: controller memory is stateful across ticks.
        loop="uvloop",
        http="httptools",
        access_log=False,
        log_level=os.getenv("LOG_LEVEL", "warning").lower(),
    )
