from __future__ import annotations

import logging
import os
from pathlib import Path
from threading import Lock

from fastapi import Body, FastAPI

from src.utils.DTOs import StepResponse
from survival_policy import PolicyConfig, SurvivalController, load_default_config

HOST = os.getenv("SURVIVAL_AGENT_HOST", "0.0.0.0")
PORT = int(os.getenv("SURVIVAL_AGENT_PORT", "9052"))

logger = logging.getLogger("survival_agent")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


def _load_config() -> PolicyConfig:
    override = os.getenv("SURVIVAL_POLICY_CONFIG")
    if override:
        cfg = PolicyConfig.from_json(Path(override))
        logger.info("Loaded policy config %s hash=%s", override, cfg.stable_hash)
        return cfg
    cfg = load_default_config()
    logger.info("Loaded default policy config hash=%s", cfg.stable_hash)
    return cfg


CONFIG = _load_config()
CONTROLLER = SurvivalController(CONFIG)
_CONTROLLER_LOCK = Lock()

app = FastAPI(title="Survival Simulator Agent Endpoint")


@app.post("/predict")
def predict(step: StepResponse = Body(...)):
    """Return one valid action for every currently alive herbivore."""
    with _CONTROLLER_LOCK:
        try:
            actions = CONTROLLER.decide_step(step)
        except Exception:
            logger.exception("Controller failure; returning conservative no-op actions")
            actions = [
                {
                    "agent_id": agent.agent_id,
                    "move_distance": 0.0,
                    "move_direction": 0.0,
                    "turn_angle": 0.0,
                    "spawn_agent": False,
                }
                for agent in step.agent_status
            ]
    return {"actions": actions}


@app.get("/")
def index():
    return {
        "message": "Agent endpoint running!",
        "policy_config_hash": CONFIG.stable_hash,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
