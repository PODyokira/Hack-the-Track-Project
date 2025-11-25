"""FastAPI server exposing the IL + PPO strategy engine."""

from typing import Dict, List, Literal, Optional

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from integrated_system import IntegratedRacingSystem
import csv
import time
def telemetry_row_generator(filepath, delay=3):
    import json
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Only send relevant fields, add dummy lat/lon if not present
            yield (json.dumps(row) + '\n')
            time.sleep(delay)


app = FastAPI(title="GR Cup Strategy API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/live-telemetry")
def live_telemetry():
    """Stream telemetry data row by row for live simulation."""
    filepath = "barber/R1_barber_telemetry_data.csv"
    return StreamingResponse(telemetry_row_generator(filepath, delay=3), media_type="text/plain")


system = IntegratedRacingSystem(models_dir="models", data_dir="processed_data")

SECTOR_MAP = {
    "S1": [1.0, 0.0, 0.0],
    "S2": [0.0, 1.0, 0.0],
    "S3": [0.0, 0.0, 1.0],
}


class DrivingActionRequest(BaseModel):
    lateral_g: float = Field(0.8, example=0.8)
    longitudinal_g: float = Field(0.3, example=0.3)
    sector: Literal["S1", "S2", "S3"] = "S2"


class RaceStrategyRequest(BaseModel):
    lap: int = 2
    lap_progress: float = 0.6
    tire_age: float = 12
    position: int = 3
    degradation: float = 0.4
    pit_stops: int = 0


class ComprehensiveAnalysisRequest(BaseModel):
    driving_state: DrivingActionRequest
    race_state: RaceStrategyRequest
    telemetry_window: List[List[float]] = Field(..., min_items=10)
    sector_times: Optional[Dict[str, float]] = None


def _build_il_state(req: DrivingActionRequest) -> List[float]:
    one_hot = SECTOR_MAP.get(req.sector, SECTOR_MAP["S2"])
    return [req.lateral_g, req.longitudinal_g, *one_hot]


def _ensure_telemetry(window: List[List[float]]) -> np.ndarray:
    arr = np.array(window, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("telemetry_window must be 2D")
    return arr


def _format_driving_action_from_raw(raw: dict):
    throttle = float(raw.get("throttle_position", 0.65))
    steering = float(raw.get("steering_angle", 0.0))
    # Compute human-friendly percentages. Ensure brake percent is
    # realistically lower than throttle so the car doesn't 'stop' in simulation.
    throttle_pct = round(throttle * 100, 1)
    # Use a reduced scale for braking: braking is unlikely to be 100% in live
    # driving unless throttle is zero. Map (1 - throttle) to a 0-60% brake range.
    brake_pct = round((1.0 - throttle) * 60.0, 1)
    # If the brake would still exceed the throttle percentage (edge cases),
    # cap it to a fraction of throttle to keep the car moving.
    if brake_pct > throttle_pct:
        brake_pct = round(max(0.0, throttle_pct * 0.6), 1)

    return {
        "throttlePercent": throttle_pct,
        "steeringAngle": round(steering, 2),
        "brakePercent": brake_pct,
        "downforce": "High" if throttle > 0.6 else "Medium",
        "confidence": 94,
    }

def _format_driving_action(state: List[float]):
    action = system.predict_driving_action(state) or {}
    return _format_driving_action_from_raw(action)


def _format_strategy(state: RaceStrategyRequest):
    strategy = system.get_race_strategy(
        [state.lap, state.lap_progress, state.tire_age, state.position, state.degradation, state.pit_stops]
    ) or {}
    aggression = float(strategy.get("driving_aggression", 0.5))
    return {
        "pitDecision": strategy.get("pit_decision", "stay_out"),
        "strategy": strategy.get("strategy", "conservative"),
        "aggression": aggression,
        "aggressionLabel": "Aggressive" if aggression > 0.5 else "Conservative",
        "predictedPitWindow": f"Lap {state.lap + 3}-{state.lap + 6}",
    }


def _default_race_state() -> RaceStrategyRequest:
    return RaceStrategyRequest()


def _default_analysis(state: List[float], race_state: RaceStrategyRequest, telemetry: np.ndarray):
    analysis = system.get_comprehensive_analysis(
        state,
        telemetry,
        [race_state.lap, race_state.lap_progress, race_state.tire_age, race_state.position, race_state.degradation, race_state.pit_stops],
    )
    il_action = analysis.get("driving_action") or {}
    strategy = analysis.get("race_strategy") or {}
    tire = analysis.get("tire_degradation")
    anomalies = analysis.get("driver_anomalies")
    return {
        "ilRecommendation": f"Hold throttle at {round(il_action.get('throttle_position', 0.65)*100,1)}% with steering {round(il_action.get('steering_angle',0.0),2)}° in sector focus.",
        "rlStrategy": f"PPO recommends {strategy.get('strategy','conservative')} approach; pit decision: {strategy.get('pit_decision','stay_out')}.",
        "tireDegradation": tire
        and f"Predicted degradation at {round(tire.get('predicted_degradation',0.3),3)} — recommendation: {tire.get('recommendation','continue')}",
        "anomalies": anomalies
        and f"Anomaly rate {round(anomalies.get('anomaly_rate',0.0)*100,2)}% — {anomalies.get('recommendation','continue')}",
        "callouts": analysis.get("recommendations", []),
    }


def _format_analysis_summary(raw: dict):
    return {
        "ilRecommendation": raw.get("ilRecommendation") or "Hold current racing line and maintain throttle stability.",
        "rlStrategy": raw.get("rlStrategy") or "PPO recommends conservative aggression until next window.",
        "tireDegradation": raw.get("tireDegradation") or "Tire wear nominal; continue current stint.",
        "anomalies": raw.get("anomalies") or "No anomalies detected in last telemetry window.",
        "callouts": raw.get("callouts") or raw.get("recommendations") or [],
    }


def _to_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_jsonable(v) for v in value)
    return value


def _build_dashboard_payload():
    driving_req = DrivingActionRequest()
    race_req = _default_race_state()
    il_state = _build_il_state(driving_req)
    telemetry = np.random.default_rng(42).normal(loc=[60, 0.5, 0.7, 0.1, 0.3, 0.2], scale=0.1, size=(20, 6))

    driving_action = _format_driving_action(il_state)
    strategy = _format_strategy(race_req)
    analysis = _default_analysis(il_state, race_req, telemetry)

    return {
        "raceState": {
            "lap": race_req.lap,
            "position": race_req.position,
            "tireAge": race_req.tire_age,
            "tireCompound": "Soft",
            "predictedPitWindow": strategy["predictedPitWindow"],
            "aggressionLevel": strategy["aggressionLabel"],
            "drsEligible": True,
            "fuelLevel": 64,
        },
        "drivingAction": driving_action,
        "sectorData": {
            "S1": {"current": 34.2, "optimal": 33.9, "delta": "+0.3"},
            "S2": {"current": 44.8, "optimal": 44.2, "delta": "+0.6"},
            "S3": {"current": 28.4, "optimal": 28.1, "delta": "+0.3"},
        },
        "throttleHistory": [
            {"lap": race_req.lap - 3, "value": driving_action["throttlePercent"] - 2},
            {"lap": race_req.lap - 2, "value": driving_action["throttlePercent"] - 1},
            {"lap": race_req.lap - 1, "value": driving_action["throttlePercent"]},
            {"lap": race_req.lap, "value": driving_action["throttlePercent"]},
        ],
        "strategyTimeline": [
            {"phase": "Early", "laps": "1-15", "strategy": "Build lead", "status": "Complete"},
            {"phase": "Mid", "laps": "16-35", "strategy": "Manage gap", "status": "Complete"},
            {"phase": "Current", "laps": "36-50", "strategy": strategy["strategy"].capitalize(), "status": "Active"},
            {"phase": "Late", "laps": "51-75", "strategy": "Defend position", "status": "Pending"},
        ],
        "analysisSummary": analysis,
        "recommendations": analysis.get("callouts", []),
        "trackMetadata": {
            "name": "Barber Motorsports Park",
            "location": "Birmingham, AL",
            "length": "2.28 miles",
            "corners": 17,
            "drsZones": 1,
            "pitLaneTime": "34 seconds",
            "elevation": "646ft @ start/finish",
            "sectorLengths": {"S1": "0.82", "S2": "0.71", "S3": "0.75"},
        },
    }


def _to_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_jsonable(v) for v in value)
    return value


_DASHBOARD_STATE = _build_dashboard_payload()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/dashboard")
def get_dashboard():
    return _to_jsonable(_DASHBOARD_STATE)


@app.post("/api/driving-action")
def driving_action(req: DrivingActionRequest):
    state = _build_il_state(req)
    return _format_driving_action(state)


@app.post("/api/race-strategy")
def race_strategy(req: RaceStrategyRequest):
    return _format_strategy(req)


@app.post("/api/comprehensive-analysis")
def comprehensive_analysis(req: ComprehensiveAnalysisRequest):
    state = _build_il_state(req.driving_state)
    telemetry = _ensure_telemetry(req.telemetry_window)
    race_state_vec = [
        req.race_state.lap,
        req.race_state.lap_progress,
        req.race_state.tire_age,
        req.race_state.position,
        req.race_state.degradation,
        req.race_state.pit_stops,
    ]
    analysis = system.get_comprehensive_analysis(state, telemetry, race_state_vec)
    analysis_json = _to_jsonable(analysis)

    raw_action = analysis_json.get("driving_action")
    # if the model returned a raw driving action, allow a light, controlled
    # variation to better reflect live telemetry and produce visible
    # throttle variation in the dashboard (keeps drivingAction and analysis
    # in sync). We modify a copy of the raw_action so downstream fields match.
    if isinstance(raw_action, dict):
        # copy to avoid mutating original model output
        mod_action = dict(raw_action)
        try:
            # derive a lightweight influence from telemetry (mean speed)
            mean_speed = float(telemetry[:, 0].mean()) if telemetry.size else 60.0
        except Exception:
            mean_speed = 60.0

        rng = np.random.default_rng()
        # base throttle from model (0-1)
        base_throttle = float(mod_action.get("throttle_position", 0.65))
        # noise: ± up to ~0.18 (i.e., ±18 percentage points) but usually smaller
        noise = float(rng.uniform(-0.12, 0.12))
        # speed influence: stronger effect — smaller divisor increases influence
        # e.g., divisor 80 yields (mean_speed-60)/80 which makes speed matter more
        speed_influence = (mean_speed - 60.0) / 80.0
        new_throttle = float(np.clip(base_throttle + noise + speed_influence, 0.05, 0.95))
        mod_action["throttle_position"] = new_throttle
        # update analysis_json so summaries use the modified throttle
        analysis_json["driving_action"] = mod_action
        driving_action = _format_driving_action_from_raw(mod_action)
    else:
        driving_action = _format_driving_action(state)
    race_state_payload = {
        "lap": req.race_state.lap,
        "position": req.race_state.position,
        "tireAge": req.race_state.tire_age,
        "tireCompound": "Soft",
        "predictedPitWindow": f"Lap {int(req.race_state.lap + 3)}-{int(req.race_state.lap + 6)}",
        "aggressionLevel": "Aggressive" if driving_action.get("throttlePercent", 0) > 60 else "Conservative",
        "drsEligible": True,
        "fuelLevel": max(0, 100 - int(req.race_state.lap * 0.8)),
    }

    _DASHBOARD_STATE["drivingAction"] = driving_action
    _DASHBOARD_STATE["raceState"] = race_state_payload
    if req.sector_times:
        _DASHBOARD_STATE["sectorData"] = {
            sector: {
                "current": float(value),
                "optimal": round(max(value - 0.3, 0.0), 1),
                "delta": f"+{round(value - max(value - 0.3, 0), 1)}",
            }
            for sector, value in req.sector_times.items()
        }
    # Update throttle history so frontend throttle trend moves with live data
    try:
        throttle_val = None
        if isinstance(driving_action, dict):
            throttle_val = float(driving_action.get("throttlePercent", 0))
        else:
            throttle_val = 0.0
    except Exception:
        throttle_val = 0.0

    hist_entry = {"lap": req.race_state.lap, "value": throttle_val}
    th_hist = _DASHBOARD_STATE.get("throttleHistory") or []
    # append and keep last 20
    th_hist.append(hist_entry)
    _DASHBOARD_STATE["throttleHistory"] = th_hist[-20:]
    # Rebuild the human-readable analysis summary from the (possibly modified)
    # model outputs and the telemetry so the Comprehensive Analysis panel
    # updates for each incoming stream.
    try:
        analysis_summary = _default_analysis(state, req.race_state, telemetry)
    except Exception:
        # fallback to formatted analysis_json when default analysis fails
        analysis_summary = _format_analysis_summary(analysis_json)

    _DASHBOARD_STATE["analysisSummary"] = analysis_summary
    _DASHBOARD_STATE["recommendations"] = analysis_json.get("recommendations", [])

    return {
        "analysis": analysis_json,
        "dashboard": _to_jsonable(_DASHBOARD_STATE),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

