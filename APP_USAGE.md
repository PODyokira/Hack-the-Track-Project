## Racing Strategy Dashboard – Usage Guide

This doc explains how to launch the full stack (FastAPI backend + Next.js frontend) and how to interpret each panel in the UI.

---

### 1. Start the backend (FastAPI + IL/PPO engine)

```bash
cd "C:\Users\podyo\OneDrive\Desktop\Toyota comp\Hack-the-Track-Project"
pip install -r requirements.txt
uvicorn api_server:app --reload
```

You should see `Integrated system ready!` and requests hitting `/api/dashboard` once the frontend connects.

### 2. Start the frontend (Next.js dashboard)

In a **new** terminal:

```bash
cd "C:\Users\podyo\OneDrive\Desktop\Toyota comp\Front"
npm install
$env:NEXT_PUBLIC_API_BASE_URL = "http://localhost:8000"   # PowerShell
npm run dev
```

Visit `http://localhost:3000`. The dashboard now streams live data from the backend.

---

### 3. Understanding the dashboard

| Section | Description |
| --- | --- |
| **Driving Action** | Shows the latest imitation-learning output: throttle %, steering angle, brake %, sector splits, and throttle trend. Updates whenever `/api/dashboard` is refreshed. |
| **Track Metadata** | Static info about Barber Motorsports Park (length, corners, pit time, sector distances). |
| **Telemetry Input** | Lets you tweak lateral/longitudinal G forces, sector focus, and sector times. Clicking **Update Analysis** sends a request to `/api/comprehensive-analysis`, which recomputes IL/PPO outputs and refreshes the cards. |
| **Race Phases** | Shows the current strategy timeline (early/mid/current/late). The highlighted phase (“Current”) reflects the latest PPO recommendation. |
| **Comprehensive Analysis** | Summaries of IL, PPO, Tire, and Anomaly callouts plus action buttons (Export/Refresh). Automatically refreshes after each telemetry upload. |

---

### 4. Typical flow

1. Launch backend + frontend.
2. Observe baseline state (dashboard auto-fetches `/api/dashboard` on load).
3. Enter new telemetry values (e.g., drop lateral G to 0.8, adjust sector times).
4. Press **Update Analysis**:
   - The frontend POSTs to `/api/comprehensive-analysis`.
   - Backend recomputes IL action, PPO strategy, callouts, etc.
   - UI refreshes with the new values (heroes, gauges, analysis text).

You can repeat the telemetry update loop as often as needed to test scenarios.

---

### 5. API reference (for custom integrations)

- `GET /api/dashboard` – returns the entire dashboard snapshot.
- `POST /api/driving-action` – raw IL output for a given G-force+sector state.
- `POST /api/race-strategy` – PPO recommendation for a lap/tire/position state.
- `POST /api/comprehensive-analysis` – runs IL + PPO + analysis and updates dashboard state.

Payload examples live in `components/telemetry-upload.tsx`.

---

### 6. Troubleshooting

- **Frontend shows “Failed to fetch”**: ensure `uvicorn` is running and `NEXT_PUBLIC_API_BASE_URL` matches the backend URL.
- **Inputs turn blank/NaN**: keep numeric fields non-empty; the UI clamps empty strings to `0`.
- **No changes after update**: verify the backend console shows a `POST /api/comprehensive-analysis ... 200 OK`. If not, check the terminal for errors.

Feel free to extend the telemetry form (lap, position, etc.)—the backend already accepts those values via `race_state`.

