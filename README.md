# RealtimeRacingStrategyEngine – AI-Powered Race Simulation & Analysis

## 📋 Project Overview

The **RealtimeRacingStrategyEngine** is an intelligent racing strategy and telemetry analysis system that combines **Imitation Learning (IL)** and **Reinforcement Learning (RL)** models to simulate and optimize race car driving performance in real-time. The system streams live telemetry data from real race records (Barber Motorsports Park), analyzes it using hybrid AI models, and displays actionable insights through a modern web dashboard.

### Key Features
- **Real-time telemetry streaming** from historical race data (CSV format)
- **Imitation Learning model** that predicts optimal throttle, steering, and brake inputs
- **PPO Reinforcement Learning agent** that recommends pit strategies and aggression levels
- **Tire degradation analysis** using LSTM predictions
- **Anomaly detection** to identify driving irregularities
- **Live dashboard** showing driving action gauges, sector times, throttle trends, and race strategy recommendations
- **Comprehensive analysis** that updates per streamed telemetry sample

---

## 🏗️ Architecture

### Tech Stack

**Backend:**
- **FastAPI** (Python) – RESTful API server
- **Uvicorn** – ASGI application server
- **NumPy & PyTorch** – ML model inference
- **Pydantic** – Data validation

**Frontend:**
- **Next.js 16** (React) – Modern web framework with server/client components
- **TypeScript** – Type-safe development
- **Tailwind CSS** – Styling & layout
- **Recharts** – Real-time chart visualization
- **Turbopack** – Fast bundler

**Data:**
- CSV telemetry files from Barber Motorsports Park races
- Processed datasets for model training

---

## 📊 System Components

### 1. **Backend (`Hack-the-Track-Project/`)**

#### Core Files

**`api_server.py`** – Main FastAPI application
- **GET `/api/live-telemetry`** – Streams CSV rows as JSON, one row per 3 seconds
- **POST `/api/comprehensive-analysis`** – Accepts telemetry window (20 rows), returns driving action, strategy, and analysis
- **GET `/api/dashboard`** – Retrieves cached dashboard state
- **POST `/api/driving-action`** – Predicts throttle/steering/brake from lateral/longitudinal G-forces
- **POST `/api/race-strategy`** – Recommends pit timing and aggression levels

**`integrated_system.py`** – Orchestrates all ML models
- Initializes IL model, PPO agent, LSTM tire predictor, and anomaly detector
- `get_comprehensive_analysis()` – Runs all models on telemetry and returns combined insights
- `predict_driving_action()` – IL model output (throttle, steering)
- `get_race_strategy()` – PPO model output (pit decision, aggression)

**`imitation_learning.py`** – Imitation Learning model
- Learns optimal driving from labeled race data
- Predicts throttle and steering angles based on G-forces and sector

**`ppo_rl.py`** – Proximal Policy Optimization (PPO) agent
- Trains on race strategy decisions
- Recommends pit timing, fuel management, and aggression levels

**`lstm_predictor.py`** – Long Short-Term Memory network
- Predicts tire degradation over race stints
- Alerts when tire wear reaches critical thresholds

**`anomaly_detection.py`** – Detects driving anomalies
- Identifies unusual braking, acceleration, or cornering patterns
- Flags data outliers or equipment issues

**`data_preprocessing.py`** – Prepares raw telemetry
- Cleans CSV data, normalizes values, engineers features

#### Data Flow (Backend)

```
CSV File (barber/R1_barber_telemetry_data.csv)
    ↓
[Live Telemetry Endpoint] streams rows
    ↓
[Frontend receives row] → builds telemetry window (20 rows)
    ↓
[POST /comprehensive-analysis]
    ↓
[IntegratedRacingSystem processes telemetry]
    ├→ IL Model (throttle/steering)
    ├→ PPO Agent (pit strategy)
    ├→ LSTM (tire degradation)
    └→ Anomaly Detector (irregularities)
    ↓
[Dashboard State Updated]
    ├→ drivingAction (updated with modified throttle)
    ├→ analysisSummary (human-readable insights)
    ├→ throttleHistory (appended for trend chart)
    └→ raceState (lap, position, fuel, etc.)
    ↓
[Return dashboard payload to frontend]
```

---

### 2. **Frontend (`Front/`)**

#### Core Components

**`app/page.tsx`** – Main dashboard layout
- Composes all dashboard panels
- Manages `liveData` state from streaming telemetry
- Uses `useDashboardData()` hook to fetch initial dashboard
- Passes `setLiveData` callback to `TelemetryUpload` to receive streamed updates

**`components/telemetry-upload.tsx`** – Telemetry input & live streaming control
- **"Go Live" button** – Starts streaming CSV rows from backend
- Builds rolling telemetry buffer (max 20 rows)
- Applies jitter to streamed rows so each sample varies realistically
- Posts telemetry windows (20 rows of speed, lateral_g, longitudinal_g, lap_progress, lat/lon) to `/api/comprehensive-analysis`
- Calls `onLiveUpdate(dashboard)` callback with results

**`components/driving-action-panel.tsx`** – Real-time driving metrics
- Displays three gauges: **Throttle**, **Steering**, **Brake** (exact backend values)
- Shows sector times (S1, S2, S3) with optimal lap time deltas
- **Throttle Trend Chart** – Line chart showing throttle history over last 20 entries, scaled 0–40% for clarity
- Updates on each streamed telemetry sample

**`components/comprehensive-analysis.tsx`** – AI recommendations
- **IL Recommendation** – Optimal throttle & steering from Imitation Learning
- **RL Strategy** – Pit decision and aggression level from PPO
- **Tire Degradation** – Predicted compound wear and pit window
- **Anomalies** – Any detected driving irregularities
- **System Callouts** – Key insights from all models

**`components/race-state-hero.tsx`** – Race status overview
- Current lap, position, tire age, fuel level
- Predicted pit window and DRS eligibility

**`components/strategy-timeline.tsx`** – Race phases & quick actions
- Displays early, mid, current, and late-race phases
- Strategy labels (e.g., "Manage gap", "Defend position")

**`components/track-metadata.tsx`** – Track information
- Barber Motorsports Park stats (length, corners, elevation, sector lengths)

#### Hooks

**`hooks/use-dashboard-data.ts`** – Fetches initial dashboard from `/api/dashboard`
- Provides `data`, `loading`, `refresh()` for static mode

**`hooks/use-toast.ts`** – Toast notifications for user feedback

#### Data Flow (Frontend)

```
[Page mounts]
    ↓
[useDashboardData hooks fetches /api/dashboard]
    ↓
[dashboardData = liveData || data]
    ↓
[Render panels with dashboardData]
    ↓
[User clicks "Go Live"]
    ↓
[TelemetryUpload streams rows]
    ├→ Parse each row from backend
    ├→ Append to telemetry buffer (max 20)
    ├→ Build telemetry_window
    └→ POST /comprehensive-analysis
    ↓
[Backend returns analysis.dashboard]
    ↓
[setLiveData(analysis.dashboard)]
    ↓
[dashboardData = liveData (now updated)]
    ↓
[All panels re-render with new values]
    ↓
[Throttle chart updates, gauges change, analysis text refreshes]
```

---

## 🚀 Key Algorithms & Models

### Imitation Learning (IL) Model
- **Goal:** Learn to drive like a human race driver
- **Input:** Lateral G-force, longitudinal G-force, sector (S1/S2/S3)
- **Output:** Throttle position (0–1), steering angle (°)
- **Training Data:** Real race telemetry labeled with ideal actions

### PPO (Proximal Policy Optimization) Agent
- **Goal:** Optimize race strategy (pit timing, aggression)
- **Input:** Lap number, lap progress, tire age, position, degradation, pit stops
- **Output:** Pit decision (stay_out / pit_now), strategy (aggressive / conservative), aggression level
- **Training Method:** Reinforcement learning to maximize race finish position

### LSTM Tire Predictor
- **Goal:** Forecast tire degradation over stint
- **Input:** Historical lap times, lateral/longitudinal forces per lap
- **Output:** Predicted degradation %, pit window recommendation
- **Benefit:** Prevents under/over-braking due to tire wear

### Anomaly Detection
- **Goal:** Identify unusual driving patterns
- **Method:** Statistical outlier detection on acceleration, braking, cornering
- **Output:** Anomaly rate %, recommendation flags

---

## 🔄 Live Telemetry Simulation

### How It Works

1. **User clicks "Go Live"** in the dashboard
2. **Frontend requests `/api/live-telemetry`** (streaming endpoint)
3. **Backend streams CSV rows** from `barber/R1_barber_telemetry_data.csv` at 3-second intervals as JSON lines
4. **Frontend receives each row:**
   - Parses the telemetry (speed, lateral_g, longitudinal_g, lap_progress, etc.)
   - Applies small random jitter (±4 speed units, ±0.12 G-forces) to simulate real sensor noise
   - Appends to rolling buffer (max 20 rows)
5. **Frontend builds telemetry_window** from buffer (ensures ≥10 rows, pads with jittered copies if needed)
6. **Frontend POSTs to `/api/comprehensive-analysis`** with:
   - Telemetry window (20 rows × 6 columns)
   - Driving state (lateral/longitudinal G, sector focus)
   - Race state (lap, position, tire age, degradation, pit stops)
   - Sector times (S1, S2, S3)
7. **Backend processes:**
   - All ML models infer from telemetry
   - Modified throttle: base model output + small noise + speed influence
   - Brake: computed as (1 - throttle) scaled to 0–60%, capped relative to throttle
   - Dashboard state updated with new values
8. **Backend returns updated dashboard** containing:
   - drivingAction (throttle %, steering °, brake %)
   - analysisSummary (human-readable IL/RL/tire/anomaly insights)
   - throttleHistory (latest entry appended, last 20 kept)
   - raceState (updated lap, fuel, pit window)
9. **Frontend receives dashboard** and calls `setLiveData(dashboard)`
10. **All panels re-render** with fresh values
11. **Repeat** for each streamed telemetry row

### Realistic Variation

- **Throttle range:** 5–95% (determined by telemetry and model + random noise)
- **Brake computation:** Max ~60% and lower than throttle to simulate realistic car control
- **Telemetry jitter:** Each sample includes ±4 km/h speed, ±0.12 G-force variation to prevent identical inputs
- **Speed influence:** Mean telemetry speed affects throttle (faster → higher throttle bias)

---

## 📈 Dashboard Panels

| Panel | Purpose | Updates Per Stream |
|-------|---------|-------------------|
| **Driving Action** | Throttle, steering, brake gauges + sector times | ✅ Yes |
| **Throttle Trend** | Line chart of throttle history (last 20 entries) | ✅ Yes |
| **Comprehensive Analysis** | IL/RL/tire/anomaly insights | ✅ Yes |
| **Race State Hero** | Lap, position, tire age, fuel, pit window | ✅ Yes |
| **Track Metadata** | Barber Motorsports Park info | ❌ Static |
| **Strategy Timeline** | Race phases (early/mid/current/late) | ✅ Updates per stream |

---

## 🔧 Setup & Running

### Prerequisites
- **Python 3.12+** (backend)
- **Node.js 18+** (frontend)
- **pnpm** or **npm** (Node package manager)

### Backend Setup

```bash
cd "Hack-the-Track-Project"

# Install Python dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn api_server:app --reload
```

Server runs on `http://localhost:8000`. API docs available at `http://localhost:8000/docs`.

### Frontend Setup

```bash
cd "Front"

# Install Node dependencies
pnpm install
# or: npm install

# Create .env.local with API endpoint
echo "NEXT_PUBLIC_API_BASE_URL=http://localhost:8000" > .env.local

# Start development server
npx next dev
```

Frontend runs on `http://localhost:3000`.

### 🧪 Run Automated Tests & Train Models

From the `Hack-the-Track-Project/` directory:

```bash
# Train all ML models (IL, PPO, LSTM, anomaly detection)
python train_all.py

# Run full test suite
python test_system.py

# Run presentation demo
python hackathon_demo.py

# Run basic examples
python example_usage.py
```

These scripts validate the ML models, demonstrate core functionality, and verify the integrated system works end-to-end.

---

## 🚀 Running the Full System

**Terminal 1 – Backend:**
```bash
cd "Hack-the-Track-Project"
uvicorn api_server:app --reload
```

**Terminal 2 – Frontend:**
```bash
cd "Front"
npx next dev
```

Then open `http://localhost:3000` in your browser and click **Go Live** to stream telemetry and see the dashboard update in real-time.

---

## 📁 Project Structure

```
Toyota comp/
├── Front/                          # Next.js React frontend
│   ├── app/
│   │   ├── page.tsx               # Main dashboard
│   │   ├── layout.tsx             # App layout
│   │   └── globals.css            # Global styles
│   ├── components/
│   │   ├── driving-action-panel.tsx
│   │   ├── comprehensive-analysis.tsx
│   │   ├── telemetry-upload.tsx
│   │   ├── race-state-hero.tsx
│   │   ├── strategy-timeline.tsx
│   │   ├── track-metadata.tsx
│   │   └── ui/                    # Shadcn UI components
│   ├── hooks/
│   │   ├── use-dashboard-data.ts  # Fetch dashboard hook
│   │   ├── use-toast.ts
│   │   └── use-mobile.ts
│   ├── lib/
│   │   └── utils.ts
│   ├── package.json
│   ├── tsconfig.json
│   ├── postcss.config.mjs
│   └── next.config.mjs
│
└── Hack-the-Track-Project/        # FastAPI + ML backend
    ├── api_server.py              # FastAPI endpoints
    ├── integrated_system.py       # ML orchestration
    ├── imitation_learning.py      # IL model
    ├── ppo_rl.py                  # PPO agent
    ├── lstm_predictor.py          # Tire degradation LSTM
    ├── anomaly_detection.py       # Anomaly detector
    ├── data_preprocessing.py      # Data cleaning
    ├── barber/                    # Race telemetry CSV files
    │   ├── R1_barber_telemetry_data.csv
    │   ├── R2_barber_telemetry_data.csv
    │   └── ...                    # Other race/weather data
    ├── models/                    # Trained ML models
    │   ├── imitation_learning_model.pth
    │   ├── ppo_agent.pth
    │   ├── lstm_model.pth
    │   └── ...
    ├── processed_data/            # Preprocessed datasets
    ├── requirements.txt
    └── README.md
```

---

## 🎯 Use Cases

### 1. **Race Strategy Optimization**
- Drivers can simulate live telemetry and see real-time pit recommendations
- Understand optimal throttle/steering curves for each sector

### 2. **Talent Development**
- New drivers learn from IL model's optimal inputs
- Compare their driving to the imitation learning baseline

### 3. **Equipment Analysis**
- Tire degradation predictions help estimate pit windows
- Anomaly detection flags mechanical issues early

### 4. **Live Race Support**
- Dashboard can stream live or historical race data
- Teams make data-driven decisions in pit lane

---

## 🔑 Key Design Decisions

### Telemetry Buffering
- Rolling 20-row buffer ensures models always receive recent context
- Prevents "stale" predictions from old data

### Controlled Jitter
- Realistic ±4 km/h speed variation prevents identical rows
- Models see natural telemetry fluctuation, produce varied outputs

### Modified Throttle Logic
- Base throttle from IL model + speed influence + random noise
- Ensures throttle varies 5–95% range for realistic visuals

### Brake Computation
- Brake = (1 - throttle) × 60%, capped to not exceed throttle
- Prevents unrealistic 90% braking while throttling high

### Dashboard State Architecture
- Single `_DASHBOARD_STATE` dict on backend updated per request
- Frontend prioritizes `liveData` (streamed) over `data` (static)
- All panels automatically re-render on state updates

---

## 📊 Example Live Session Flow

```
[Dashboard loads at 12:00:00]
User sees: Throttle 65%, Brake 21%, Steering -2.3°
Analysis: "Hold throttle at 65% with steering -2.3° in S2 sector focus"

[User clicks Go Live]

[12:00:03] Stream row 1 → Throttle 58%, Brake 25%, Steering 1.8°
[12:00:06] Stream row 2 → Throttle 72%, Brake 17%, Steering -4.2°
[12:00:09] Stream row 3 → Throttle 61%, Brake 23%, Steering 0.5°
...
[12:00:57] Stream row 20 → Throttle 68%, Brake 19%, Steering -1.1°

Throttle Trend chart: Shows 20 points oscillating between ~55% and ~75%
Comprehensive Analysis: "IL Model recommends 68% throttle positioning"
Race Strategy: "PPO suggests aggressive driving, pit window Lap 45-48"
```

---

## 🎨 UI Features

- **Real-time gauge updates** – Throttle, steering, brake change per streamed sample
- **Line chart visualization** – Throttle trend shows realistic up/down movement
- **Color-coded metrics** – Neon cyan, green, pink for visual hierarchy
- **Sector time deltas** – Shows improvement/degradation vs. optimal
- **Responsive layout** – Works on desktop (optimized for 1920×1080+)
- **Dark theme** – Slate/neon color scheme mimics professional racing telemetry displays

---

## 🚨 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Cannot connect to API" | Ensure backend is running on port 8000 and `.env.local` has correct URL |
| Telemetry not streaming | Check `barber/R1_barber_telemetry_data.csv` exists; backend may need file path update |
| Dashboard not updating | Open browser console; check for `[liveData]` logs; verify network tab shows POST 200 OK |
| Models not found | Run backend from `Hack-the-Track-Project/` directory; `models/` folder must be present |
| Jitter too extreme | Adjust noise scales in `telemetry-upload.tsx` (jitter scales) or `api_server.py` (divisor in speed_influence) |

---

## 📝 License & Credits

**Project:** GR Cup Strategy Engine (Hackathon/Portfolio project)  
**Track:** Barber Motorsports Park, Birmingham, AL  
**Data:** Real race telemetry from Toyota GR Cup series  
**Models:** Custom IL, PPO, LSTM, and anomaly detection implementations  

---

## 🎓 What You've Built

A **full-stack AI-powered racing simulation system** that:
- ✅ Streams real race telemetry in real-time
- ✅ Feeds data to hybrid ML models (IL + RL + LSTM + anomaly detection)
- ✅ Returns actionable insights instantly
- ✅ Displays live dashboards with realistic variation
- ✅ Scales from historical data to live race support

---

## 💡 Future Enhancements

- Multi-lap history for trend analysis
- Lap-by-lap comparison (yours vs. optimal baseline)
- Tire pressure & temperature monitoring
- Fuel consumption predictions
- Multi-driver competitive analysis
- Real-time pit crew alerts
- Mobile app for pit lane feedback

---

## 👥 Project Team

**Developed by:**

- **Najb Yassine** 
- **Hammach Oussama** 

This project was created with passion for racing, AI engineering, and real-time systems. Both team members contributed equally to the architecture, implementation, and optimization of the RealtimeRacingStrategyEngine.

---

**Made with ❤️ for racing enthusiasts & AI engineers**
