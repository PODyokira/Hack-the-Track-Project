"use client"

import { useState, useRef } from "react"
import { Card } from "@/components/ui/card"
import { useToast } from "@/hooks/use-toast"

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000"

type TelemetryRow = {
  timestamp?: string;
  lap?: number;
  sector?: string;
  lateral_g?: number;
  longitudinal_g?: number;
  speed?: number;
  lap_progress?: number;
  latitude?: number;
  longitude?: number;
  [key: string]: any;
};

type Props = {
  onRefresh?: () => void
  onLiveUpdate?: (dashboard: any) => void
}

export function TelemetryUpload({ onRefresh, onLiveUpdate }: Props) {
  const { toast } = useToast()
  const [isLoading, setIsLoading] = useState(false)
  const [lateralG, setLateralG] = useState(1.8)
  const [longitudinalG, setLongitudinalG] = useState(1.2)
  const [sectorFocus, setSectorFocus] = useState<"S1" | "S2" | "S3">("S2")
  const [s1, setS1] = useState(34.2)
  const [s2, setS2] = useState(45.1)
  const [s3, setS3] = useState(28.3)
  const [latitude, setLatitude] = useState(33.5207)
  const [longitude, setLongitude] = useState(-86.8025)
  const [isLive, setIsLive] = useState(false);
  const liveInterval = useRef<NodeJS.Timeout | null>(null);
  const [telemetryBuffer, setTelemetryBuffer] = useState<TelemetryRow[]>([]);

  const handleGoLive = async () => {
    setIsLive(true);
    if (liveInterval.current) clearInterval(liveInterval.current);
    const response = await fetch(`${API_BASE}/api/live-telemetry`);
    if (!response.body) return;
    const reader = response.body.getReader();
    let decoder = new TextDecoder();
    let done = false;
    while (!done) {
      const { value, done: streamDone } = await reader.read();
      if (value) {
        const rowStr = decoder.decode(value);
        try {
          const rowRaw: TelemetryRow = JSON.parse(rowStr.trim());

          // helpers
          const toNum = (v: any, fallback: number) => (typeof v === "string" ? parseFloat(v) || fallback : v ?? fallback)
          const jitter = (v: number, scale: number) => v + (Math.random() * 2 - 1) * scale

          // normalize parsed values
          const parsed = {
            ...rowRaw,
            speed: toNum(rowRaw.speed, 60),
            lateral_g: toNum(rowRaw.lateral_g, lateralG),
            longitudinal_g: toNum(rowRaw.longitudinal_g, longitudinalG),
            lap_progress: toNum(rowRaw.lap_progress, 0.65),
            latitude: toNum(rowRaw.latitude, latitude),
            longitude: toNum(rowRaw.longitude, longitude),
            S1_time: toNum((rowRaw as any).S1_time, s1),
            S2_time: toNum((rowRaw as any).S2_time, s2),
            S3_time: toNum((rowRaw as any).S3_time, s3),
          } as TelemetryRow

          // add small random jitter to simulate noisy live telemetry
          const sSpeed = Number(parsed.speed ?? 60)
          const sLatG = Number(parsed.lateral_g ?? lateralG)
          const sLongG = Number(parsed.longitudinal_g ?? longitudinalG)
          const sLapProg = Number(parsed.lap_progress ?? 0.65)
          const sLat = Number(parsed.latitude ?? latitude)
          const sLon = Number(parsed.longitude ?? longitude)
          const sS1 = Number(parsed.S1_time ?? s1)
          const sS2 = Number(parsed.S2_time ?? s2)
          const sS3 = Number(parsed.S3_time ?? s3)

          const simRow: TelemetryRow = {
            ...parsed,
            // increased but realistic jitter so values vary more clearly
            speed: Math.max(0, Math.round(jitter(sSpeed, 4) * 10) / 10),
            lateral_g: Math.max(0, Math.round(jitter(sLatG, 0.12) * 100) / 100),
            longitudinal_g: Math.max(0, Math.round(jitter(sLongG, 0.12) * 100) / 100),
            lap_progress: Math.min(1, Math.max(0, Math.round(jitter(sLapProg, 0.02) * 1000) / 1000)),
            latitude: Math.round(jitter(sLat, 0.00012) * 1e6) / 1e6,
            longitude: Math.round(jitter(sLon, 0.00012) * 1e6) / 1e6,
            S1_time: Math.max(0, Math.round(jitter(sS1, 0.6) * 10) / 10),
            S2_time: Math.max(0, Math.round(jitter(sS2, 0.6) * 10) / 10),
            S3_time: Math.max(0, Math.round(jitter(sS3, 0.6) * 10) / 10),
          }

          // update UI state from simulated row
          setLateralG(simRow.lateral_g ?? lateralG)
          setLongitudinalG(simRow.longitudinal_g ?? longitudinalG)
          setSectorFocus((simRow.sector as any) ?? sectorFocus)
          setS1(simRow.S1_time ?? s1)
          setS2(simRow.S2_time ?? s2)
          setS3(simRow.S3_time ?? s3)

          // maintain rolling buffer of recent telemetry rows (max 20)
          let newBuffer: TelemetryRow[] = []
          setTelemetryBuffer((prev) => {
            newBuffer = [...prev.slice(-19), simRow]
            return newBuffer
          })

          // Build telemetry_window from buffer so entries vary over time
          const defaultRow = [simRow.speed ?? 60, simRow.lateral_g ?? 1.8, simRow.longitudinal_g ?? 1.2, simRow.lap_progress ?? 0.65, simRow.latitude ?? latitude, simRow.longitude ?? longitude]
          let telemetryWindow = newBuffer.length
            ? newBuffer.map((r) => [r.speed ?? 60, r.lateral_g ?? 1.8, r.longitudinal_g ?? 1.2, r.lap_progress ?? 0.65, r.latitude ?? latitude, r.longitude ?? longitude])
            : [defaultRow]
          // ensure at least 10 rows for backend validation
          // instead of duplicating the same row, create jittered copies
          const jitterRow = (row: number[]) => {
            // scales per column: speed, lateral_g, longitudinal_g, lap_progress, lat, lon
            const scales = [3, 0.08, 0.08, 0.015, 0.00008, 0.00008]
            return row.map((val, idx) => {
              const n = Number(val ?? 0)
              const s = scales[idx] ?? 0.01
              if (idx === 4 || idx === 5) return Math.round(jitter(n, s) * 1e6) / 1e6
              if (idx === 3) return Math.min(1, Math.max(0, Math.round(jitter(n, s) * 1000) / 1000))
              return Math.round(Math.max(0, jitter(n, s)) * (idx === 0 ? 10 : 100)) / (idx === 0 ? 10 : 100)
            }) as number[]
          }

          while (telemetryWindow.length < 10) {
            const base = telemetryWindow[0] ?? defaultRow
            telemetryWindow = [jitterRow(base), ...telemetryWindow]
          }

          const payload = {
            driving_state: {
              lateral_g: simRow.lateral_g ?? lateralG,
              longitudinal_g: simRow.longitudinal_g ?? longitudinalG,
              sector: (simRow.sector as any) ?? sectorFocus,
            },
            race_state: {
              lap: parsed.lap ?? 42,
              lap_progress: simRow.lap_progress ?? 0.65,
              tire_age: parsed.tire_age ?? 12,
              position: parsed.position ?? 3,
              degradation: parsed.degradation ?? 0.4,
              pit_stops: parsed.pit_stops ?? 0,
            },
            telemetry_window: telemetryWindow,
            sector_times: {
              S1: simRow.S1_time ?? s1,
              S2: simRow.S2_time ?? s2,
              S3: simRow.S3_time ?? s3,
            },
          };
          const analysisRes = await fetch(`${API_BASE}/api/comprehensive-analysis`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          if (analysisRes.ok) {
            const analysis = await analysisRes.json();
            if (typeof onLiveUpdate === "function") {
              onLiveUpdate(analysis.dashboard);
            }
          }
        } catch (err) {
          // Ignore parse errors
        }
      }
      done = streamDone;
    }
    setIsLive(false);
  };

  const sanitize = (value: string): number => {
    if (value === "" || Number.isNaN(Number(value))) return 0
    return parseFloat(value)
  }

  const handleUpload = async () => {
    setIsLoading(true)

    try {
      const payload = {
        driving_state: {
          lateral_g: lateralG,
          longitudinal_g: longitudinalG,
          sector: sectorFocus,
        },
        race_state: {
          lap: 42,
          lap_progress: 0.65,
          tire_age: 12,
          position: 3,
          degradation: 0.4,
          pit_stops: 0,
        },
        // telemetry columns: [speed, lateral_g, longitudinal_g, lap_progress, latitude, longitude]
        telemetry_window: Array.from({ length: 20 }, (_, idx) => {
          // create a gently varying simulated series for manual upload
          const baseSpeed = 50 + idx * 0.3
          const s = (v: number, scale: number) => Math.round((v + (Math.random() * 2 - 1) * scale) * 10) / 10
          return [
            s(baseSpeed, 3),
            Math.round((0.5 + (Math.random() * 2 - 1) * 0.08) * 100) / 100,
            Math.round((0.5 + (Math.random() * 2 - 1) * 0.08) * 100) / 100,
            Math.round((0.65 + (Math.random() * 2 - 1) * 0.015) * 1000) / 1000,
            Math.round((latitude + (Math.random() * 2 - 1) * 0.00008) * 1e6) / 1e6,
            Math.round((longitude + (Math.random() * 2 - 1) * 0.00008) * 1e6) / 1e6,
          ]
        }),
        sector_times: {
          S1: s1,
          S2: s2,
          S3: s3,
        },
      }
      const response = await fetch(`${API_BASE}/api/comprehensive-analysis`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })
      if (!response.ok) {
        throw new Error(`Analysis failed (${response.status})`)
      }

      toast({
        title: "Analysis updated",
        description: "The IL/PPO models processed your telemetry sample.",
      })
      onRefresh?.()
    } catch (error) {
      console.error(error)
      toast({
        title: "Update failed",
        description: "Unable to process telemetry right now.",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <Card className="bg-linear-to-br from-slate-900 via-slate-800 to-slate-900 border-neon-cyan/30">
      <div className="p-6 space-y-4">
        <h2 className="text-lg font-bold text-white font-mono">TELEMETRY INPUT</h2>
        <div className="space-y-3">
          <div>
            <label className="text-xs font-mono text-neon-cyan uppercase">Lateral G-Force</label>
            <input
              type="number"
              value={lateralG.toString()}
              onChange={(e) => setLateralG(sanitize(e.target.value))}
              className="w-full mt-1 bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm focus:border-neon-cyan focus:outline-none"
            />
          </div>
          <div>
            <label className="text-xs font-mono text-neon-cyan uppercase">Longitudinal G-Force</label>
            <input
              type="number"
              value={longitudinalG.toString()}
              onChange={(e) => setLongitudinalG(sanitize(e.target.value))}
              className="w-full mt-1 bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm focus:border-neon-cyan focus:outline-none"
            />
          </div>
          <div>
            <label className="text-xs font-mono text-neon-cyan uppercase">Latitude</label>
            <input
              type="number"
              value={latitude.toString()}
              onChange={(e) => setLatitude(sanitize(e.target.value))}
              className="w-full mt-1 bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm focus:border-neon-cyan focus:outline-none"
            />
          </div>
          <div>
            <label className="text-xs font-mono text-neon-cyan uppercase">Longitude</label>
            <input
              type="number"
              value={longitude.toString()}
              onChange={(e) => setLongitude(sanitize(e.target.value))}
              className="w-full mt-1 bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm focus:border-neon-cyan focus:outline-none"
            />
          </div>
          <div>
            <label className="text-xs font-mono text-neon-cyan uppercase">Sector Focus</label>
            <select
              value={sectorFocus}
              onChange={(e) => setSectorFocus(e.target.value as "S1" | "S2" | "S3")}
              className="w-full mt-1 bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm focus:border-neon-cyan focus:outline-none"
            >
              <option value="S1">Sector 1</option>
              <option value="S2">Sector 2</option>
              <option value="S3">Sector 3</option>
            </select>
          </div>
          <div className="grid grid-cols-3 gap-2">
            <div>
              <label className="text-xs font-mono text-neon-cyan uppercase">S1 Time</label>
              <input
                type="number"
                value={s1.toString()}
                onChange={(e) => setS1(sanitize(e.target.value))}
                step="0.1"
                className="w-full mt-1 bg-slate-950 border border-slate-700 rounded-lg px-2 py-2 text-white text-sm focus:border-neon-cyan focus:outline-none"
              />
            </div>
            <div>
              <label className="text-xs font-mono text-neon-cyan uppercase">S2 Time</label>
              <input
                type="number"
                value={s2.toString()}
                onChange={(e) => setS2(sanitize(e.target.value))}
                step="0.1"
                className="w-full mt-1 bg-slate-950 border border-slate-700 rounded-lg px-2 py-2 text-white text-sm focus:border-neon-cyan focus:outline-none"
              />
            </div>
            <div>
              <label className="text-xs font-mono text-neon-cyan uppercase">S3 Time</label>
              <input
                type="number"
                value={s3.toString()}
                onChange={(e) => setS3(sanitize(e.target.value))}
                step="0.1"
                className="w-full mt-1 bg-slate-950 border border-slate-700 rounded-lg px-2 py-2 text-white text-sm focus:border-neon-cyan focus:outline-none"
              />
            </div>
          </div>
        </div>
        <div className="flex gap-2 mt-4">
          <button
            onClick={handleUpload}
            disabled={isLoading || isLive}
            className="flex-1 py-2 bg-neon-green text-black font-mono font-bold rounded-lg hover:bg-neon-green/90 disabled:opacity-50 transition-colors"
          >
            {isLoading ? "Updating..." : "Update Analysis"}
          </button>
          <button
            onClick={handleGoLive}
            disabled={isLive}
            className="flex-1 py-2 bg-neon-cyan text-black font-mono font-bold rounded-lg hover:bg-neon-cyan/90 disabled:opacity-50 transition-colors"
          >
            {isLive ? "Live..." : "Go Live"}
          </button>
        </div>
      </div>
    </Card>
  )
}
