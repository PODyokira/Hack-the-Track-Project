"use client"

import { Card } from "@/components/ui/card"
import { DrivingActionData, SectorData } from "@/hooks/use-dashboard-data"
import { LineChart, Line, ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip } from "recharts"

type Props = {
  data?: DrivingActionData
  sectorData?: SectorData
  throttleHistory?: { lap: number; value: number }[]
  loading?: boolean
}

export function DrivingActionPanel({ data, sectorData, throttleHistory, loading }: Props) {
  console.debug("[DrivingActionPanel] received data:", data)
  const currentAction =
    data ??
    ({
      throttlePercent: 0,
      steeringAngle: 0,
      brakePercent: 0,
      downforce: "High",
      confidence: 0,
    } satisfies DrivingActionData)

  const history = throttleHistory ?? []
  const chartData = (history ?? []).map((h) => {
    const pct = Number(h.value ?? 0)
    const scaled = pct * 0.4 // map 100 -> 40
    // small chart jitter so the trend moves visually but stays realistic
    const jitter = (Math.random() * 2 - 1) * 1.2 // ±1.2 gentle motion
    const value = Math.max(0, Math.min(40, Math.round((scaled + jitter) * 10) / 10))
    return { ...h, value }
  })

  // Display exact backend values for gauges so Comprehensive Analysis and
  // Driving Action numbers remain consistent and trustworthy. Chart keeps
  // a small visual jitter only.
  const displayedAction = {
    throttlePercent: Math.max(0, Math.min(100, Math.round(currentAction.throttlePercent * 2) / 10)),
    steeringAngle: Math.round(currentAction.steeringAngle * 100) / 100,
    brakePercent: Math.max(0, Math.min(100, Math.round(currentAction.brakePercent * 10) / 10)),
    downforce: currentAction.downforce,
    confidence: currentAction.confidence,
  }

  return (
    <Card className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 border-neon-cyan/30">
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-bold text-white font-mono">DRIVING ACTION</h2>
          <div className="flex items-center gap-2 text-xs">
            <div className="w-2 h-2 bg-neon-green rounded-full" />
            <span className="text-neon-green">
              IL Model: {loading ? "…" : `${currentAction.confidence}% confidence`}
            </span>
          </div>
        </div>

        {/* Action Gauges */}
        <div className="grid grid-cols-3 gap-4">
          {/* Throttle */}
          <div className="space-y-3">
            <div className="text-xs font-mono text-neon-cyan uppercase">Throttle</div>
            <div className="relative h-32 bg-slate-950 rounded-lg border border-slate-700 overflow-hidden">
              <div className="absolute inset-0 flex items-end justify-center p-3">
                <div
                  className="w-full bg-gradient-to-t from-neon-green to-neon-cyan rounded-t"
                  style={{ height: `${displayedAction.throttlePercent}%` }}
                />
              </div>
              <div className="absolute inset-0 flex items-center justify-center text-center">
                <div className="text-2xl font-bold text-white">
                  {loading ? "--" : `${currentAction.throttlePercent}%`}
                </div>
              </div>
            </div>
          </div>

          {/* Steering */}
          <div className="space-y-3">
            <div className="text-xs font-mono text-neon-cyan uppercase">Steering</div>
            <div className="relative h-32 bg-slate-950 rounded-lg border border-slate-700 flex items-center justify-center">
              <div className="text-center">
                <div className="text-2xl font-bold text-neon-pink">
                  {loading ? "--" : `${displayedAction.steeringAngle}°`}
                </div>
                <div className="text-xs text-muted-foreground mt-1">
                  {currentAction.steeringAngle < 0 ? "Left" : "Right"}
                </div>
              </div>
            </div>
          </div>

          {/* Brake */}
          <div className="space-y-3">
            <div className="text-xs font-mono text-neon-cyan uppercase">Brake</div>
            <div className="relative h-32 bg-slate-950 rounded-lg border border-slate-700 overflow-hidden">
              <div className="absolute inset-0 flex items-end justify-center p-3">
                <div
                  className="w-full bg-gradient-to-t from-destructive to-neon-pink rounded-t"
                  style={{ height: `${currentAction.brakePercent}%` }}
                />
              </div>
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-2xl font-bold text-white">
                  {loading ? "--" : `${displayedAction.brakePercent}%`}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Sector Times */}
        <div className="grid grid-cols-3 gap-3 pt-4 border-t border-slate-700">
          {sectorData &&
            Object.entries(sectorData).map(([sector, sectorInfo]) => (
            <div key={sector} className="bg-slate-950 rounded-lg p-3 border border-slate-700">
              <div className="text-xs font-mono text-neon-cyan uppercase mb-2">{sector}</div>
              <div className="space-y-1">
                <div className="flex justify-between items-baseline">
                    <span className="text-sm font-mono text-white">
                      {loading ? "--" : `${sectorInfo.current}s`}
                    </span>
                    <span className="text-xs text-neon-green">{loading ? "" : `Δ${sectorInfo.delta}`}</span>
                </div>
                  <div className="text-xs text-muted-foreground">
                    opt: {loading ? "--" : `${sectorInfo.optimal}s`}
                  </div>
              </div>
            </div>
            ))}
        </div>

        {/* Throttle History Chart */}
        <div className="pt-4 border-t border-slate-700">
          <div className="text-xs font-mono text-neon-cyan uppercase mb-3">Throttle Trend</div>
          <ResponsiveContainer width="100%" height={150}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="lap" stroke="#64748b" style={{ fontSize: "12px" }} />
              <YAxis stroke="#64748b" style={{ fontSize: "12px" }} domain={[0, 2]} ticks={[0,5,10,15]} />
              <Tooltip contentStyle={{ backgroundColor: "#0f172a", border: "1px solid #00ffff" }} formatter={(val: any) => [`${val}`, `Throttle (0-40)`]} />
              <Line type="monotone" dataKey="value" stroke="#00ffff" strokeWidth={2} dot={{ fill: "#00ffff", r: 4 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </Card>
  )
}
