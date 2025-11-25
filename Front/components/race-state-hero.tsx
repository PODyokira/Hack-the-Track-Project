"use client"

import { Card } from "@/components/ui/card"
import { RaceState } from "@/hooks/use-dashboard-data"

type Props = {
  data?: RaceState
  loading?: boolean
}

export function RaceStateHero({ data, loading }: Props) {
  const raceState =
    data ??
    ({
      lap: 0,
      position: 0,
      tireAge: 0,
      tireCompound: "Soft",
      predictedPitWindow: "--",
      aggressionLevel: "Calculating",
      drsEligible: false,
      fuelLevel: 0,
    } satisfies RaceState)

  return (
    <Card className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 border-neon-cyan/30 overflow-hidden">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_1px_1px,_#00ffff_1px,_transparent_1px)] bg-[length:40px_40px] opacity-5" />

      <div className="relative p-8">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          {/* Lap */}
          <div className="space-y-2">
            <div className="text-xs font-mono text-neon-cyan uppercase tracking-wider">Lap</div>
            <div className="text-4xl font-bold text-white">{raceState.lap}</div>
            <div className="text-xs text-muted-foreground">{loading ? "Loading…" : "Current lap"}</div>
          </div>

          {/* Position */}
          <div className="space-y-2">
            <div className="text-xs font-mono text-neon-cyan uppercase tracking-wider">Position</div>
            <div className="text-4xl font-bold text-white">P{raceState.position}</div>
            <div className="text-xs text-muted-foreground">{loading ? "Updating…" : "+2.3s to leader"}</div>
          </div>

          {/* Tire Info */}
          <div className="space-y-2">
            <div className="text-xs font-mono text-neon-cyan uppercase tracking-wider">Tires</div>
            <div className="text-2xl font-bold text-white">{raceState.tireCompound}</div>
            <div className="text-xs text-muted-foreground">Age: {raceState.tireAge} laps</div>
          </div>

          {/* Strategy */}
          <div className="space-y-2">
            <div className="text-xs font-mono text-neon-cyan uppercase tracking-wider">Pit Window</div>
            <div className="text-2xl font-bold text-neon-green">{raceState.predictedPitWindow}</div>
            <div className="text-xs text-muted-foreground">{loading ? "Calculating…" : "Recommended"}</div>
          </div>
        </div>

        {/* Bottom Row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mt-8 pt-8 border-t border-slate-700">
          {/* Aggression */}
          <div className="space-y-2">
            <div className="text-xs font-mono text-neon-purple uppercase tracking-wider">Aggression</div>
            <div className="text-xl font-bold text-neon-purple">{raceState.aggressionLevel}</div>
          </div>

          {/* DRS */}
          <div className="space-y-2">
            <div className="text-xs font-mono text-neon-pink uppercase tracking-wider">DRS Status</div>
            <div className={`text-xl font-bold ${raceState.drsEligible ? "text-neon-green" : "text-slate-500"}`}>
              {raceState.drsEligible ? "ELIGIBLE" : "UNAVAILABLE"}
            </div>
          </div>

          {/* Fuel */}
          <div className="space-y-2">
            <div className="text-xs font-mono text-neon-cyan uppercase tracking-wider">Fuel Level</div>
            <div className="text-xl font-bold text-white">{raceState.fuelLevel}%</div>
          </div>

          {/* Status */}
          <div className="space-y-2">
            <div className="text-xs font-mono text-neon-cyan uppercase tracking-wider">Model Status</div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-neon-green rounded-full animate-pulse" />
              <span className="text-white">{loading ? "Syncing" : "Running"}</span>
            </div>
          </div>
        </div>
      </div>
    </Card>
  )
}
