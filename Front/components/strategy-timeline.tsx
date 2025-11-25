"use client"

import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { StrategyItem } from "@/hooks/use-dashboard-data"

type Props = {
  items?: StrategyItem[]
}

export function StrategyTimeline({ items }: Props) {
  const scenarios =
    items ??
    [
      { phase: "Early", laps: "1-15", strategy: "Build lead", status: "Complete" },
      { phase: "Mid", laps: "16-35", strategy: "Manage gap", status: "Complete" },
      { phase: "Current", laps: "36-50", strategy: "Aggressive push", status: "Active" },
      { phase: "Late", laps: "51-75", strategy: "Defend position", status: "Pending" },
    ]

  return (
    <Card className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 border-neon-cyan/30">
      <div className="p-6 space-y-4">
        <h2 className="text-lg font-bold text-white font-mono">RACE PHASES</h2>

        <div className="space-y-3">
          {scenarios.map((scenario, i) => (
            <div
              key={i}
              className={`p-3 rounded-lg border transition-colors ${
                scenario.status === "Active"
                  ? "bg-slate-800 border-neon-green/50"
                  : scenario.status === "Complete"
                    ? "bg-slate-950 border-slate-700"
                    : "bg-slate-950/50 border-slate-700"
              }`}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex-1">
                  <div className="font-mono text-sm font-bold text-neon-cyan">{scenario.phase}</div>
                  <div className="text-xs text-muted-foreground mt-1">Laps {scenario.laps}</div>
                  <div className="text-xs text-foreground mt-2">{scenario.strategy}</div>
                </div>
                <Badge
                  variant={scenario.status === "Active" ? "default" : "outline"}
                  className={scenario.status === "Active" ? "bg-neon-green text-black" : ""}
                >
                  {scenario.status}
                </Badge>
              </div>
            </div>
          ))}
        </div>

        {/* Quick Actions */}
        <div className="pt-4 border-t border-slate-700 space-y-2">
          <button className="w-full px-3 py-2 bg-neon-cyan text-black rounded-lg font-mono text-sm font-bold hover:bg-neon-cyan/90 transition-colors">
            Simulate Pit
          </button>
            {/* 'Load Scenario' button removed per request; keep other quick actions */}
            {/* Load Scenario button has been removed */}
        </div>
      </div>
    </Card>
  )
}
