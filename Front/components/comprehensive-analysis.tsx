"use client"

import { Card } from "@/components/ui/card"
import { AnalysisSummary } from "@/hooks/use-dashboard-data"

type Props = {
  data?: AnalysisSummary
  recommendations?: string[]
  loading?: boolean
}

export function ComprehensiveAnalysis({ data, recommendations, loading }: Props) {
  console.debug("[ComprehensiveAnalysis] received data:", data)
  const summary =
    data ??
    ({
      ilRecommendation: "Maintain current racing line through Turn 6, increase throttle application in S2",
      rlStrategy: "PPO agent recommends conservative pit strategy due to current tire degradation curve",
      tireDegradation: "Soft compounds experiencing 2.3% performance loss per lap - pit window optimal at Lap 46",
      anomalies: "Detected 0.8s variance in Turn 8 apex speed vs. baseline - recommend brake pressure check",
      callouts: [
        "IL Model: Throttle optimization achieving +0.6% lap time improvement",
        "RL Agent: Aggression level should increase 15% in final stint",
        "Tire Analysis: Compound degradation tracking model predictions within 0.2%",
        "Track State: Surface grip reduced 3% at apex sectors after lap 35",
      ],
    } satisfies AnalysisSummary)

  return (
    <Card className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 border-neon-cyan/30">
      <div className="p-6 space-y-6">
        <h2 className="text-lg font-bold text-white font-mono">COMPREHENSIVE ANALYSIS</h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* IL Model */}
          <div className="space-y-2">
            <div className="text-xs font-mono text-neon-cyan uppercase">Imitation Learning Model</div>
            <p className="text-sm text-foreground leading-relaxed">
              {loading ? "Updating..." : summary.ilRecommendation}
            </p>
          </div>

          {/* RL Strategy */}
          <div className="space-y-2">
            <div className="text-xs font-mono text-neon-purple uppercase">PPO Reinforcement Learning</div>
            <p className="text-sm text-foreground leading-relaxed">{loading ? "Updating..." : summary.rlStrategy}</p>
          </div>

          {/* Tire Analysis */}
          <div className="space-y-2">
            <div className="text-xs font-mono text-neon-pink uppercase">Tire Degradation</div>
            <p className="text-sm text-foreground leading-relaxed">
              {loading ? "Updating..." : summary.tireDegradation}
            </p>
          </div>

          {/* Anomalies */}
          <div className="space-y-2">
            <div className="text-xs font-mono text-destructive uppercase">Anomalies Detected</div>
            <p className="text-sm text-foreground leading-relaxed">{loading ? "Updating..." : summary.anomalies}</p>
          </div>
        </div>

        {/* Callouts */}
        <div className="pt-6 border-t border-slate-700 space-y-3">
          <div className="text-xs font-mono text-neon-green uppercase">System Callouts</div>
          <div className="space-y-2">
            {(summary.callouts ?? recommendations ?? []).map((callout, i) => (
              <div key={i} className="flex gap-3 text-sm">
                <div className="w-1.5 h-1.5 rounded-full bg-neon-green mt-1.5 flex-shrink-0" />
                <p className="text-foreground leading-relaxed">{callout}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Action Buttons */}
        <div className="pt-6 border-t border-slate-700 flex gap-3">
          {/* Quick-action buttons removed per request; placeholder keeps layout stable */}
          <div className="flex-1" />
        </div>
      </div>
    </Card>
  )
}
