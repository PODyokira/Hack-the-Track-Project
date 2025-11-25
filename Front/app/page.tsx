"use client"
import { useState, useEffect } from "react"
import { RaceStateHero } from "@/components/race-state-hero"
import { DrivingActionPanel } from "@/components/driving-action-panel"
import { StrategyTimeline } from "@/components/strategy-timeline"
import { TelemetryUpload } from "@/components/telemetry-upload"
import { TrackMetadata } from "@/components/track-metadata"
import { ComprehensiveAnalysis } from "@/components/comprehensive-analysis"
import { useDashboardData } from "@/hooks/use-dashboard-data"

export default function Dashboard() {
  const { data, loading, refresh } = useDashboardData()
  const [liveData, setLiveData] = useState<any | null>(null)
  const dashboardData = liveData || data

  // Debug: log live updates when receiving streamed dashboard data
    useEffect(() => {
      if (liveData) console.log("[liveData]", liveData)
    }, [liveData])

  return (
    <main className="min-h-screen bg-background text-foreground">
      {/* Header */}
      <header className="border-b border-border bg-background/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-2 h-2 bg-neon-green rounded-full animate-pulse" />
            <h1 className="text-2xl font-bold font-sans">GR Cup Strategy Engine</h1>
          </div>
          <div className="text-xs text-muted-foreground">Barber Motorsports Park • IL+RL Hybrid</div>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 py-8 space-y-8">
        {/* Hero Section */}
        <RaceStateHero data={dashboardData?.raceState} loading={loading} />

        {/* Grid Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column */}
          <div className="lg:col-span-2 space-y-8">
            <DrivingActionPanel
              data={dashboardData?.drivingAction}
              sectorData={dashboardData?.sectorData}
              throttleHistory={dashboardData?.throttleHistory}
              loading={loading}
            />
            <TelemetryUpload onRefresh={refresh} onLiveUpdate={setLiveData} />
          </div>

          {/* Right Column */}
          <div className="space-y-8">
            <TrackMetadata data={dashboardData?.trackMetadata} />
            <StrategyTimeline items={dashboardData?.strategyTimeline} />
          </div>
        </div>

        {/* Full Width Analysis */}
        <ComprehensiveAnalysis data={dashboardData?.analysisSummary} recommendations={dashboardData?.recommendations} loading={loading} />
      </div>
    </main>
  )
}
