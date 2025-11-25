import { useCallback, useEffect, useState } from "react"

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000"

export interface RaceState {
  lap: number
  position: number
  tireAge: number
  tireCompound: string
  predictedPitWindow: string
  aggressionLevel: string
  drsEligible: boolean
  fuelLevel: number
}

export interface DrivingActionData {
  throttlePercent: number
  steeringAngle: number
  brakePercent: number
  downforce: string
  confidence: number
}

export interface SectorData {
  [sector: string]: {
    current: number
    optimal: number
    delta: string
  }
}

export interface StrategyItem {
  phase: string
  laps: string
  strategy: string
  status: string
}

export interface AnalysisSummary {
  ilRecommendation?: string | null
  rlStrategy?: string | null
  tireDegradation?: string | null
  anomalies?: string | null
  callouts?: string[]
}

export interface DashboardData {
  raceState: RaceState
  drivingAction: DrivingActionData
  sectorData: SectorData
  throttleHistory: { lap: number; value: number }[]
  strategyTimeline: StrategyItem[]
  analysisSummary: AnalysisSummary
  recommendations: string[]
  trackMetadata: {
    name: string
    location: string
    length: string
    corners: number
    drsZones: number
    pitLaneTime: string
    elevation: string
    sectorLengths: Record<string, string>
  }
}

export function useDashboardData() {
  const [data, setData] = useState<DashboardData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchData = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`${API_BASE}/api/dashboard`)
      if (!response.ok) {
        throw new Error(`API error: ${response.status}`)
      }
      const payload = (await response.json()) as DashboardData
      setData(payload)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load dashboard data")
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchData()
  }, [fetchData])

  return { data, loading, error, refresh: fetchData }
}

