"use client"

import { Card } from "@/components/ui/card"

type Props = {
  data?: {
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

export function TrackMetadata({ data }: Props) {
  const trackInfo =
    data ??
    {
      name: "Barber Motorsports Park",
      location: "Birmingham, AL",
      length: "2.28 miles",
      corners: 17,
      drsZones: 1,
      pitLaneTime: "34 seconds",
      elevation: "646ft @ SF",
      sectorLengths: { S1: "0.82", S2: "0.71", S3: "0.75" },
    }

  return (
    <Card className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 border-neon-cyan/30">
      <div className="p-6 space-y-6">
        <div>
          <h2 className="text-lg font-bold text-white font-mono">{trackInfo.name}</h2>
          <p className="text-sm text-muted-foreground mt-1">{trackInfo.location}</p>
        </div>

        <div className="space-y-3">
          <div className="flex justify-between items-center text-sm border-b border-slate-700 pb-2">
            <span className="font-mono text-neon-cyan">Track Length</span>
            <span className="text-white font-bold">{trackInfo.length}</span>
          </div>
          <div className="flex justify-between items-center text-sm border-b border-slate-700 pb-2">
            <span className="font-mono text-neon-cyan">Corners</span>
            <span className="text-white font-bold">{trackInfo.corners}</span>
          </div>
          <div className="flex justify-between items-center text-sm border-b border-slate-700 pb-2">
            <span className="font-mono text-neon-cyan">DRS Zones</span>
            <span className="text-white font-bold">{trackInfo.drsZones}</span>
          </div>
          <div className="flex justify-between items-center text-sm border-b border-slate-700 pb-2">
            <span className="font-mono text-neon-cyan">Pit Lane Time</span>
            <span className="text-white font-bold">{trackInfo.pitLaneTime}</span>
          </div>
          <div className="flex justify-between items-center text-sm">
            <span className="font-mono text-neon-cyan">Elevation Drop</span>
            <span className="text-white font-bold">{trackInfo.elevation}</span>
          </div>
        </div>

        <div className="pt-4 border-t border-slate-700">
          <div className="text-xs font-mono text-neon-cyan uppercase mb-3">Sector Breakdown</div>
          <div className="space-y-2">
            {Object.entries(trackInfo.sectorLengths).map(([sector, length]) => (
              <div key={sector} className="flex justify-between text-sm">
                <span className="text-muted-foreground">{sector}</span>
                <span className="text-neon-pink font-mono">{length} mi</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </Card>
  )
}
