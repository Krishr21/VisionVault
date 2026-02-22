import React, { useMemo, useState } from 'react'

type IngestResponse = { video_id: string; status: string; chunks_indexed: number }

type SearchHit = {
  video_id: string
  start: number
  end: number
  score: number
  transcript: string
  caption: string
  thumbnail_path?: string
}

type SearchResponse = { query: string; hits: SearchHit[] }

type VideoMeta = {
  video_id: string
  source_type: 'youtube' | 'local'
  source: string
}

const API = 'http://localhost:8000'

function fmtTime(s: number) {
  if (!Number.isFinite(s)) return ''
  const sec = Math.max(0, Math.floor(s))
  const hh = Math.floor(sec / 3600)
  const mm = Math.floor((sec % 3600) / 60)
  const ss = sec % 60
  return hh > 0 ? `${hh}:${String(mm).padStart(2, '0')}:${String(ss).padStart(2, '0')}` : `${mm}:${String(ss).padStart(2, '0')}`
}

function classifyStatus(text: string): 'ok' | 'warn' | 'err' {
  const t = (text || '').toLowerCase()
  if (t.includes('failed') || t.includes('error') || t.includes('traceback') || t.includes('exception')) return 'err'
  if (t.includes('warning') || t.includes('required') || t.includes('set a video_id')) return 'warn'
  if (t.includes('done') || t.includes('got')) return 'ok'
  return 'warn'
}

function getThumbSrc(p?: string) {
  if (!p) return null
  // Backend now returns either an absolute URL or a relative API path like:
  //   /videos/{video_id}/frames/{frame_file}
  if (p.startsWith('http://') || p.startsWith('https://')) return p
  if (p.startsWith('/')) return `${API}${p}`
  return null
}

export default function App() {
  const [sourceType, setSourceType] = useState<'local' | 'youtube'>('youtube')
  const [source, setSource] = useState('')
  const [fps, setFps] = useState(1)
  const [maxFrames, setMaxFrames] = useState<number>(10)
  const [enableCaptions, setEnableCaptions] = useState<boolean>(false)

  const [videoId, setVideoId] = useState<string>('')
  const [query, setQuery] = useState('show me when the CEO discussed the budget')
  const [hits, setHits] = useState<SearchHit[]>([])
  const [busy, setBusy] = useState(false)
  const [log, setLog] = useState<string>('')
  const [videoMeta, setVideoMeta] = useState<VideoMeta | null>(null)
  const [ingestController, setIngestController] = useState<AbortController | null>(null)
  const [playerStart, setPlayerStart] = useState<number>(0)

  const statusClass = useMemo(() => classifyStatus(log), [log])

  async function ingest() {
    setBusy(true)
    setLog('Ingesting… (downloads + models can take a while on first run)')
    setHits([])
    try {
      const inferredType: 'local' | 'youtube' = source.trim().startsWith('http') ? 'youtube' : sourceType
      const controller = new AbortController()
      setIngestController(controller)
      const res = await fetch(`${API}/ingest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source_type: inferredType, source, fps, max_frames: maxFrames, enable_captions: enableCaptions }),
        signal: controller.signal
      })
      if (!res.ok) throw new Error(await res.text())
      const data = (await res.json()) as IngestResponse
      setVideoId(data.video_id)
  setVideoMeta(null)
      setLog(`Ingest done. video_id=${data.video_id} • chunks_indexed=${data.chunks_indexed}`)
    } catch (e: any) {
      if (e?.name === 'AbortError') {
        setLog('Ingest cancelled (request aborted)')
      } else {
        setLog(`Ingest failed: ${e?.message || String(e)}`)
      }
    } finally {
      setBusy(false)
      setIngestController(null)
    }
  }

  async function stopIngest() {
    // Cancel browser request immediately
    ingestController?.abort()
    setLog('Cancelling ingest…')

    // Best-effort: ask backend to stop too
    try {
      await fetch(`${API}/ingest/cancel`, { method: 'POST' })
    } catch {
      // ignore
    }
  }

  async function search() {
    if (!videoId) {
      setLog('Set a video_id (ingest first)')
      return
    }
    setBusy(true)
    setLog('Searching…')
    try {
      const res = await fetch(`${API}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_id: videoId, query, top_k: 5 })
      })
      if (!res.ok) throw new Error(await res.text())
      const data = (await res.json()) as SearchResponse
      setHits(data.hits)
      setLog(`Got ${data.hits.length} hits`)

      // Best-effort: fetch metadata so we can show "Open YouTube at this time".
      try {
        const metaRes = await fetch(`${API}/videos/${videoId}/meta`)
        if (metaRes.ok) {
          const m = (await metaRes.json()) as VideoMeta
          setVideoMeta(m)
        }
      } catch {
        // ignore
      }
    } catch (e: any) {
      setLog(`Search failed: ${e?.message || String(e)}`)
    } finally {
      setBusy(false)
    }
  }

  function youtubeLinkAt(seconds: number) {
    if (!videoMeta || videoMeta.source_type !== 'youtube') return null
    const url = new URL(videoMeta.source)
    // YouTube supports both t= and start= in different contexts; t= works well for watch URLs.
    url.searchParams.set('t', String(Math.max(0, Math.floor(seconds))))
    return url.toString()
  }

  function youtubeEmbedSrc(startSeconds: number) {
    if (!videoMeta || videoMeta.source_type !== 'youtube') return null
    const u = new URL(videoMeta.source)
    // Try to extract the video id from common youtube URL forms.
    let vid = u.searchParams.get('v')
    if (!vid && (u.hostname.includes('youtu.be'))) {
      vid = u.pathname.replace('/', '')
    }
    if (!vid && u.pathname.includes('/embed/')) {
      const parts = u.pathname.split('/embed/')
      vid = parts[1]?.split('/')[0]
    }
    if (!vid) return null

    const start = Math.max(0, Math.floor(startSeconds || 0))
    // enablejsapi=1 allows future enhancements; modestbranding reduces chrome.
    return `https://www.youtube.com/embed/${vid}?start=${start}&autoplay=1&rel=0&modestbranding=1`
  }

  function applyPreset(preset: 'fast' | 'balanced' | 'visual') {
    if (preset === 'fast') {
      setFps(0)
      setMaxFrames(0)
      setEnableCaptions(false)
      setLog('Preset: Fast (transcript-only)')
      return
    }
    if (preset === 'balanced') {
      setFps(1)
      setMaxFrames(10)
      setEnableCaptions(false)
      setLog('Preset: Balanced (a few frames, no captioning)')
      return
    }
    setFps(0.5)
    setMaxFrames(25)
    setEnableCaptions(true)
    setLog('Preset: Visual (caption frames — slower)')
  }

  return (
    <div className="container">
      <div className="header">
        <div className="brand">
          <div className="logo" />
          <div>
            <h1 className="h1">VisionVault</h1>
            <p className="subtitle">Ingest a video → semantic-search spoken + visual moments.</p>
          </div>
        </div>
        <div className="pill">
          <strong>{busy ? 'Working' : 'Ready'}</strong>
          <span>• API {API.replace('http://', '')}</span>
        </div>
      </div>

      <div className="grid">
        <section className="card">
          <h2>1) Ingest</h2>

          <div className="row" style={{ justifyContent: 'space-between' }}>
            <div className="row">
              <button className="btn secondary" disabled={busy} onClick={() => applyPreset('fast')}>Fast</button>
              <button className="btn secondary" disabled={busy} onClick={() => applyPreset('balanced')}>Balanced</button>
              <button className="btn secondary" disabled={busy} onClick={() => applyPreset('visual')}>Visual</button>
            </div>
            <div className="small">Start with <b>Fast</b> to get results quickly.</div>
          </div>

          <div className="row" style={{ marginTop: 10 }}>
            <div className="field" style={{ minWidth: 160 }}>
              <div className="label">Source</div>
              <select className="select" value={sourceType} onChange={(e) => setSourceType(e.target.value as any)}>
                <option value="youtube">YouTube URL</option>
                <option value="local">Local MP4 path</option>
              </select>
            </div>
            <div className="field" style={{ flex: 1, minWidth: 280 }}>
              <div className="label">URL / Path</div>
              <input
                className="input"
                placeholder={sourceType === 'youtube' ? 'https://www.youtube.com/watch?v=…' : '/absolute/path/to/video.mp4'}
                value={source}
                onChange={(e) => setSource(e.target.value)}
              />
            </div>
          </div>

          <div className="row" style={{ marginTop: 10 }}>
            <div className="field">
              <div className="label">FPS (0 = no frames)</div>
              <input className="input" style={{ width: 160 }} type="number" step="0.5" value={fps} onChange={(e) => setFps(Number(e.target.value))} />
            </div>
            <div className="field">
              <div className="label">Max frames (0 = none)</div>
              <input className="input" style={{ width: 170 }} type="number" value={maxFrames} onChange={(e) => setMaxFrames(Number(e.target.value))} />
            </div>
            <div className="field">
              <div className="label">Vision</div>
              <label className="pill" style={{ cursor: 'pointer' }}>
                <input type="checkbox" checked={enableCaptions} onChange={(e) => setEnableCaptions(e.target.checked)} />
                <span>Caption frames (slower)</span>
              </label>
            </div>
          </div>

          <div className="row" style={{ marginTop: 12, justifyContent: 'space-between' }}>
            <div className="field" style={{ flex: 1, minWidth: 280 }}>
              <div className="label">video_id (auto-filled after ingest)</div>
              <input
                className="input"
                value={videoId}
                onChange={(e) => setVideoId(e.target.value)}
                placeholder="(auto-filled after ingest)"
              />
              <div className="small">Tip: don’t type a custom video_id — use the generated one returned by ingest.</div>
            </div>

            <button className="btn success" disabled={busy || !source.trim()} onClick={ingest}>
              {busy ? 'Ingesting…' : 'Ingest'}
            </button>
            <button className="btn secondary" disabled={!busy} onClick={stopIngest}>
              Stop
            </button>
          </div>

          <div className={`status ${statusClass}`}>{log || 'Status will show here.'}</div>
        </section>

        <section className="card">
          <h2>2) Search</h2>

          {videoMeta?.source_type === 'youtube' ? (
            <div className="player" style={{ marginBottom: 12 }}>
              <div className="label">Player (YouTube)</div>
              <div className="playerFrame">
                <iframe
                  key={playerStart}
                  className="playerIframe"
                  src={youtubeEmbedSrc(playerStart) || undefined}
                  title="YouTube player"
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                  allowFullScreen
                />
              </div>
              <div className="small">Tip: click “Play here” on a hit to jump to that moment.</div>
            </div>
          ) : null}

          <div className="field">
            <div className="label">Query</div>
            <textarea className="textarea" value={query} onChange={(e) => setQuery(e.target.value)} />
            <div className="small">Try: “budget”, “product marketing strategy”, “CEO said”, “timeline”, etc.</div>
          </div>

          <div className="row" style={{ marginTop: 12, justifyContent: 'space-between' }}>
            <div className="pill"><strong>{hits.length}</strong> hits</div>
            <div className="row">
              <button className="btn secondary" disabled={busy} onClick={() => { setHits([]); setLog('Cleared results') }}>Clear</button>
              <button className="btn" disabled={busy || !videoId.trim()} onClick={search}>{busy ? 'Searching…' : 'Search'}</button>
            </div>
          </div>

          <ul className="hits">
            {hits.map((h, i) => {
              const thumb = getThumbSrc(h.thumbnail_path)
              const yt = youtubeLinkAt(h.start)
              return (
                <li key={i} className="hit">
                  <div className="hitTop">
                    <div className="time">{fmtTime(h.start)} → {fmtTime(h.end)}</div>
                    <div className="score">score {h.score.toFixed(3)} • {h.video_id}</div>
                  </div>

                  <div className="kv"><b>Transcript:</b> {h.transcript || <span className="small">(empty)</span>}</div>
                  <div className="kv"><b>Visual:</b> {h.caption || <span className="small">(captions disabled)</span>}</div>

                  {h.thumbnail_path ? (
                    <div className="thumb">
                      {thumb ? (
                        <img
                          src={thumb}
                          alt="thumbnail"
                          loading="lazy"
                          style={{ width: '100%', maxHeight: 220, objectFit: 'cover', borderRadius: 12, border: '1px solid rgba(255,255,255,0.08)' }}
                          onError={(e) => {
                            // Hide broken images without noisy UI.
                            ;(e.currentTarget as HTMLImageElement).style.display = 'none'
                          }}
                        />
                      ) : (
                        <div className="small">(thumbnail unavailable)</div>
                      )}
                      <div className="small">
                        Tip: we return the timestamp range; you can jump to that moment in the original video.
                        {yt ? (
                          <div style={{ marginTop: 6 }}>
                            <a className="pill" href={yt} target="_blank" rel="noreferrer">
                              Open YouTube @ {fmtTime(h.start)}
                            </a>
                            {videoMeta?.source_type === 'youtube' ? (
                              <button
                                className="pill"
                                style={{ marginLeft: 8, cursor: 'pointer' }}
                                onClick={() => setPlayerStart(h.start)}
                              >
                                Play here
                              </button>
                            ) : null}
                          </div>
                        ) : null}
                      </div>
                    </div>
                  ) : null}
                </li>
              )
            })}
          </ul>

          <div className="footer">First run downloads models. Use <b>Fast</b> preset to validate end-to-end quickly.</div>
        </section>
      </div>
    </div>
  )
}
