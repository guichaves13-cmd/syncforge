"use client";
import { useEffect, useRef, useState } from "react";
import { subscribeJob } from "@/lib/api";
import { CheckCircle2, Circle, Loader2, AlertCircle } from "lucide-react";

type Ev = Record<string, unknown> & { event?: string; step?: string };

type StageState = "pending" | "active" | "done" | "failed";
type Stage = { key: string; label: string; state: StageState; subtext?: string };

const STAGES: Array<Pick<Stage, "key" | "label">> = [
  { key: "tts", label: "TTS narration" },
  { key: "build_engine", label: "SyncEngine init" },
  { key: "sync", label: "Sync (intent → retrieve → rank → verify)" },
  { key: "compose", label: "Compose video" },
  { key: "karaoke", label: "Burn karaoke subs" },
];

export default function JobTimeline({ jobId }: { jobId: string }) {
  const [stages, setStages] = useState<Stage[]>(
    () => STAGES.map((s) => ({ ...s, state: "pending" as StageState }))
  );
  const [clauseLog, setClauseLog] = useState<string[]>([]);
  const [final, setFinal] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const tail = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const close = subscribeJob(
      jobId,
      (ev: Ev) => {
        setConnected(true);
        const t = ev.event;
        if (t === "step") {
          setStages((prev) => prev.map((s) => {
            if (s.key !== ev.step) return s;
            const status = (ev as any).status as string | undefined;
            return { ...s, state: status === "done" ? "done" :
                                  status === "failed" ? "failed" : "active",
                          subtext: summarize(ev) };
          }));
        } else if (t === "clause_done") {
          const i = ev.i, total = ev.total;
          const source = (ev as any).source ?? "?";
          const score = (ev as any).score ?? 0;
          const ok = (ev as any).solved;
          setClauseLog((prev) => [
            ...prev,
            `[${i}/${total}] ${ok ? "✓" : "✗"} ${source} score=${score}`,
          ]);
        } else if (t === "done") {
          const r = (ev as any).result ?? ev;
          setFinal(r.final ?? null);
          setStages((prev) => prev.map((s) =>
            s.state === "active" || s.state === "pending" ? { ...s, state: "done" as StageState } : s
          ));
        } else if (t === "failed") {
          setError(String((ev as any).error ?? "unknown"));
        }
      },
      () => setConnected(false),
    );
    return close;
  }, [jobId]);

  useEffect(() => {
    tail.current?.scrollTo({ top: tail.current.scrollHeight });
  }, [clauseLog.length]);

  return (
    <div className="grid lg:grid-cols-2 gap-4">
      <div className="card">
        <h2 className="font-medium mb-4 flex items-center gap-2">
          Pipeline
          <span className={`pill ${connected ? "bg-ok/10 text-ok" : "bg-line text-ink-subtle"}`}>
            <span className={`w-1.5 h-1.5 rounded-full ${connected ? "bg-ok" : "bg-ink-subtle"}`} />
            {connected ? "live" : "connecting"}
          </span>
        </h2>
        <ol className="space-y-3">
          {stages.map((s) => <StageRow key={s.key} s={s} />)}
        </ol>
        {final && (
          <div className="mt-5 pt-4 border-t border-line">
            <div className="text-xs text-ink-muted mb-1">Final video</div>
            <div className="font-mono text-xs break-all bg-bg-soft p-2 rounded">{final}</div>
          </div>
        )}
        {error && (
          <div className="mt-5 pt-4 border-t border-line">
            <div className="flex items-center gap-2 text-err text-sm">
              <AlertCircle size={16} /> {error}
            </div>
          </div>
        )}
      </div>

      <div className="card">
        <h2 className="font-medium mb-3">Clause-by-clause</h2>
        <div ref={tail} className="font-mono text-xs space-y-1 max-h-96 overflow-y-auto">
          {clauseLog.length === 0
            ? <div className="text-ink-subtle">Waiting for sync stage…</div>
            : clauseLog.map((l, i) => (
                <div key={i} className={l.includes("✓") ? "text-ink" : "text-ink-muted"}>{l}</div>
              ))}
        </div>
      </div>
    </div>
  );
}

function StageRow({ s }: { s: Stage }) {
  const icon = s.state === "done"   ? <CheckCircle2 size={16} className="text-ok" />
             : s.state === "active" ? <Loader2 size={16} className="text-accent animate-spin" />
             : s.state === "failed" ? <AlertCircle size={16} className="text-err" />
             :                        <Circle size={16} className="text-ink-subtle" />;
  return (
    <li className="flex items-start gap-2.5">
      <div className="mt-0.5">{icon}</div>
      <div className="flex-1">
        <div className={`text-sm ${s.state === "pending" ? "text-ink-muted" : "text-ink"}`}>{s.label}</div>
        {s.subtext && <div className="text-xs text-ink-subtle">{s.subtext}</div>}
      </div>
    </li>
  );
}

function summarize(ev: Ev): string | undefined {
  const e = ev as any;
  if (e.sentences != null) return `${e.sentences} sentences detected`;
  if (e.clauses != null) return `${e.clauses} clauses queued`;
  if (e.solved != null && e.total != null) return `${e.solved}/${e.total} segments solved`;
  return undefined;
}
