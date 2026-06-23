"use client";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { api, type CreateJobReq } from "@/lib/api";
import { Sparkles, Mic, Video, Loader2 } from "lucide-react";

const VOICES = [
  { v: "en-US-AndrewNeural", label: "Andrew (en, male)" },
  { v: "en-US-EmmaNeural", label: "Emma (en, female)" },
  { v: "en-US-AvaNeural", label: "Ava (en, female, soft)" },
  { v: "pt-BR-AntonioNeural", label: "Antônio (pt-BR, male)" },
  { v: "pt-BR-FranciscaNeural", label: "Francisca (pt-BR, female)" },
  { v: "es-ES-AlvaroNeural", label: "Alvaro (es, male)" },
];

type Mode = "tts_only" | "avatar_overlay" | "avatar_full";

export default function CreatePage() {
  const router = useRouter();
  const [form, setForm] = useState<CreateJobReq>({
    title: "", theme: "", language: "en", voice: "en-US-AndrewNeural",
    target_sec: 600, mode: "tts_only",
    enable_vision_verify: true, enable_embeddings: true,
    enable_generative_fallback: false,
  });
  const [submitting, setSubmitting] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  function set<K extends keyof CreateJobReq>(k: K, v: CreateJobReq[K]) {
    setForm((f) => ({ ...f, [k]: v }));
  }

  async function submit() {
    if (!form.title.trim()) { setErr("Title is required"); return; }
    setSubmitting(true); setErr(null);
    try {
      const r = await api.createJob(form);
      router.push(`/jobs/${r.job_id}`);
    } catch (e) {
      setErr(String(e));
      setSubmitting(false);
    }
  }

  return (
    <div className="p-8 max-w-3xl mx-auto">
      <header className="mb-8">
        <h1 className="text-2xl font-semibold">Create a video</h1>
        <p className="text-sm text-ink-muted mt-1">
          Title → narration → semantic b-roll → composed video.
        </p>
      </header>

      <div className="card space-y-5">
        {/* Mode picker */}
        <div>
          <label className="label">Mode</label>
          <div className="grid grid-cols-3 gap-2">
            <ModeChip selected={form.mode === "tts_only"} icon={<Mic size={14} />}
              label="Narration only" desc="TTS voice + b-roll"
              onClick={() => set("mode", "tts_only")} />
            <ModeChip selected={form.mode === "avatar_overlay"} icon={<Video size={14} />}
              label="Avatar corner" desc="Talking head + b-roll"
              onClick={() => set("mode", "avatar_overlay")} />
            <ModeChip selected={form.mode === "avatar_full"} icon={<Sparkles size={14} />}
              label="Avatar full" desc="Fullscreen avatar"
              onClick={() => set("mode", "avatar_full")} />
          </div>
        </div>

        {/* Title */}
        <div>
          <label className="label">Title</label>
          <input className="input" value={form.title}
            onChange={(e) => set("title", e.target.value)}
            placeholder="5 Table Tennis Brands ROBBING You Blind" />
        </div>

        {/* Theme + language + voice */}
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="label">Theme (visual keywords)</label>
            <input className="input" value={form.theme}
              onChange={(e) => set("theme", e.target.value)}
              placeholder="ping pong paddle blade rubber tournament" />
          </div>
          <div>
            <label className="label">Language</label>
            <select className="input" value={form.language}
              onChange={(e) => set("language", e.target.value)}>
              <option value="en">English</option>
              <option value="pt">Português (BR)</option>
              <option value="es">Español</option>
            </select>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="label">Voice</label>
            <select className="input" value={form.voice}
              onChange={(e) => set("voice", e.target.value)}>
              {VOICES.map((v) => (
                <option key={v.v} value={v.v}>{v.label}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="label">Target length</label>
            <select className="input" value={form.target_sec}
              onChange={(e) => set("target_sec", Number(e.target.value))}>
              <option value={60}>1 min</option>
              <option value={300}>5 min</option>
              <option value={600}>10 min</option>
              <option value={1020}>17 min</option>
              <option value={1800}>30 min</option>
            </select>
          </div>
        </div>

        {/* Feature flags */}
        <div>
          <label className="label">Quality features</label>
          <div className="flex flex-wrap gap-2">
            <Toggle on={form.enable_vision_verify!}
              onChange={(v) => set("enable_vision_verify", v)}
              label="Gemini Vision verify" />
            <Toggle on={form.enable_embeddings!}
              onChange={(v) => set("enable_embeddings", v)}
              label="Multimodal embeddings" />
            <Toggle on={form.enable_generative_fallback!}
              onChange={(v) => set("enable_generative_fallback", v)}
              label="Veo 3 generative fallback" />
          </div>
        </div>

        {err && <div className="text-sm text-err">{err}</div>}

        <div className="flex justify-end">
          <button onClick={submit} disabled={submitting} className="btn-primary disabled:opacity-50">
            {submitting && <Loader2 size={14} className="animate-spin" />}
            {submitting ? "Starting…" : "Start job"}
          </button>
        </div>
      </div>
    </div>
  );
}

function ModeChip({ selected, icon, label, desc, onClick }: {
  selected: boolean; icon: React.ReactNode; label: string; desc: string; onClick: () => void;
}) {
  return (
    <button onClick={onClick}
      className={`text-left p-3 rounded-lg border transition-colors ${
        selected ? "border-accent bg-accent-soft" : "border-line hover:border-ink-subtle"
      }`}>
      <div className="flex items-center gap-1.5 mb-1 text-accent">{icon}<span className="font-medium text-ink text-sm">{label}</span></div>
      <div className="text-xs text-ink-muted">{desc}</div>
    </button>
  );
}

function Toggle({ on, onChange, label }: {
  on: boolean; onChange: (v: boolean) => void; label: string;
}) {
  return (
    <button onClick={() => onChange(!on)}
      className={`pill border ${
        on ? "bg-accent-soft border-accent/40 text-ink" : "border-line text-ink-muted hover:text-ink"
      }`}>
      <span className={`w-1.5 h-1.5 rounded-full ${on ? "bg-accent" : "bg-ink-subtle"}`} />
      {label}
    </button>
  );
}
