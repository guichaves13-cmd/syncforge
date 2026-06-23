"use client";
import { useEffect, useState } from "react";
import { Key, Save, Check } from "lucide-react";

type Secrets = Record<string, string>;

const KEYS: Array<{ k: keyof Secrets; label: string; help: string }> = [
  { k: "GEMINI_API_KEY",     label: "Gemini",     help: "Vision verify + intent extraction + embeddings." },
  { k: "GROQ_API_KEY",       label: "Groq",       help: "Fastest LLM tier (Llama 3.3 70B)." },
  { k: "CEREBRAS_API_KEY",   label: "Cerebras",   help: "LLM fallback." },
  { k: "OPENROUTER_API_KEY", label: "OpenRouter", help: "Free Llama 3.3 fallback." },
  { k: "DEEPSEEK_API_KEY",   label: "DeepSeek",   help: "Cheap LLM fallback." },
  { k: "PEXELS_API_KEY",     label: "Pexels",     help: "Free HD videos + photos." },
  { k: "PIXABAY_API_KEY",    label: "Pixabay",    help: "Free HD videos." },
  { k: "COVERR_API_KEY",     label: "Coverr",     help: "Optional cinemagraphs (or scrape)." },
  { k: "ELEVENLABS_API_KEY", label: "ElevenLabs", help: "Premium voice (optional)." },
];

const STORAGE_KEY = "syncforge.secrets";

export default function SettingsPage() {
  const [secrets, setSecrets] = useState<Secrets>({});
  const [savedAt, setSavedAt] = useState<number | null>(null);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) setSecrets(JSON.parse(raw));
    } catch {}
  }, []);

  function save() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(secrets));
    setSavedAt(Date.now());
    setTimeout(() => setSavedAt(null), 2000);
  }

  return (
    <div className="p-8 max-w-3xl mx-auto">
      <header className="mb-8">
        <h1 className="text-2xl font-semibold">Settings</h1>
        <p className="text-sm text-ink-muted mt-1">
          API keys are stored in your browser's localStorage. The backend still reads
          from <code className="text-accent">.env</code>.
        </p>
      </header>

      <div className="card space-y-4">
        <div className="flex items-center gap-2 text-ink-muted text-sm">
          <Key size={14} /> Provider keys
        </div>
        {KEYS.map(({ k, label, help }) => (
          <div key={k}>
            <label className="label flex items-center gap-2">
              {label}
              {secrets[k] && <span className="text-ok text-[10px]">●set</span>}
            </label>
            <input className="input font-mono" type="password"
              placeholder={help} value={secrets[k] ?? ""}
              onChange={(e) => setSecrets({ ...secrets, [k]: e.target.value })} />
          </div>
        ))}

        <div className="flex justify-end pt-3">
          <button onClick={save} className="btn-primary">
            {savedAt ? <Check size={14} /> : <Save size={14} />}
            {savedAt ? "Saved" : "Save"}
          </button>
        </div>
      </div>
    </div>
  );
}
