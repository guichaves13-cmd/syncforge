import Link from "next/link";
import { Zap, Layers, Eye } from "lucide-react";

export default function Landing() {
  return (
    <div className="max-w-5xl mx-auto px-6 py-20">
      <header className="flex items-center justify-between mb-20">
        <div className="flex items-center gap-2 font-semibold">
          <div className="w-7 h-7 rounded-md bg-accent grid place-items-center">
            <Zap size={16} className="text-white" />
          </div>
          SyncForge
        </div>
        <nav className="flex gap-6 text-sm text-ink-muted">
          <a href="https://github.com/" target="_blank" rel="noreferrer"
             className="hover:text-ink">GitHub</a>
          <Link href="/dashboard" className="hover:text-ink">Open App</Link>
        </nav>
      </header>

      <section className="mb-16">
        <h1 className="text-5xl md:text-6xl font-semibold tracking-tight leading-[1.05]">
          Every clip you see <br/>
          <span className="text-accent">matches what you hear.</span>
        </h1>
        <p className="text-ink-muted mt-6 max-w-2xl text-lg">
          SyncForge generates videos where the b-roll is semantically aligned with the
          narration — frame by frame. Pulls from YouTube, Pexels, Pixabay, Coverr,
          Mixkit and Wikimedia. Verified by Gemini 2.5 Pro Vision.
        </p>
        <div className="flex gap-3 mt-8">
          <Link href="/create" className="btn-primary">Create a video</Link>
          <Link href="/dashboard" className="btn-ghost">View dashboard</Link>
        </div>
      </section>

      <section className="grid md:grid-cols-3 gap-4">
        <Feature icon={<Eye size={18} />} title="Vision verification"
          body="Gemini 2.5 Pro watches 8 frames of each candidate and rejects anachronism + off-topic." />
        <Feature icon={<Layers size={18} />} title="7 stock sources"
          body="Multi-source parallel search with anti-repeat via pHash + global cooldown." />
        <Feature icon={<Zap size={18} />} title="WebSocket live"
          body="Every stage publishes events. Watch your video build in real-time." />
      </section>
    </div>
  );
}

function Feature({ icon, title, body }: { icon: React.ReactNode; title: string; body: string }) {
  return (
    <div className="card">
      <div className="flex items-center gap-2 mb-2 text-accent">{icon}<span className="font-medium text-ink">{title}</span></div>
      <p className="text-sm text-ink-muted">{body}</p>
    </div>
  );
}
