"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { api, type Job } from "@/lib/api";
import { ArrowRight, CheckCircle2, Loader2, XCircle, Clock } from "lucide-react";

type JobBrief = Pick<Job, "id" | "status" | "progress">;

export default function Dashboard() {
  const [jobs, setJobs] = useState<JobBrief[]>([]);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    const load = () =>
      api.listJobs().then((d) => { setJobs(d.jobs); setErr(null); })
        .catch((e) => setErr(String(e)));
    load();
    const t = setInterval(load, 2000);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="p-8 max-w-5xl mx-auto">
      <header className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-semibold">Dashboard</h1>
          <p className="text-sm text-ink-muted mt-1">Recent jobs and their live status.</p>
        </div>
        <Link href="/create" className="btn-primary">New job</Link>
      </header>

      {err && <div className="card border-err/30 text-err text-sm mb-4">{err}</div>}

      {jobs.length === 0 ? (
        <div className="card text-center py-16 text-ink-muted">
          <Clock className="mx-auto mb-3" size={24} />
          No jobs yet. <Link href="/create" className="text-accent">Create one →</Link>
        </div>
      ) : (
        <div className="grid gap-2">
          {jobs.map((j) => <JobRow key={j.id} job={j} />)}
        </div>
      )}
    </div>
  );
}

function JobRow({ job }: { job: JobBrief }) {
  const icon = job.status === "done"
    ? <CheckCircle2 size={16} className="text-ok" />
    : job.status === "failed"
    ? <XCircle size={16} className="text-err" />
    : <Loader2 size={16} className="text-accent animate-spin" />;
  return (
    <Link href={`/jobs/${job.id}`}
      className="card flex items-center justify-between hover:border-accent/40 transition-colors group">
      <div className="flex items-center gap-3">
        {icon}
        <div>
          <div className="font-mono text-sm">{job.id}</div>
          <div className="text-xs text-ink-muted capitalize">{job.status} · {Math.round(job.progress * 100)}%</div>
        </div>
      </div>
      <ArrowRight size={16} className="text-ink-subtle group-hover:text-accent" />
    </Link>
  );
}
