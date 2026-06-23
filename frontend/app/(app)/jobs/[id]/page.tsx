"use client";
import Link from "next/link";
import { useParams } from "next/navigation";
import { ArrowLeft } from "lucide-react";
import JobTimeline from "@/components/JobTimeline";

export default function JobPage() {
  const { id } = useParams<{ id: string }>();
  return (
    <div className="p-8 max-w-5xl mx-auto">
      <Link href="/dashboard" className="btn-ghost mb-4">
        <ArrowLeft size={14} /> Back
      </Link>
      <header className="mb-6">
        <div className="text-xs uppercase tracking-wider text-ink-muted">Job</div>
        <h1 className="text-xl font-mono mt-0.5">{id}</h1>
      </header>
      <JobTimeline jobId={id} />
    </div>
  );
}
