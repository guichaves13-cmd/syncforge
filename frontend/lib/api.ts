export const API = process.env.NEXT_PUBLIC_API ?? "http://localhost:8000";

export type Job = {
  id: string;
  status: "queued" | "running" | "done" | "failed";
  progress: number;
  events: Array<Record<string, unknown>>;
  result?: { ok?: boolean; final?: string; solved?: number; total?: number };
};

export type CreateJobReq = {
  title: string;
  theme?: string;
  language?: string;
  voice?: string;
  target_sec?: number;
  mode?: "tts_only" | "avatar_overlay" | "avatar_full";
  enable_vision_verify?: boolean;
  enable_embeddings?: boolean;
  enable_generative_fallback?: boolean;
};

async function jsonFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    cache: "no-store",
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json() as Promise<T>;
}

export const api = {
  health: () => jsonFetch<{ ok: boolean; jobs: number }>(`${API}/api/health`),
  createJob: (req: CreateJobReq) =>
    jsonFetch<{ job_id: string; status: string }>(`${API}/api/jobs`, {
      method: "POST", body: JSON.stringify(req),
    }),
  getJob: (id: string) => jsonFetch<Job>(`${API}/api/jobs/${id}`),
  listJobs: () => jsonFetch<{ jobs: Array<Pick<Job, "id" | "status" | "progress">> }>(`${API}/api/jobs`),
};

export function subscribeJob(
  id: string,
  onEvent: (ev: Record<string, unknown>) => void,
  onClose?: () => void,
): () => void {
  const wsURL = API.replace(/^http/, "ws") + `/ws/jobs/${id}`;
  const ws = new WebSocket(wsURL);
  ws.onmessage = (e) => {
    try { onEvent(JSON.parse(e.data)); }
    catch { onEvent({ event: "raw", payload: e.data }); }
  };
  ws.onclose = () => onClose?.();
  return () => ws.close();
}
