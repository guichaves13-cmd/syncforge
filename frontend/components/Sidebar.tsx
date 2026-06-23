"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { LayoutDashboard, PlusCircle, Settings, Zap } from "lucide-react";

const NAV = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/create", label: "Create", icon: PlusCircle },
  { href: "/settings", label: "Settings", icon: Settings },
];

export default function Sidebar() {
  const path = usePathname();
  return (
    <aside className="w-56 border-r border-line bg-bg-soft min-h-screen flex flex-col">
      <Link href="/" className="flex items-center gap-2 px-5 h-14 border-b border-line font-semibold">
        <div className="w-7 h-7 rounded-md bg-accent grid place-items-center">
          <Zap size={16} className="text-white" />
        </div>
        SyncForge
      </Link>
      <nav className="p-2 flex-1">
        {NAV.map(({ href, label, icon: Icon }) => {
          const active = path?.startsWith(href);
          return (
            <Link key={href} href={href}
              className={`flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition-colors ${
                active ? "bg-accent-soft text-ink" : "text-ink-muted hover:text-ink hover:bg-bg-card"
              }`}>
              <Icon size={16} /> {label}
            </Link>
          );
        })}
      </nav>
      <div className="p-4 text-xs text-ink-subtle border-t border-line">
        <div>SyncForge v0.1</div>
        <div className="text-ink-subtle/70">Phase 2 — UI</div>
      </div>
    </aside>
  );
}
