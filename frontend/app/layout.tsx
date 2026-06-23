import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SyncForge",
  description: "Semantic narration-to-video sync, real footage, no junk.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-bg text-ink">{children}</body>
    </html>
  );
}
