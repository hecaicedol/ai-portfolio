import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Self-Healing Multi-Agent Pipeline",
  description:
    "A document-processing pipeline that detects its own failures, reflects on past errors, and self-corrects.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
