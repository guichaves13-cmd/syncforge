/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    NEXT_PUBLIC_API: process.env.NEXT_PUBLIC_API || "http://localhost:8000",
  },
};
module.exports = nextConfig;
