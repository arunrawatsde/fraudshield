// The browser only talks to this Next.js server. Requests to /api/* are proxied to the FastAPI
// backend, so there is no CORS setup and the backend URL never reaches the browser.
// Rewrites are fixed at build time: in Docker, API_URL is passed as a build arg.
const API_URL = process.env.API_URL || "http://localhost:8000";

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  experimental: { proxyTimeout: 300_000 }, // an investigation can take a while
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_URL}/:path*` }];
  },
};

export default nextConfig;
