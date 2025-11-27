import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  async rewrites() {
    return [
      {
        source: "/api/:path*", // 访问
        destination: "http://localhost:8000/:path*", // 代理到后端
      },
    ];
  },
};

export default nextConfig;
