"""
One-command startup script for Apex3D Backend.
Usage:
    python start_server.py
    python start_server.py --port 8080 --no-reload
"""

import sys
import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Apex3D Backend Server Runner")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true", default=True, help="Enable hot reload")
    parser.add_argument("--no-reload", action="store_false", dest="reload", help="Disable hot reload")

    args = parser.parse_args()

    print("=" * 60)
    print("  Apex3D - Intelligent Vehicle Recognition & 3D Reconstruction")
    print(f"  Starting server on http://{args.host}:{args.port}")
    print(f"  API Docs (Swagger UI): http://localhost:{args.port}/docs")
    print("=" * 60)

    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()

