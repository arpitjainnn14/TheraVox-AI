#!/usr/bin/env python3
"""
Optimized startup script for TheraVox AI v2.0
Sets environment variables for better performance before starting the server.
"""

import os
import sys
import logging

# Set environment variables for optimization
os.environ["OPENCV_THREADS"] = "2"
os.environ["TORCH_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

# Disable warnings for cleaner output
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    """Start the optimized TheraVox server."""
    print("=" * 60)
    print("🚀 Starting TheraVox AI v2.0 (Refactored)")
    print("=" * 60)
    print("\n📊 Performance Settings:")
    print(f"   - OpenCV threads: {os.environ.get('OPENCV_THREADS', 'default')}")
    print(f"   - PyTorch threads: {os.environ.get('TORCH_NUM_THREADS', 'default')}")
    print("   - Lazy loading enabled for all AI models")
    print("   - Modular architecture with clean separation of concerns")
    print("\n✨ Features:")
    print("   - Text emotion analysis (Transformer-based)")
    print("   - Audio emotion recognition (Whisper SER)")
    print("   - Vision facial emotion detection (DeepFace)")
    print("\n💡 Startup Note:")
    print("   - Server will start quickly (models load on first use)")
    print("   - First request to each feature may take longer")
    print("   - Subsequent requests will be fast")
    print("\n" + "=" * 60 + "\n")
    print("⏳ Loading application...\n")
    
    try:
        import uvicorn
        print("✓ Uvicorn imported")
        
        print("📦 Loading FastAPI application...")
        from main import app
        print("✓ Application loaded\n")
        
        print("=" * 60)
        print("✅ Server starting on http://127.0.0.1:8000")
        print("=" * 60 + "\n")
        
        # Start the server with optimized settings
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            workers=1,
            loop="asyncio",
            access_log=False,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 TheraVox server stopped gracefully")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
