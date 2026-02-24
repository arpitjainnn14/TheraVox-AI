import { useState, useEffect } from 'react';
import { useCamera } from '../hooks/useCamera';
import { analyzeFrame, saveScreenshot } from '../lib/api';
import HeroSection from '../components/shared/HeroSection';
import EmotionDisplay from '../components/shared/EmotionDisplay';

interface Face {
  x: number;
  y: number;
  w: number;
  h: number;
  emotion: string;
  confidence: number;
  emoji?: string;
  description?: string;
}

interface FaceResult extends Face {
  id: string;
}

export default function VisionPage() {
  const { videoRef, canvasRef, isInitialized, error, initCamera, captureFrame } = useCamera();

  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isContinuous, setIsContinuous] = useState(false);
  const [analysisInterval, setAnalysisInterval] = useState<ReturnType<typeof setInterval> | null>(null);
  const [faces, setFaces] = useState<FaceResult[]>([]);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  // Initialize camera on mount
  useEffect(() => {
    initCamera();
  }, [initCamera]);

  // Handle continuous analysis
  useEffect(() => {
    if (!isContinuous || !isInitialized) return;

    const interval = setInterval(async () => {
      const frameData = captureFrame();
      if (frameData) {
        try {
          const result = await analyzeFrame(frameData);
          if (result.faces && result.faces.length > 0) {
            const newFaces: FaceResult[] = (result.faces as Face[]).map((face, i) => ({
              ...face,
              id: `face-${i}-${Date.now()}`,
            }));
            setFaces(newFaces);
            setAnalysisError(null);
          }
        } catch (err) {
          setAnalysisError((err as Error).message);
        }
      }
    }, 1000);

    setAnalysisInterval(interval);

    return () => clearInterval(interval);
  }, [isContinuous, isInitialized, captureFrame]);

  const handleToggleContinuous = async () => {
    if (!isContinuous) {
      setIsContinuous(true);
      setAnalysisError(null);
    } else {
      setIsContinuous(false);
      if (analysisInterval) {
        clearInterval(analysisInterval);
        setAnalysisInterval(null);
      }
    }
  };

  const handleSnapFrame = async () => {
    const frameData = captureFrame();
    if (!frameData) {
      setAnalysisError('Failed to capture frame');
      return;
    }

    setIsAnalyzing(true);
    setAnalysisError(null);

    try {
      const result = await analyzeFrame(frameData);
      if (result.faces && result.faces.length > 0) {
        const newFaces: FaceResult[] = (result.faces as Face[]).map((face, i) => ({
          ...face,
          id: `face-${i}-${Date.now()}`,
        }));
        setFaces(newFaces);
      }
    } catch (err) {
      setAnalysisError((err as Error).message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleSaveScreenshot = async () => {
    const frameData = captureFrame();
    if (!frameData) {
      setAnalysisError('Failed to capture screenshot');
      return;
    }

    try {
      await saveScreenshot(frameData);
      setAnalysisError(null);
      alert('Screenshot saved successfully!');
    } catch (err) {
      setAnalysisError(`Failed to save screenshot: ${(err as Error).message}`);
    }
  };

  return (
    <>
      <HeroSection title="Vision Analysis" subtitle="Real-time emotion detection from your face" />

      <div className="grid" style={{ marginTop: '32px' }}>
      <div className="card" style={{ gridColumn: '1 / -1' }}>
          <div style={{ position: 'relative', marginBottom: '24px' }}>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              style={{
                width: '100%',
                borderRadius: '12px',
                backgroundColor: '#000',
                display: isInitialized ? 'block' : 'none',
              }}
            />
            <canvas ref={canvasRef} style={{ display: 'none' }} />

            {!isInitialized && error && (
              <div
                style={{
                  padding: '32px',
                  textAlign: 'center',
                  backgroundColor: '#fee2e2',
                  borderRadius: '12px',
                  color: '#991b1b',
                }}
              >
                <p>
                  <strong>Camera access denied:</strong> {error.message}
                </p>
                <p>Please allow camera access in your browser settings.</p>
              </div>
            )}

            {!isInitialized && !error && (
              <div
                style={{
                  padding: '32px',
                  textAlign: 'center',
                  backgroundColor: '#f0f0f0',
                  borderRadius: '12px',
                  color: '#666',
                }}
              >
                <p>Initializing camera...</p>
              </div>
            )}
          </div>

          <div
            id="analysisStatus"
            style={{
              textAlign: 'center',
              marginBottom: '16px',
              minHeight: '20px',
              color: isContinuous ? '#10b981' : '#666',
              fontWeight: isContinuous ? 'bold' : 'normal',
            }}
          >
            {isContinuous && 'Analyzing frames in real-time...'}
            {isAnalyzing && 'Analyzing frame...'}
          </div>

          <div
            style={{
              display: 'flex',
              gap: '12px',
              justifyContent: 'center',
              flexWrap: 'wrap',
              marginBottom: '24px',
            }}
          >
            <button
              id="startAnalysis"
              className={`btn ${isContinuous ? 'btn-danger' : 'btn-primary'}`}
              onClick={handleToggleContinuous}
              disabled={!isInitialized}
            >
              {isContinuous ? 'Stop Analysis' : 'Start Analysis'}
            </button>

            <button
              id="snap"
              className="btn btn-secondary"
              onClick={handleSnapFrame}
              disabled={!isInitialized || isAnalyzing}
            >
              {isAnalyzing ? 'Analyzing...' : 'Capture Frame'}
            </button>

            <button
              id="saveShot"
              className="btn btn-secondary"
              onClick={handleSaveScreenshot}
              disabled={!isInitialized}
            >
              Save Screenshot
            </button>
          </div>

          {analysisError && (
            <div
              style={{
                padding: '12px',
                backgroundColor: '#fee2e2',
                color: '#991b1b',
                borderRadius: '8px',
                marginBottom: '16px',
              }}
            >
              {analysisError}
            </div>
          )}
        </div>

        {faces.length > 0 && (
          <div id="faces" style={{ gridColumn: '1 / -1' }}>
            <h2>Detected Emotions</h2>
            <div className="grid">
              {faces.map((face) => (
                <div key={face.id} className="card">
                  <EmotionDisplay
                    emotion={face.emotion}
                    emoji={face.emoji || '😊'}
                    confidence={face.confidence}
                    description={face.description || 'Face detected'}
                  />
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="card" style={{ gridColumn: '1 / -1', marginTop: '32px' }}>
          <h3>Tips for Best Results</h3>
          <ul style={{ listStyle: 'none', padding: 0 }}>
            <li>🌟 Good lighting - Face should be well-lit</li>
            <li>📷 Center your face - Keep face centered in the frame</li>
            <li>⚡ Use continuous analysis - For real-time emotion tracking</li>
            <li>💾 Save screenshots - Keep records of your analysis sessions</li>
          </ul>
        </div>
      </div>
    </>
  );
}
