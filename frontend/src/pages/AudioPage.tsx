import { useState, useRef } from 'react';
import { useAudioRecorder } from '../hooks/useAudioRecorder';
import { analyzeAudio } from '../lib/api';
import HeroSection from '../components/shared/HeroSection';
import EmotionDisplay from '../components/shared/EmotionDisplay';
import EmotionSkeleton from '../components/shared/EmotionSkeleton';
import type { EmotionAnalysisResponse } from '../lib/api';

export default function AudioPage() {
  const { state, statusMessage, elapsedSeconds, startRecording, stopRecording } =
    useAudioRecorder();

  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioPreviewRef = useRef<HTMLAudioElement>(null);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<EmotionAnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setError(null);
    setResult(null);

    if (audioPreviewRef.current) {
      const url = URL.createObjectURL(file);
      audioPreviewRef.current.src = url;
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.currentTarget.style.backgroundColor = '#f5f2ed';
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.currentTarget.style.backgroundColor = 'transparent';
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.currentTarget.style.backgroundColor = 'transparent';

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      if (file.type.startsWith('audio/')) {
        handleFileSelect(file);
      } else {
        setError('Please drop an audio file');
      }
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.currentTarget.files;
    if (files && files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  const handleAnalyzeFile = async () => {
    if (!selectedFile) {
      setError('Please select a file');
      return;
    }

    setIsAnalyzing(true);
    setError(null);

    try {
      const response = await analyzeAudio(selectedFile);
      setResult(response);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleStopRecording = async () => {
    const blob = await stopRecording();
    if (blob) {
      handleFileSelect(new File([blob], 'recording.wav', { type: 'audio/wav' }));
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <>
      <HeroSection title="Audio Analysis" subtitle="Detect emotions from voice and speech" />

      <div className="grid" style={{ marginTop: '32px' }}>
      <div className="card" style={{ gridColumn: '1 / -1' }}>
          <h3>Upload Audio File</h3>

          <div
            id="audioDropZone"
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            style={{
              padding: '48px',
              border: '2px dashed #d5cfc5',
              borderRadius: '12px',
              textAlign: 'center',
              cursor: 'pointer',
              marginBottom: '24px',
              transition: 'all 0.25s ease',
              backgroundColor: 'transparent',
            }}
          >
            <div style={{ fontSize: '48px', marginBottom: '16px' }}>🎙️</div>
            <p style={{ fontSize: '18px', marginBottom: '8px' }}>
              <strong>Drag & drop your audio file here</strong>
            </p>
            <p style={{ color: '#6b665c' }}>or click to browse</p>
            <input
              ref={fileInputRef}
              type="file"
              accept="audio/*"
              onChange={handleFileInputChange}
              style={{ display: 'none' }}
            />
          </div>

          {selectedFile && (
            <div style={{ marginBottom: '24px' }}>
              <p style={{ marginBottom: '8px', fontSize: '14px', color: '#6b665c' }}>
                <strong>Selected file:</strong> {selectedFile.name}
              </p>
              <audio
                ref={audioPreviewRef}
                controls
                style={{ width: '100%', marginBottom: '16px' }}
              />
            </div>
          )}

          {error && (
            <div
              style={{
                padding: '12px',
                backgroundColor: '#fee2e2',
                color: '#991b1b',
                borderRadius: '8px',
                marginBottom: '16px',
              }}
            >
              {error}
            </div>
          )}

          <button
            onClick={handleAnalyzeFile}
            disabled={isAnalyzing || !selectedFile}
            className="btn btn-primary"
            style={{ width: '100%', marginBottom: '24px' }}
          >
            {isAnalyzing ? 'Analyzing...' : 'Analyze Audio'}
          </button>

          <hr style={{ margin: '32px 0', border: 'none', borderTop: '1px solid #e5e0d8' }} />

          <h3>Record Audio</h3>

          {state === 'idle' && (
            <button
              onClick={startRecording}
              className="btn btn-primary"
              style={{ width: '100%' }}
            >
              🎤 Start Recording
            </button>
          )}

          {state === 'recording' && (
            <>
              <div
                style={{
                  textAlign: 'center',
                  marginBottom: '16px',
                  fontSize: '24px',
                  fontWeight: 'bold',
                  color: '#d97757',
                }}
              >
                {formatTime(elapsedSeconds)}
              </div>
              <button
                onClick={handleStopRecording}
                className="btn btn-danger"
                style={{ width: '100%' }}
              >
                ⏹️ Stop Recording
              </button>
            </>
          )}

          {(state === 'encoding' || state === 'done' || state === 'error') && (
            <p style={{ textAlign: 'center', color: '#6b665c' }}>{statusMessage}</p>
          )}
        </div>

        {isAnalyzing && (
          <div id="audioResult" style={{ gridColumn: '1 / -1' }}>
            <h2>Analyzing...</h2>
            <EmotionSkeleton />
          </div>
        )}

        {result && (
          <div id="audioResult" style={{ gridColumn: '1 / -1' }}>
            <h2>Result</h2>
            <div className="card">
              <EmotionDisplay
                emotion={result.emotion}
                emoji={result.emoji}
                confidence={result.confidence}
                description={result.description}
              />
            </div>
          </div>
        )}

        <div className="card" style={{ gridColumn: '1 / -1', marginTop: '32px' }}>
          <h3>Tips for Best Results</h3>
          <ul style={{ listStyle: 'none', padding: 0 }}>
            <li>📱 Clear recordings - Use a quiet environment</li>
            <li>⏱️ Keep it short - Shorter clips analyze faster</li>
            <li>🎤 Avoid background noise - Minimize room noise</li>
            <li>📍 Speak clearly - Enunciate for better accuracy</li>
          </ul>
        </div>
      </div>
    </>
  );
}
