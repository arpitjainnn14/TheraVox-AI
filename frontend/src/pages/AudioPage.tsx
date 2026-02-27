import { useState, useRef } from 'react';
import { useAudioRecorder } from '../hooks/useAudioRecorder';
import { analyzeAudio } from '../lib/api';
import HeroSection from '../components/shared/HeroSection';
import EmotionDisplay from '../components/shared/EmotionDisplay';
import EmotionSkeleton from '../components/shared/EmotionSkeleton';
import EmotionPostcard from '../components/shared/EmotionPostcard';
import type { EmotionAnalysisResponse } from '../lib/api';

export default function AudioPage() {
  const { state, statusMessage, elapsedSeconds, startRecording, stopRecording } =
    useAudioRecorder();

  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioPreviewRef = useRef<HTMLAudioElement>(null);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
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
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);

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
      setError('Please select or record an audio file first');
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

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <>
      <HeroSection title="Audio Analysis" subtitle="Detect emotions from voice and speech" />

      <div className="container">
        {/* Two-column input grid */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))',
          gap: '24px',
          marginBottom: '0',
        }}>
          {/* Upload Card */}
          <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
            <div style={{ marginBottom: '20px' }}>
              <h3 style={{ margin: '0 0 6px', fontSize: '17px', fontWeight: '600' }}>Upload Audio File</h3>
              <p style={{ margin: 0, fontSize: '14px', color: 'var(--muted)' }}>
                Supports MP3, WAV, M4A, OGG and other audio formats
              </p>
            </div>

            {/* Drop zone */}
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              style={{
                flex: '1',
                padding: '40px 24px',
                border: `2px dashed ${isDragging ? 'var(--brand)' : '#e5e0d8'}`,
                borderRadius: '12px',
                textAlign: 'center',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                backgroundColor: isDragging ? 'var(--surface-secondary)' : 'transparent',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                minHeight: '160px',
                marginBottom: '16px',
              }}
            >
              <div style={{ fontSize: '36px', marginBottom: '12px' }}>🎙️</div>
              <p style={{ fontSize: '15px', fontWeight: '600', margin: '0 0 4px', color: 'var(--text)' }}>
                Drag & drop your audio file here
              </p>
              <p style={{ color: 'var(--muted)', margin: 0, fontSize: '13px' }}>
                or click to browse
              </p>
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                onChange={handleFileInputChange}
                style={{ display: 'none' }}
              />
            </div>

            {/* File preview */}
            {selectedFile && (
              <div style={{
                marginBottom: '16px',
                padding: '14px',
                backgroundColor: 'var(--surface-secondary)',
                borderRadius: '10px',
                border: '1px solid var(--border)',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px' }}>
                  <span style={{ fontSize: '20px' }}>🎵</span>
                  <div style={{ overflow: 'hidden' }}>
                    <p style={{ margin: 0, fontSize: '14px', fontWeight: '600', color: 'var(--text)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {selectedFile.name}
                    </p>
                    <p style={{ margin: 0, fontSize: '12px', color: 'var(--muted)' }}>
                      {formatFileSize(selectedFile.size)}
                    </p>
                  </div>
                </div>
                <audio ref={audioPreviewRef} controls style={{ width: '100%', height: '36px' }} />
              </div>
            )}

            {error && (
              <div style={{
                padding: '12px',
                backgroundColor: '#fee2e2',
                color: '#991b1b',
                borderRadius: '8px',
                marginBottom: '16px',
                fontSize: '14px',
              }}>
                {error}
              </div>
            )}

            <button
              onClick={handleAnalyzeFile}
              disabled={isAnalyzing || !selectedFile}
              className="btn btn-primary"
              style={{ width: '100%', marginTop: 'auto' }}
            >
              {isAnalyzing ? 'Analyzing...' : 'Analyze Audio'}
            </button>
          </div>

          {/* Record Card */}
          <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
            <div style={{ marginBottom: '20px' }}>
              <h3 style={{ margin: '0 0 6px', fontSize: '17px', fontWeight: '600' }}>Record Your Voice</h3>
              <p style={{ margin: 0, fontSize: '14px', color: 'var(--muted)' }}>
                Use your microphone to capture speech in real time
              </p>
            </div>

            {/* Record zone */}
            <div style={{
              flex: '1',
              padding: '40px 24px',
              border: `1px solid ${state === 'recording' ? 'var(--emotion-angry)' : '#e5e0d8'}`,
              borderRadius: '12px',
              backgroundColor: state === 'recording' ? '#fff5f5' : 'var(--surface-secondary)',
              textAlign: 'center',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              minHeight: '160px',
              marginBottom: '16px',
              transition: 'all 0.3s ease',
            }}>
              <div style={{ fontSize: '36px', marginBottom: '12px' }}>
                {state === 'recording' ? '⏺️' : '🎤'}
              </div>

              {state === 'idle' && (
                <>
                  <p style={{ fontSize: '15px', fontWeight: '600', margin: '0 0 16px', color: 'var(--text)' }}>
                    Tap to start recording
                  </p>
                  <button onClick={startRecording} className="btn btn-primary">
                    Start Recording
                  </button>
                </>
              )}

              {state === 'recording' && (
                <>
                  <p style={{ fontSize: '15px', fontWeight: '600', margin: '0 0 4px', color: 'var(--emotion-angry)' }}>
                    Recording in progress
                  </p>
                  <div style={{ fontSize: '28px', fontWeight: '700', color: 'var(--emotion-angry)', margin: '0 0 16px', fontVariantNumeric: 'tabular-nums' }}>
                    {formatTime(elapsedSeconds)}
                  </div>
                  <button
                    onClick={handleStopRecording}
                    style={{
                      background: 'var(--emotion-angry)',
                      color: 'white',
                      border: 'none',
                      padding: '12px 24px',
                      borderRadius: '8px',
                      cursor: 'pointer',
                      fontWeight: '600',
                      fontSize: '14px',
                    }}
                  >
                    ⏹ Stop Recording
                  </button>
                </>
              )}

              {(state === 'encoding' || state === 'done' || state === 'error') && (
                <p style={{ color: 'var(--muted)', margin: 0, fontSize: '14px' }}>{statusMessage}</p>
              )}
            </div>

            {/* Recording tips */}
            <div style={{
              padding: '14px',
              backgroundColor: 'var(--surface-tertiary)',
              borderRadius: '10px',
              border: '1px solid var(--border)',
              marginBottom: '16px',
            }}>
              <p style={{ margin: '0 0 8px', fontSize: '13px', fontWeight: '600', color: 'var(--text)' }}>
                Tips for best results
              </p>
              <ul style={{ margin: 0, padding: '0 0 0 16px', fontSize: '13px', color: 'var(--muted)', lineHeight: '1.7' }}>
                <li>Speak in a quiet environment</li>
                <li>Keep clips under 30 seconds</li>
                <li>Enunciate clearly and naturally</li>
              </ul>
            </div>

            <button
              onClick={handleAnalyzeFile}
              disabled={isAnalyzing || !selectedFile || state === 'recording'}
              className="btn btn-primary"
              style={{ width: '100%', marginTop: 'auto' }}
            >
              {isAnalyzing ? 'Analyzing...' : 'Analyze Recording'}
            </button>
          </div>
        </div>

        {/* Results */}
        {isAnalyzing && (
          <div id="audioResult" style={{ marginTop: '32px' }}>
            <h2 style={{ marginBottom: '16px' }}>Analyzing...</h2>
            <EmotionSkeleton />
          </div>
        )}

        {result && (
          <div id="audioResult" style={{ marginTop: '32px' }}>
            <h2 style={{ marginBottom: '16px' }}>Result</h2>
            <div className="card">
              <EmotionDisplay
                emotion={result.emotion}
                emoji={result.emoji}
                confidence={result.confidence}
                description={result.description}
              />
            </div>
            <div className="card" style={{ marginTop: '16px' }}>
              <EmotionPostcard
                emotion={result.emotion}
                emoji={result.emoji}
                confidence={result.confidence}
              />
            </div>
          </div>
        )}

        {/* About section */}
        <div className="card" style={{ marginTop: '32px' }}>
          <h3 style={{ marginTop: 0, marginBottom: '12px' }}>About Audio Analysis</h3>
          <p style={{ color: 'var(--text-secondary)', marginBottom: '16px', lineHeight: '1.6' }}>
            Our speech emotion recognition model analyzes vocal features — including tone, pitch, tempo, and rhythm — to detect emotional states from spoken audio.
          </p>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
            gap: '12px',
          }}>
            {[
              { icon: '🧠', label: 'Deep learning model', desc: 'Fine-tuned on diverse speech datasets' },
              { icon: '🎯', label: '8 emotion labels', desc: 'Angry, calm, happy, sad, fearful & more' },
              { icon: '⚡', label: 'Fast inference', desc: 'Results in seconds' },
              { icon: '🔒', label: 'Private', desc: 'Audio is not stored after analysis' },
            ].map(({ icon, label, desc }) => (
              <div key={label} style={{
                padding: '14px',
                backgroundColor: 'var(--surface-secondary)',
                borderRadius: '10px',
                border: '1px solid var(--border)',
              }}>
                <div style={{ fontSize: '22px', marginBottom: '6px' }}>{icon}</div>
                <p style={{ margin: '0 0 2px', fontSize: '13px', fontWeight: '600', color: 'var(--text)' }}>{label}</p>
                <p style={{ margin: 0, fontSize: '12px', color: 'var(--muted)' }}>{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </>
  );
}
