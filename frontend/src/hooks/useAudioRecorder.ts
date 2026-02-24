import { useState, useRef, useCallback, useEffect } from 'react';
import { encodeWav } from '../lib/wavEncoder';

export type RecordingState = 'idle' | 'recording' | 'encoding' | 'done' | 'error';

export interface UseAudioRecorderReturn {
  state: RecordingState;
  statusMessage: string;
  elapsedSeconds: number;
  audioBlob: Blob | null;
  startRecording: () => Promise<void>;
  stopRecording: () => Promise<Blob | null>;
  cleanup: () => void;
}

const RECORDING_TIMEOUT_MS = 15000; // 15 second auto-stop

export function useAudioRecorder(): UseAudioRecorderReturn {
  const [state, setState] = useState<RecordingState>('idle');
  const [statusMessage, setStatusMessage] = useState('Ready to record');
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);

  const audioCtxRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const buffersRef = useRef<Float32Array[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const startTimeRef = useRef<number>(0);

  const startRecording = useCallback(async () => {
    try {
      setState('recording');
      setStatusMessage('Recording...');
      setElapsedSeconds(0);
      buffersRef.current = [];
      startTimeRef.current = Date.now();

      // Get audio stream
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;

      // Create AudioContext if not exists
      if (!audioCtxRef.current) {
        audioCtxRef.current = new (window.AudioContext ||
          (window as any).webkitAudioContext)();
      }

      const audioContext = audioCtxRef.current;
      const source = audioContext.createMediaStreamSource(stream);

      // Create ScriptProcessor
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.addEventListener('audioprocess', (event) => {
        const inputData = event.inputBuffer.getChannelData(0);
        const buffer = new Float32Array(inputData);
        buffersRef.current.push(buffer);
      });

      source.connect(processor);
      processor.connect(audioContext.destination);

      // Start timer
      timerRef.current = setInterval(() => {
        const elapsed = Math.floor((Date.now() - startTimeRef.current) / 1000);
        setElapsedSeconds(elapsed);
      }, 100);

      // Auto-stop after timeout
      timeoutRef.current = setTimeout(() => {
        setStatusMessage('Recording limit reached');
        stopRecording();
      }, RECORDING_TIMEOUT_MS);
    } catch (error) {
      setState('error');
      setStatusMessage(`Recording failed: ${(error as Error).message}`);
    }
  }, []);

  const stopRecording = useCallback(async (): Promise<Blob | null> => {
    try {
      setState('encoding');
      setStatusMessage('Encoding audio...');

      // Clear timers
      if (timerRef.current) clearInterval(timerRef.current);
      if (timeoutRef.current) clearTimeout(timeoutRef.current);

      // Stop recording
      if (processorRef.current) {
        processorRef.current.disconnect();
        processorRef.current = null;
      }

      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      }

      // Encode WAV
      if (buffersRef.current.length === 0) {
        setState('error');
        setStatusMessage('No audio recorded');
        return null;
      }

      const audioContext = audioCtxRef.current;
      if (!audioContext) {
        setState('error');
        setStatusMessage('Audio context not initialized');
        return null;
      }

      const wavBlob = encodeWav(buffersRef.current, audioContext.sampleRate);
      setAudioBlob(wavBlob);
      setState('done');
      setStatusMessage('Recording complete');

      return wavBlob;
    } catch (error) {
      setState('error');
      setStatusMessage(`Encoding failed: ${(error as Error).message}`);
      return null;
    }
  }, []);

  const cleanup = useCallback(() => {
    if (timerRef.current) clearInterval(timerRef.current);
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    if (processorRef.current) processorRef.current.disconnect();
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cleanup();
    };
  }, [cleanup]);

  return {
    state,
    statusMessage,
    elapsedSeconds,
    audioBlob,
    startRecording,
    stopRecording,
    cleanup,
  };
}
