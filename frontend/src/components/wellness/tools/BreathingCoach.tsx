import { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import type { WellnessAction } from '../../../hooks/useWellnessStore';

const PATTERNS = {
  calm: {
    name: 'Calm (4-4)',
    phases: [
      { name: 'Inhale', duration: 4 },
      { name: 'Hold', duration: 4 },
    ],
    total: 8,
  },
  box: {
    name: 'Box (4-4-4-4)',
    phases: [
      { name: 'Inhale', duration: 4 },
      { name: 'Hold', duration: 4 },
      { name: 'Exhale', duration: 4 },
      { name: 'Hold', duration: 4 },
    ],
    total: 16,
  },
  '478': {
    name: '4-7-8 Sleep',
    phases: [
      { name: 'Inhale', duration: 4 },
      { name: 'Hold', duration: 7 },
      { name: 'Exhale', duration: 8 },
    ],
    total: 19,
  },
  energize: {
    name: 'Energize (2-2)',
    phases: [
      { name: 'Inhale', duration: 2 },
      { name: 'Hold', duration: 2 },
    ],
    total: 4,
  },
};

type PatternKey = keyof typeof PATTERNS;

interface BreathingCoachProps {
  dispatch: React.Dispatch<WellnessAction>;
}

export default function BreathingCoach({ dispatch }: BreathingCoachProps) {
  const [isRunning, setIsRunning] = useState(false);
  const [pattern, setPattern] = useState<PatternKey>('calm');
  const [sessionSeconds, setSessionSeconds] = useState(0);
  const [currentPhase, setCurrentPhase] = useState(0);
  const [phaseProgress, setPhaseProgress] = useState(0);

  const startTimeRef = useRef<number>(0);
  const phaseStartRef = useRef<number>(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const cycleCountRef = useRef(0);

  useEffect(() => {
    if (!isRunning) return;

    const patternData = PATTERNS[pattern];
    startTimeRef.current = Date.now();
    phaseStartRef.current = Date.now();
    cycleCountRef.current = 0;

    timerRef.current = setInterval(() => {
      const now = Date.now();
      const elapsed = Math.floor((now - startTimeRef.current) / 1000);
      setSessionSeconds(elapsed);

      let totalElapsedInPhases = 0;
      let phaseIndex = 0;
      let foundPhase = false;

      // Cycle through phases to find which one we're in
      for (let cycle = 0; cycle < 100; cycle++) {
        for (let i = 0; i < patternData.phases.length; i++) {
          const phaseDuration = patternData.phases[i].duration;
          if (totalElapsedInPhases + phaseDuration > elapsed) {
            phaseIndex = i;
            foundPhase = true;
            break;
          }
          totalElapsedInPhases += phaseDuration;
        }
        if (foundPhase) break;
        cycleCountRef.current = cycle + 1;
      }

      setCurrentPhase(phaseIndex);

      const phaseStart = totalElapsedInPhases;
      const phaseDuration = patternData.phases[phaseIndex].duration;
      const phaseElapsed = elapsed - phaseStart;
      const progress = Math.min(phaseElapsed / phaseDuration, 1);
      setPhaseProgress(progress);
    }, 100);

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isRunning, pattern]);

  const handleStart = () => {
    setIsRunning(true);
    setSessionSeconds(0);
    setCurrentPhase(0);
    setPhaseProgress(0);
  };

  const handleStop = () => {
    setIsRunning(false);
    if (timerRef.current) clearInterval(timerRef.current);

    // Record breathing minutes
    const minutes = Math.max(1, Math.round(sessionSeconds / 60));
    if (sessionSeconds >= 30) {
      dispatch({ type: 'ADD_BREATHING_MINUTES', payload: minutes });
    }
  };

  const patternData = PATTERNS[pattern];
  const currentPhaseName = patternData.phases[currentPhase].name;
  const scaleMin = currentPhaseName === 'Inhale' ? 0.9 : 1.1;
  const scaleMax = currentPhaseName === 'Inhale' ? 1.1 : 0.9;
  const scale = scaleMin + (scaleMax - scaleMin) * phaseProgress;

  return (
    <div className="card" style={{ gridColumn: '1 / -1' }}>
      <h2>Breathing Coach</h2>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', alignItems: 'center' }}>
        <div style={{ textAlign: 'left', paddingRight: '20px' }}>
          <motion.div
            id="breathCircle"
            animate={{ scale }}
            transition={{ type: 'tween', duration: 0.1 }}
            style={{
              width: '120px',
              height: '120px',
              borderRadius: '50%',
              backgroundColor: '#d97757',
              margin: '0 0 24px 0',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#fff',
              fontSize: '14px',
              fontWeight: 'bold',
            }}
          >
            {currentPhaseName}
          </motion.div>

          <div style={{ marginBottom: '24px' }}>
            <div
              style={{
                fontSize: '32px',
                fontWeight: 'bold',
                color: '#d97757',
                marginBottom: '8px',
              }}
            >
              {sessionSeconds}s
            </div>
            <div style={{ color: '#6b665c', fontSize: '14px' }}>
              {isRunning ? 'Breathing...' : 'Ready to start'}
            </div>
          </div>

          <div style={{ marginBottom: '24px' }}>
            <div
              style={{
                height: '4px',
                backgroundColor: '#e5e0d8',
                borderRadius: '2px',
                overflow: 'hidden',
                marginBottom: '8px',
              }}
            >
              <motion.div
                animate={{ width: `${phaseProgress * 100}%` }}
                transition={{ type: 'tween', duration: 0.1 }}
                style={{
                  height: '100%',
                  backgroundColor: '#d97757',
                }}
              />
            </div>
          </div>

          <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-start' }}>
            {!isRunning ? (
              <button onClick={handleStart} className="btn btn-primary">
                Start
              </button>
            ) : (
              <button onClick={handleStop} className="btn btn-danger">
                Stop
              </button>
            )}
          </div>
        </div>

        <div style={{ borderLeft: '1px solid var(--border)', paddingLeft: '24px' }}>
          <h3 style={{ marginTop: 0 }}>Choose Pattern</h3>
          {Object.entries(PATTERNS).map(([key, data]) => (
            <button
              key={key}
              onClick={() => {
                setPattern(key as PatternKey);
                setCurrentPhase(0);
                setPhaseProgress(0);
              }}
              disabled={isRunning}
              style={{
                display: 'block',
                width: '100%',
                padding: '12px',
                marginBottom: '8px',
                borderRadius: '8px',
                border: pattern === key ? '2px solid #d97757' : '1px solid #e5e0d8',
                backgroundColor: pattern === key ? '#fdf4f1' : '#fff',
                color: pattern === key ? '#d97757' : '#1d1b18',
                cursor: isRunning ? 'not-allowed' : 'pointer',
                fontWeight: pattern === key ? '600' : '400',
                fontSize: '14px',
                transition: 'all 0.25s',
              }}
            >
              {data.name}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
