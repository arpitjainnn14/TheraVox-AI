import { useState, useMemo } from 'react';
import { analyzeText } from '../lib/api';
import HeroSection from '../components/shared/HeroSection';
import EmotionDisplay from '../components/shared/EmotionDisplay';
import EmotionSkeleton from '../components/shared/EmotionSkeleton';
import type { EmotionAnalysisResponse } from '../lib/api';

const EXAMPLE_TEXTS = [
  {
    label: 'Happy',
    text: 'I just got great news! I passed my exam and got the job I wanted. Life is amazing!',
  },
  {
    label: 'Sad',
    text: 'I miss my old friends. Everything feels lonely and empty these days.',
  },
  {
    label: 'Angry',
    text: 'This is absolutely ridiculous! I cannot believe they would do this to me!',
  },
  {
    label: 'Fearful',
    text: 'I am terrified about the upcoming presentation. What if I fail?',
  },
  {
    label: 'Calm',
    text: 'Everything is okay. I feel peaceful and content with where I am.',
  },
];

export default function TextPage() {
  const [text, setText] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<EmotionAnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const counts = useMemo(() => {
    return {
      words: text
        .trim()
        .split(/\s+/)
        .filter((w) => w.length > 0).length,
      characters: text.length,
    };
  }, [text]);

  const handleExample = (exampleText: string) => {
    setText(exampleText);
  };

  const handleAnalyze = async () => {
    if (!text.trim()) {
      setError('Please enter some text to analyze');
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setResult(null);

    try {
      const response = await analyzeText(text);
      setResult(response);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <>
      <HeroSection title="Text Analysis" subtitle="Analyze emotions in written text" />

      <div className="container">
        <div className="card" style={{ gridColumn: '1 / -1' }}>
          <div style={{ marginBottom: '24px' }}>
            <label style={{ display: 'block', marginBottom: '12px', fontWeight: '600' }}>
              Choose an example or write your own:
            </label>
            <select
              onChange={(e) => {
                if (e.target.value) {
                  handleExample(e.target.value);
                }
              }}
              defaultValue=""
              style={{
                width: '100%',
                padding: '12px',
                borderRadius: '8px',
                border: '1px solid #e5e0d8',
                fontSize: '16px',
                fontFamily: 'inherit',
              }}
            >
              <option value="">Select an example...</option>
              {EXAMPLE_TEXTS.map((example, i) => (
                <option key={i} value={example.text}>
                  {example.label}
                </option>
              ))}
            </select>
          </div>

          <div style={{ marginBottom: '16px', fontSize: '14px', color: '#6b665c' }}>
            {counts.words} words • {counts.characters} characters
          </div>

          <textarea
            id="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={8}
            placeholder="Enter text here to analyze emotions..."
            style={{
              width: '100%',
              padding: '16px',
              borderRadius: '8px',
              border: '1px solid #e5e0d8',
              fontSize: '16px',
              fontFamily: 'inherit',
              marginBottom: '24px',
              resize: 'vertical',
            }}
          />

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
            onClick={handleAnalyze}
            disabled={isAnalyzing || !text.trim()}
            className="btn btn-primary"
            style={{ width: '100%' }}
          >
            {isAnalyzing ? 'Analyzing...' : 'Analyze Text'}
          </button>
        </div>

        {isAnalyzing && (
          <div id="textResult" style={{ gridColumn: '1 / -1' }}>
            <h2>Analyzing...</h2>
            <EmotionSkeleton />
          </div>
        )}

        {result && (
          <div id="textResult" style={{ gridColumn: '1 / -1' }}>
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
          <h3>About Text Analysis</h3>
          <p>
            Our dual-engine text analysis uses transformer models with fallback lexicon-based
            analysis. Features include negation handling, intensity modifiers, emotion priority
            system, and confidence calibration for accurate emotion detection.
          </p>
        </div>
      </div>
    </>
  );
}
