import { motion } from 'framer-motion';
import ConfidenceBar from './ConfidenceBar';

interface EmotionDisplayProps {
  emotion: string;
  emoji: string;
  confidence: number;
  description: string;
}

export default function EmotionDisplay({
  emotion,
  emoji,
  confidence,
  description,
}: EmotionDisplayProps) {
  const emotionClass = emotion.toLowerCase();

  return (
    <motion.div
      className="emotion-result"
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4, ease: 'easeOut' }}
    >
      <div className={`emotion-badge ${emotionClass}`}>
        <span className="emotion-badge__emoji">{emoji}</span>
        <span className="emotion-badge__label">{emotion}</span>
      </div>

      <ConfidenceBar confidence={confidence} />

      <p className="emotion-description">{description}</p>
    </motion.div>
  );
}
