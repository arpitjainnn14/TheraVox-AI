import { motion, useMotionValue, useTransform } from 'framer-motion';
import { useEffect } from 'react';

export default function BlobBackground() {
  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);

  // Transform mouse position to parallax effect
  const blob1X = useTransform(mouseX, [-1, 1], [-3, 3]);
  const blob1Y = useTransform(mouseY, [-1, 1], [-3, 3]);

  const blob2X = useTransform(mouseX, [-1, 1], [-6, 6]);
  const blob2Y = useTransform(mouseY, [-1, 1], [-6, 6]);

  const blob3X = useTransform(mouseX, [-1, 1], [-9, 9]);
  const blob3Y = useTransform(mouseY, [-1, 1], [-9, 9]);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      const xPercent = e.clientX / window.innerWidth - 0.5;
      const yPercent = e.clientY / window.innerHeight - 0.5;
      mouseX.set(xPercent);
      mouseY.set(yPercent);
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, [mouseX, mouseY]);

  return (
    <div className="hero__bg">
      <motion.span className="blob b1" style={{ x: blob1X, y: blob1Y }} />
      <motion.span className="blob b2" style={{ x: blob2X, y: blob2Y }} />
      <motion.span className="blob b3" style={{ x: blob3X, y: blob3Y }} />
    </div>
  );
}
