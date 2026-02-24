import React, { useRef, useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { motion, useMotionValue, useSpring, useTransform } from 'framer-motion';
import type { Variants } from 'framer-motion';
// import HeroSection from '../components/shared/HeroSection'; // We will use a custom 3D hero below instead

// --- 3D Tilt Card Component ---
const TiltCard = ({ children, style, className }: { children: React.ReactNode, style?: React.CSSProperties, className?: string }) => {
  const ref = useRef<HTMLDivElement>(null);

  const x = useMotionValue(0);
  const y = useMotionValue(0);

  const mouseXSpring = useSpring(x, { stiffness: 400, damping: 30 });
  const mouseYSpring = useSpring(y, { stiffness: 400, damping: 30 });

  const rotateX = useTransform(mouseYSpring, [-0.5, 0.5], ["12deg", "-12deg"]);
  const rotateY = useTransform(mouseXSpring, [-0.5, 0.5], ["-12deg", "12deg"]);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!ref.current) return;
    const rect = ref.current.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    const xPct = mouseX / width - 0.5;
    const yPct = mouseY / height - 0.5;
    x.set(xPct);
    y.set(yPct);
  };

  const handleMouseLeave = () => {
    x.set(0);
    y.set(0);
  };

  return (
    <motion.div
      ref={ref}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      style={{
        rotateX,
        rotateY,
        transformStyle: "preserve-3d",
        perspective: 1200,
        ...style
      }}
      className={className}
    >
      <div style={{ transform: "translateZ(40px)", transformStyle: "preserve-3d", width: '100%', height: '100%' }}>
        {children}
      </div>
    </motion.div>
  );
};

// --- Animations ---
const containerVariants: Variants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.1, delayChildren: 0.1 },
  },
};

const itemVariants: Variants = {
  hidden: { opacity: 0, y: 30, scale: 0.95 },
  visible: { 
    opacity: 1, 
    y: 0, 
    scale: 1, 
    transition: { 
      duration: 0.7, 
      ease: "easeOut" 
    } 
  },
};

export default function HomePage() {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  
  // Track mouse for abstract background parallax
  useEffect(() => {
    const handleGlobalMouseMove = (e: MouseEvent) => {
      setMousePosition({
        x: (e.clientX / window.innerWidth - 0.5) * 40,
        y: (e.clientY / window.innerHeight - 0.5) * 40,
      });
    };
    window.addEventListener('mousemove', handleGlobalMouseMove);
    return () => window.removeEventListener('mousemove', handleGlobalMouseMove);
  }, []);

  return (
    <div style={{ overflowX: 'hidden' }}>
      
      {/* --- Custom 3D Hero Section --- */}
      <section style={{ 
        position: 'relative', 
        minHeight: '85vh', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        padding: '120px 24px 60px',
        overflow: 'hidden'
      }}>
        {/* Animated Background Blobs */}
        <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none', zIndex: -1 }}>
          <motion.div
            animate={{ 
              x: mousePosition.x * -1.5,
              y: mousePosition.y * -1.5,
              rotate: [0, 5, -5, 0]
            }}
            transition={{ type: 'spring', damping: 50, stiffness: 100, rotate: { repeat: Infinity, duration: 20, ease: "linear" } }}
            style={{
              position: 'absolute',
              top: '-10%',
              left: '-10%',
              width: '60vw',
              height: '60vw',
              background: 'radial-gradient(circle, rgba(217, 119, 87, 0.15) 0%, transparent 60%)',
              filter: 'blur(60px)',
              borderRadius: '50%'
            }}
          />
          <motion.div
            animate={{ 
              x: mousePosition.x * 1.2,
              y: mousePosition.y * 1.2,
              rotate: [0, -10, 10, 0]
            }}
            transition={{ type: 'spring', damping: 50, stiffness: 100, rotate: { repeat: Infinity, duration: 25, ease: "linear" } }}
            style={{
              position: 'absolute',
              bottom: '-20%',
              right: '-10%',
              width: '50vw',
              height: '50vw',
              background: 'radial-gradient(circle, rgba(122, 154, 140, 0.15) 0%, transparent 60%)',
              filter: 'blur(60px)',
              borderRadius: '50%'
            }}
          />
          <motion.div
            animate={{ 
              x: mousePosition.x * 0.8,
              y: mousePosition.y * -0.8
            }}
            transition={{ type: 'spring', damping: 50, stiffness: 100 }}
            style={{
              position: 'absolute',
              top: '30%',
              left: '50%',
              transform: 'translateX(-50%)',
              width: '40vw',
              height: '40vw',
              background: 'radial-gradient(circle, rgba(201, 169, 98, 0.12) 0%, transparent 60%)',
              filter: 'blur(50px)',
              borderRadius: '50%'
            }}
          />
        </div>

        {/* Hero Content */}
        <motion.div 
          className="container"
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          style={{ textAlign: 'center', position: 'relative', zIndex: 1, perspective: 1000 }}
        >
          <motion.div variants={itemVariants} style={{ display: 'inline-block', marginBottom: '24px' }}>
            <span style={{ 
              padding: '8px 16px', 
              borderRadius: '30px', 
              background: 'rgba(217, 119, 87, 0.1)', 
              color: 'var(--brand)', 
              fontWeight: '600', 
              fontSize: '14px',
              border: '1px solid rgba(217, 119, 87, 0.2)',
              boxShadow: '0 4px 12px rgba(217, 119, 87, 0.05)'
            }}>
              ✨ Welcome to the future of emotional intelligence
            </span>
          </motion.div>
          <motion.h1 
            variants={itemVariants} 
            style={{ 
              fontSize: 'clamp(42px, 8vw, 72px)', 
              fontWeight: '800', 
              lineHeight: 1.1, 
              marginBottom: '24px',
              letterSpacing: '-0.03em',
              background: 'linear-gradient(135deg, var(--text) 0%, var(--text-secondary) 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              textShadow: '0 10px 30px rgba(0,0,0,0.05)' // subtle 3d depth text
            }}
          >
            Understand Emotions,<br/> Transform Lives
          </motion.h1>
          <motion.p 
            variants={itemVariants} 
            style={{ 
              fontSize: 'clamp(18px, 2vw, 24px)', 
              color: 'var(--text-secondary)', 
              maxWidth: '700px', 
              margin: '0 auto 48px',
              lineHeight: 1.5 
            }}
          >
            TheraVox AI uses advanced multimodal emotion analysis to help you understand yourself better through Vision, Text, and Audio insights.
          </motion.p>
          <motion.div variants={itemVariants} style={{ display: 'flex', gap: '16px', justifyContent: 'center', flexWrap: 'wrap' }}>
            <motion.div whileHover={{ scale: 1.05, y: -2 }} whileTap={{ scale: 0.95 }}>
              <Link to="/register" style={{ 
                padding: '18px 36px', 
                background: 'linear-gradient(135deg, var(--brand), var(--accent-ochre))', 
                color: 'white', 
                borderRadius: '16px', 
                textDecoration: 'none', 
                fontWeight: '600', 
                fontSize: '18px', 
                display: 'inline-block',
                boxShadow: '0 8px 24px rgba(217, 119, 87, 0.3)',
                border: '1px solid rgba(255,255,255,0.2)'
              }}>
                Get Started Free
              </Link>
            </motion.div>
            <motion.div whileHover={{ scale: 1.05, y: -2 }} whileTap={{ scale: 0.95 }}>
              <Link to="/vision" style={{ 
                padding: '18px 36px', 
                background: 'rgba(255, 255, 255, 0.8)', 
                backdropFilter: 'blur(10px)',
                color: 'var(--text)', 
                borderRadius: '16px', 
                textDecoration: 'none', 
                fontWeight: '600', 
                fontSize: '18px', 
                display: 'inline-block',
                border: '1px solid var(--border)',
                boxShadow: '0 8px 24px rgba(0, 0, 0, 0.05)'
              }}>
                Try Live Demo
              </Link>
            </motion.div>
          </motion.div>
        </motion.div>
      </section>

      {/* --- Interactive 3D Features Section --- */}
      <motion.section
        className="container"
        variants={containerVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, margin: '0px 0px -15% 0px' }}
        style={{ padding: '80px 24px', perspective: 1000 }}
      >
        <motion.div style={{ textAlign: 'center', marginBottom: '80px' }} variants={itemVariants}>
          <h2 style={{ fontSize: 'clamp(32px, 5vw, 48px)', marginBottom: '16px', fontWeight: '800', letterSpacing: '-0.02em' }}>
            Three Powerful Dimensions
          </h2>
          <p style={{ fontSize: '18px', color: 'var(--text-secondary)', maxWidth: '600px', margin: '0 auto' }}>
            Hover over the cards to interact with our cutting-edge modalities.
          </p>
        </motion.div>

        <motion.div 
          variants={containerVariants} 
          style={{ 
            display: 'flex', 
            flexWrap: 'nowrap',
            gap: '32px', 
            overflowX: 'auto', 
            padding: '20px 24px 60px',
            margin: '0 -24px',
            scrollSnapType: 'x mandatory',
            WebkitOverflowScrolling: 'touch',
            scrollbarWidth: 'none',
            justifyContent: 'flex-start'
          }} 
          className="no-scrollbar"
        >
          <style>{`
            .no-scrollbar::-webkit-scrollbar { display: none; }
            @media (min-width: 1280px) {
              .no-scrollbar { justify-content: center !important; }
            }
          `}</style>
          
          {/* Vision Card */}
          <TiltCard style={{ scrollSnapAlign: 'center', flex: '0 0 auto', width: 'clamp(280px, 80vw, 380px)' }}>
            <motion.div
              variants={itemVariants}
              style={{
                background: 'linear-gradient(145deg, rgba(255,255,255,0.6), rgba(255,255,255,0.9))',
                backdropFilter: 'blur(20px)',
                border: '1px solid rgba(255,255,255,0.8)',
                borderRadius: '24px',
                padding: '40px 32px',
                height: '100%',
                boxShadow: '0 20px 40px rgba(0,0,0,0.06)',
                position: 'relative',
                overflow: 'hidden'
              }}
            >
              <div style={{ position: 'absolute', top: '-10%', right: '-10%', width: '150px', height: '150px', background: 'var(--brand-muted)', filter: 'blur(40px)', borderRadius: '50%', zIndex: 0 }} />
              
              <div style={{ position: 'relative', zIndex: 1, transform: 'translateZ(30px)' }}>
                <div style={{
                  fontSize: '48px',
                  marginBottom: '24px',
                  width: '80px',
                  height: '80px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  background: 'white',
                  borderRadius: '20px',
                  boxShadow: '0 10px 20px rgba(217, 119, 87, 0.15)',
                  border: '1px solid rgba(217, 119, 87, 0.2)'
                }}>👁️</div>
                <h3 style={{ fontSize: '24px', fontWeight: '700', marginBottom: '16px', color: 'var(--text)' }}>Vision Analysis</h3>
                <p style={{ color: 'var(--text-secondary)', lineHeight: '1.6', marginBottom: '24px' }}>
                  Real-time emotion detection from facial expressions using live webcam processing with unparalleled accuracy.
                </p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {['Live detection', 'Micro-expressions', 'Confidence metrics'].map(feature => (
                    <div key={feature} style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '15px', color: 'var(--text-secondary)' }}>
                      <span style={{ color: 'var(--brand)', fontWeight: 'bold' }}>✓</span> {feature}
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          </TiltCard>

          {/* Text Card */}
          <TiltCard style={{ scrollSnapAlign: 'center', flex: '0 0 auto', width: 'clamp(280px, 80vw, 380px)' }}>
            <motion.div
              variants={itemVariants}
              style={{
                background: 'linear-gradient(145deg, rgba(255,255,255,0.6), rgba(255,255,255,0.9))',
                backdropFilter: 'blur(20px)',
                border: '1px solid rgba(255,255,255,0.8)',
                borderRadius: '24px',
                padding: '40px 32px',
                height: '100%',
                boxShadow: '0 20px 40px rgba(0,0,0,0.06)',
                position: 'relative',
                overflow: 'hidden'
              }}
            >
              <div style={{ position: 'absolute', top: '-10%', right: '-10%', width: '150px', height: '150px', background: 'rgba(122, 154, 140, 0.15)', filter: 'blur(40px)', borderRadius: '50%', zIndex: 0 }} />
              
              <div style={{ position: 'relative', zIndex: 1, transform: 'translateZ(30px)' }}>
                <div style={{
                  fontSize: '48px',
                  marginBottom: '24px',
                  width: '80px',
                  height: '80px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  background: 'white',
                  borderRadius: '20px',
                  boxShadow: '0 10px 20px rgba(122, 154, 140, 0.15)',
                  border: '1px solid rgba(122, 154, 140, 0.2)'
                }}>📝</div>
                <h3 style={{ fontSize: '24px', fontWeight: '700', marginBottom: '16px', color: 'var(--text)' }}>Text Analysis</h3>
                <p style={{ color: 'var(--text-secondary)', lineHeight: '1.6', marginBottom: '24px' }}>
                  Advanced NLP sentiment analysis extracting nuanced emotions from written text with profound contextual awareness.
                </p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {['Context awareness', 'Subtext parsing', 'Tone mapping'].map(feature => (
                    <div key={feature} style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '15px', color: 'var(--text-secondary)' }}>
                      <span style={{ color: 'var(--accent-sage)', fontWeight: 'bold' }}>✓</span> {feature}
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          </TiltCard>

          {/* Audio Card */}
          <TiltCard style={{ scrollSnapAlign: 'center', flex: '0 0 auto', width: 'clamp(280px, 80vw, 380px)' }}>
            <motion.div
              variants={itemVariants}
              style={{
                background: 'linear-gradient(145deg, rgba(255,255,255,0.6), rgba(255,255,255,0.9))',
                backdropFilter: 'blur(20px)',
                border: '1px solid rgba(255,255,255,0.8)',
                borderRadius: '24px',
                padding: '40px 32px',
                height: '100%',
                boxShadow: '0 20px 40px rgba(0,0,0,0.06)',
                position: 'relative',
                overflow: 'hidden'
              }}
            >
              <div style={{ position: 'absolute', top: '-10%', right: '-10%', width: '150px', height: '150px', background: 'rgba(201, 169, 98, 0.15)', filter: 'blur(40px)', borderRadius: '50%', zIndex: 0 }} />
              
              <div style={{ position: 'relative', zIndex: 1, transform: 'translateZ(30px)' }}>
                <div style={{
                  fontSize: '48px',
                  marginBottom: '24px',
                  width: '80px',
                  height: '80px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  background: 'white',
                  borderRadius: '20px',
                  boxShadow: '0 10px 20px rgba(201, 169, 98, 0.15)',
                  border: '1px solid rgba(201, 169, 98, 0.2)'
                }}>🎤</div>
                <h3 style={{ fontSize: '24px', fontWeight: '700', marginBottom: '16px', color: 'var(--text)' }}>Audio Analysis</h3>
                <p style={{ color: 'var(--text-secondary)', lineHeight: '1.6', marginBottom: '24px' }}>
                  Deep voice emotion recognition analyzing speech patterns, prosody, and tone to decode spoken states.
                </p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {['Vocal prosody', 'Stress indicators', 'Real-time processing'].map(feature => (
                    <div key={feature} style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '15px', color: 'var(--text-secondary)' }}>
                      <span style={{ color: 'var(--accent-ochre)', fontWeight: 'bold' }}>✓</span> {feature}
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          </TiltCard>

        </motion.div>

        {/* Scroll hint / indicator */}
        <motion.div 
          variants={itemVariants} 
          style={{ 
            display: 'flex', 
            justifyContent: 'center', 
            gap: '8px', 
            marginTop: '-20px',
            opacity: 0.5
          }}
        >
          {[0, 1, 2].map(i => (
            <div key={i} style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--brand)', opacity: i === 0 ? 1 : 0.3 }} />
          ))}
        </motion.div>
      </motion.section>

      {/* --- Abstract Floating Elements Section --- */}
      <motion.section
        className="container"
        variants={containerVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, margin: '0px 0px -15% 0px' }}
        style={{ paddingTop: '80px', paddingBottom: '120px' }}
      >
        <div style={{
          background: 'linear-gradient(135deg, var(--surface) 0%, var(--surface-secondary) 100%)',
          borderRadius: '32px',
          padding: '80px 40px',
          border: '1px solid var(--border-subtle)',
          boxShadow: 'inset 0 0 40px rgba(255,255,255,0.5), 0 20px 60px rgba(0,0,0,0.03)',
          position: 'relative',
          overflow: 'hidden'
        }}>
          {/* Glass floating orbs inside */}
          <motion.div animate={{ y: [0, -20, 0], x: [0, 10, 0] }} transition={{ repeat: Infinity, duration: 6, ease: 'easeInOut' }} style={{ position: 'absolute', top: '10%', left: '10%', width: '100px', height: '100px', background: 'rgba(255,255,255,0.4)', borderRadius: '50%', backdropFilter: 'blur(10px)', border: '1px solid rgba(255,255,255,0.6)', boxShadow: '0 10px 30px rgba(0,0,0,0.05)' }} />
          <motion.div animate={{ y: [0, 20, 0], x: [0, -10, 0] }} transition={{ repeat: Infinity, duration: 8, ease: 'easeInOut' }} style={{ position: 'absolute', bottom: '10%', right: '15%', width: '150px', height: '150px', background: 'rgba(255,255,255,0.3)', borderRadius: '50%', backdropFilter: 'blur(12px)', border: '1px solid rgba(255,255,255,0.5)', boxShadow: '0 10px 30px rgba(0,0,0,0.05)' }} />

          <motion.div style={{ textAlign: 'center', position: 'relative', zIndex: 1 }} variants={itemVariants}>
            <h2 style={{ fontSize: 'clamp(28px, 5vw, 44px)', marginBottom: '24px', fontWeight: '800' }}>
              Why Choose TheraVox?
            </h2>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '32px', marginTop: '60px' }}>
              {[
                { icon: '🎯', title: 'Accurate Detection', desc: 'Industry-leading AI models.' },
                { icon: '🔒', title: 'Privacy First', desc: 'Your data stays encrypted and secure.' },
                { icon: '📊', title: 'Deep Analytics', desc: 'Track trends in your emotional well-being.' },
                { icon: '🚀', title: 'Fast & Efficient', desc: 'Real-time analysis with minimal latency.' },
              ].map((item, idx) => (
                <motion.div
                  key={idx}
                  variants={itemVariants}
                  whileHover={{ y: -8, scale: 1.02 }}
                  style={{
                    padding: '32px',
                    background: 'rgba(255, 255, 255, 0.7)',
                    backdropFilter: 'blur(20px)',
                    borderRadius: '24px',
                    border: '1px solid rgba(255,255,255,0.9)',
                    boxShadow: '0 10px 30px rgba(0,0,0,0.03)'
                  }}
                >
                  <div style={{ fontSize: '40px', marginBottom: '16px' }}>{item.icon}</div>
                  <h4 style={{ fontSize: '20px', fontWeight: '700', marginBottom: '12px', color: 'var(--text)' }}>{item.title}</h4>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '15px', margin: 0 }}>{item.desc}</p>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </motion.section>

      {/* --- Premium CTA Section --- */}
      <motion.section
        className="container"
        variants={containerVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, margin: '0px 0px -15% 0px' }}
        style={{
          paddingBottom: '120px',
        }}
      >
        <TiltCard>
          <div style={{
            position: 'relative',
            padding: '80px 40px',
            textAlign: 'center',
            background: 'linear-gradient(135deg, var(--brand), var(--brand-dark))',
            borderRadius: '32px',
            color: 'white',
            overflow: 'hidden',
            boxShadow: '0 30px 60px rgba(217, 119, 87, 0.25)',
            border: '1px solid rgba(255,255,255,0.1)'
          }}>
            {/* Background design elements */}
            <div style={{ position: 'absolute', top: '-50%', left: '-10%', width: '300px', height: '300px', background: 'rgba(255,255,255,0.1)', borderRadius: '50%', filter: 'blur(30px)' }} />
            <div style={{ position: 'absolute', bottom: '-50%', right: '-10%', width: '400px', height: '400px', background: 'rgba(201, 169, 98, 0.2)', borderRadius: '50%', filter: 'blur(40px)' }} />
            
            <div style={{ position: 'relative', zIndex: 1, transform: 'translateZ(40px)' }}>
              <motion.h2 variants={itemVariants} style={{ fontSize: 'clamp(32px, 5vw, 48px)', marginBottom: '24px', fontWeight: '800', letterSpacing: '-0.02em', textShadow: '0 4px 12px rgba(0,0,0,0.1)' }}>
                Embark on Your Wellness Journey
              </motion.h2>
              <motion.p variants={itemVariants} style={{ fontSize: '20px', marginBottom: '48px', maxWidth: '600px', margin: '0 auto 48px', opacity: 0.9 }}>
                Join thousands using TheraVox AI to decode their emotions and unlock better mental clarity.
              </motion.p>
              
              <motion.div variants={itemVariants} style={{ display: 'flex', gap: '20px', justifyContent: 'center', flexWrap: 'wrap' }}>
                <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                  <Link to="/register" style={{ 
                    padding: '18px 40px', 
                    background: 'white', 
                    color: 'var(--brand-dark)', 
                    borderRadius: '16px', 
                    textDecoration: 'none', 
                    fontWeight: '700', 
                    fontSize: '18px', 
                    boxShadow: '0 10px 20px rgba(0,0,0,0.1)',
                    display: 'inline-block'
                  }}>
                    Create Free Account
                  </Link>
                </motion.div>
              </motion.div>
            </div>
          </div>
        </TiltCard>
      </motion.section>

    </div>
  );
}
