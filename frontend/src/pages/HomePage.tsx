import React, { useRef } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { motion, useMotionValue, useSpring, useMotionTemplate } from 'framer-motion';

const TILT_LIMIT = 8; // Max degrees of rotation

function TiltCard({ children, style }: { children: React.ReactNode, style?: React.CSSProperties }) {
  const ref = useRef<HTMLDivElement>(null);
  
  const x = useMotionValue(0);
  const y = useMotionValue(0);
  
  const xSpring = useSpring(x, { stiffness: 400, damping: 40 });
  const ySpring = useSpring(y, { stiffness: 400, damping: 40 });
  
  const transform = useMotionTemplate`perspective(1000px) rotateX(${xSpring}deg) rotateY(${ySpring}deg)`;

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!ref.current) return;
    
    const rect = ref.current.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;
    
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    
    // Values from -0.5 to 0.5
    const xPct = mouseX / width - 0.5;
    const yPct = mouseY / height - 0.5;
    
    x.set(yPct * TILT_LIMIT * -1); // rotateX
    y.set(xPct * TILT_LIMIT); // rotateY
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
        transform,
        transformStyle: 'preserve-3d',
        background: 'var(--surface, #FFFFFF)',
        border: '1px solid var(--border, #E3E1DE)',
        borderRadius: '16px',
        padding: '32px',
        boxShadow: '0 4px 6px rgba(29, 27, 24, 0.02)',
        cursor: 'default',
        display: 'flex',
        flexDirection: 'column',
        position: 'relative',
        ...style
      }}
      whileHover={{ 
        boxShadow: '0 24px 48px rgba(29, 27, 24, 0.08)',
        borderColor: 'var(--border-strong, #D1CFC9)',
        scale: 1.02,
        zIndex: 10
      }}
      transition={{ type: 'spring', stiffness: 400, damping: 30 }}
      role="article"
      tabIndex={0}
      aria-labelledby="card-title"
    >
      <div style={{ transform: 'translateZ(40px)', flex: 1, display: 'flex', flexDirection: 'column' }}>
        {children}
      </div>
    </motion.div>
  );
}

// Stagger variants for smoother entry
const fadeUpContainer = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.15
    }
  }
};

const fadeUpItem = {
  hidden: { opacity: 0, y: 30 },
  show: { opacity: 1, y: 0, transition: { type: 'spring' as const, stiffness: 300, damping: 24 } }
};

export default function HomePage() {
  const { isAuthenticated } = useAuth();

  return (
    <motion.div 
      className="home-page" 
      style={{ maxWidth: '1000px', margin: '0 auto', padding: '64px 24px', fontFamily: 'var(--font-body, "Inter", sans-serif)' }}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.6 }}
    >
      {/* Hero Section */}
      <motion.section 
        style={{ marginBottom: '96px', marginTop: '60px' }}
        variants={fadeUpContainer}
        initial="hidden"
        animate="show"
        aria-label="Welcome Section"
      >
        <motion.h1 style={{ 
          fontSize: 'clamp(40px, 6vw, 64px)', 
          fontWeight: '400', 
          lineHeight: '1.1', 
          marginBottom: '24px',
          fontFamily: "'Charter', 'Georgia', serif",
          color: 'var(--text, #1D1D1B)',
          letterSpacing: '-0.02em',
          maxWidth: '850px'
        }} variants={fadeUpItem}>
          Understand Emotions, Transform Lives
        </motion.h1>
        <motion.p style={{ 
          fontSize: '22px', 
          color: 'var(--text-secondary, #4A4640)', 
          maxWidth: '750px', 
          marginBottom: '48px',
          lineHeight: '1.6' 
        }} variants={fadeUpItem}>
          TheraVox AI uses advanced multimodal emotion analysis to help you understand yourself better through Vision, Text, and Audio insights. Discover patterns in your mental well-being in real-time.
        </motion.p>
        <motion.div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }} variants={fadeUpItem}>
          {!isAuthenticated && (
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Link to="/register" style={{ 
                padding: '14px 28px', 
                background: 'var(--brand, #9EACCA)', 
                color: '#FFFFFF', 
                borderRadius: '8px', 
                textDecoration: 'none', 
                fontWeight: '500', 
                fontSize: '16px', 
                display: 'inline-block',
                transition: 'background 0.2s ease',
                border: 'none',
                boxShadow: '0 4px 12px rgba(158, 172, 202, 0.4)',
                cursor: 'pointer'
              }}
              onMouseOver={(e) => e.currentTarget.style.background = 'var(--brand-hover, #8A9ABC)'}
              onMouseOut={(e) => e.currentTarget.style.background = 'var(--brand, #9EACCA)'}
              aria-label="Get Started with TheraVox">
                Get Started
              </Link>
            </motion.div>
          )}
        </motion.div>
      </motion.section>

      {/* Features Section */}
      <motion.section 
        style={{ marginBottom: '100px' }}
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, margin: "-100px" }}
        variants={fadeUpContainer}
        aria-labelledby="features-heading"
      >
        <motion.h2 id="features-heading" style={{ fontSize: '36px', fontWeight: '400', marginBottom: '16px', fontFamily: "'Charter', 'Georgia', serif", letterSpacing: '-0.01em', color: 'var(--text, #1D1D1B)' }} variants={fadeUpItem}>
          Three Powerful Dimensions
        </motion.h2>
        <motion.p style={{ fontSize: '20px', color: 'var(--text-secondary, #4A4640)', marginBottom: '48px', maxWidth: '600px' }} variants={fadeUpItem}>
          Interact with our cutting-edge modalities. Accurate, secure, and fast.
        </motion.p>

        <motion.div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', 
          gap: '24px' 
        }} variants={fadeUpItem}>
          
          {/* Vision Card */}
          <TiltCard>
            <div style={{
              fontSize: '32px',
              marginBottom: '24px',
              width: '56px',
              height: '56px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: 'var(--surface-secondary, #FAF8F5)',
              borderRadius: '12px',
              border: '1px solid var(--border-subtle, #EAE6DF)',
              color: 'var(--text, #1D1D1B)',
              boxShadow: 'inset 0 1px 2px rgba(255,255,255,0.5), 0 2px 4px rgba(0,0,0,0.02)'
            }} aria-hidden="true">👁️</div>
            <h3 id="card-title-vision" style={{ fontSize: '22px', fontWeight: '600', marginBottom: '16px', color: 'var(--text)' }}>Vision Analysis</h3>
            <p style={{ color: 'var(--text-tertiary, #6E6E6A)', fontSize: '16px', lineHeight: '1.6', marginBottom: '32px' }}>
              Real-time emotion detection from facial expressions using live webcam processing.
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginTop: 'auto' }}>
              {['Live detection', 'Micro-expressions', 'Confidence metrics'].map(feature => (
                <div key={feature} style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '15px', color: 'var(--text-secondary)' }}>
                  <span style={{ color: 'var(--brand, #9EACCA)' }} aria-hidden="true">•</span> <span>{feature}</span>
                </div>
              ))}
            </div>
          </TiltCard>

          {/* Text Card */}
          <TiltCard>
            <div style={{
              fontSize: '32px',
              marginBottom: '24px',
              width: '56px',
              height: '56px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: 'var(--surface-secondary, #FAF8F5)',
              borderRadius: '12px',
              border: '1px solid var(--border-subtle, #EAE6DF)',
              color: 'var(--text, #1D1D1B)',
              boxShadow: 'inset 0 1px 2px rgba(255,255,255,0.5), 0 2px 4px rgba(0,0,0,0.02)'
            }} aria-hidden="true">📝</div>
            <h3 id="card-title-text" style={{ fontSize: '22px', fontWeight: '600', marginBottom: '16px', color: 'var(--text)' }}>Text Analysis</h3>
            <p style={{ color: 'var(--text-tertiary, #6E6E6A)', fontSize: '16px', lineHeight: '1.6', marginBottom: '32px' }}>
              Advanced NLP sentiment analysis extracting nuanced emotions from written text.
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginTop: 'auto' }}>
              {['Context awareness', 'Subtext parsing', 'Tone mapping'].map(feature => (
                <div key={feature} style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '15px', color: 'var(--text-secondary)' }}>
                  <span style={{ color: 'var(--brand, #9EACCA)' }} aria-hidden="true">•</span> <span>{feature}</span>
                </div>
              ))}
            </div>
          </TiltCard>

          {/* Audio Card */}
          <TiltCard>
            <div style={{
              fontSize: '32px',
              marginBottom: '24px',
              width: '56px',
              height: '56px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: 'var(--surface-secondary, #FAF8F5)',
              borderRadius: '12px',
              border: '1px solid var(--border-subtle, #EAE6DF)',
              color: 'var(--text, #1D1D1B)',
              boxShadow: 'inset 0 1px 2px rgba(255,255,255,0.5), 0 2px 4px rgba(0,0,0,0.02)'
            }} aria-hidden="true">🎤</div>
            <h3 id="card-title-audio" style={{ fontSize: '22px', fontWeight: '600', marginBottom: '16px', color: 'var(--text)' }}>Audio Analysis</h3>
            <p style={{ color: 'var(--text-tertiary, #6E6E6A)', fontSize: '16px', lineHeight: '1.6', marginBottom: '32px' }}>
              Deep voice emotion recognition analyzing speech patterns, prosody, and tone.
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginTop: 'auto' }}>
              {['Vocal prosody', 'Stress indicators', 'Real-time processing'].map(feature => (
                <div key={feature} style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '15px', color: 'var(--text-secondary)' }}>
                  <span style={{ color: 'var(--brand, #9EACCA)' }} aria-hidden="true">•</span> <span>{feature}</span>
                </div>
              ))}
            </div>
          </TiltCard>
        </motion.div>
      </motion.section>

      {/* Why Choose Section */}
      <motion.section 
        style={{ marginBottom: '100px' }}
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, margin: "-100px" }}
        variants={fadeUpContainer}
        aria-labelledby="why-choose-heading"
      >
        <motion.h2 id="why-choose-heading" style={{ fontSize: '36px', fontWeight: '400', marginBottom: '40px', fontFamily: "'Charter', 'Georgia', serif", letterSpacing: '-0.01em', color: 'var(--text, #1D1D1B)' }} variants={fadeUpItem}>
          Why Choose TheraVox?
        </motion.h2>
        <motion.div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '24px' }} variants={fadeUpContainer}>
          {[
            { icon: '🎯', title: 'Accurate Detection', desc: 'Industry-leading AI models.' },
            { icon: '🔒', title: 'Privacy First', desc: 'Your data stays completely secure.' },
            { icon: '📊', title: 'Deep Analytics', desc: 'Track trends in well-being.' },
            { icon: '🚀', title: 'Fast & Efficient', desc: 'Real-time analysis.' },
          ].map((item, idx) => (
            <motion.div key={idx} variants={fadeUpItem} whileHover={{ y: -8, scale: 1.02, transition: { type: 'spring', stiffness: 400, damping: 25 } }} style={{
              padding: '28px',
              background: 'var(--surface, #FFFFFF)',
              borderRadius: '16px',
              border: '1px solid var(--border, #EAE6DF)',
              boxShadow: '0 4px 12px rgba(29, 27, 24, 0.03)',
              cursor: 'default',
              display: 'flex',
              flexDirection: 'column'
            }}>
              <motion.div 
                style={{ fontSize: '28px', marginBottom: '20px', display: 'inline-block', transformOrigin: 'center' }}
                whileHover={{ rotate: [0, -10, 10, -10, 0], scale: 1.2, transition: { duration: 0.5 } }}
                aria-hidden="true"
              >
                {item.icon}
              </motion.div>
              <h4 style={{ fontSize: '18px', fontWeight: '600', marginBottom: '10px', color: 'var(--text)' }}>{item.title}</h4>
              <p style={{ color: 'var(--text-secondary, #4A4640)', fontSize: '15px', margin: 0, lineHeight: '1.6' }}>{item.desc}</p>
            </motion.div>
          ))}
        </motion.div>
      </motion.section>

      {/* CTA Section */}
      <motion.section 
        style={{ 
          padding: '80px 48px', 
          background: 'var(--surface-secondary, #FAF8F5)', 
          borderRadius: '24px',
          border: '1px solid var(--border, #E3E1DE)',
          textAlign: 'center',
          position: 'relative',
          overflow: 'hidden',
          boxShadow: '0 10px 30px rgba(0,0,0,0.02)'
        }}
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, margin: "-100px" }}
        variants={fadeUpContainer}
        aria-labelledby="cta-heading"
      >
        {/* Subtle background decoration for the CTA that looks high-end */}
        <motion.div 
          style={{
            position: 'absolute',
            top: '-50%',
            left: '-10%',
            width: '600px',
            height: '600px',
            background: 'radial-gradient(circle, var(--brand, #9EACCA) 0%, transparent 60%)',
            opacity: 0.08,
            zIndex: 0,
            pointerEvents: 'none',
            borderRadius: '50%'
          }} 
          animate={{ scale: [1, 1.05, 1], opacity: [0.08, 0.12, 0.08] }}
          transition={{ repeat: Infinity, duration: 8, ease: "easeInOut" }}
          aria-hidden="true" 
        />
        
        <div style={{ position: 'relative', zIndex: 1 }}>
          <motion.h2 id="cta-heading" style={{ fontSize: '40px', fontWeight: '400', marginBottom: '20px', fontFamily: "'Charter', 'Georgia', serif", letterSpacing: '-0.01em', color: 'var(--text, #1D1D1B)' }} variants={fadeUpItem}>
            Embark on Your Wellness Journey
          </motion.h2>
          <motion.p style={{ fontSize: '20px', color: 'var(--text-secondary, #4A4640)', marginBottom: '40px', maxWidth: '650px', margin: '0 auto 40px', lineHeight: '1.6' }} variants={fadeUpItem}>
            Join thousands using TheraVox AI to decode their emotions and unlock better mental clarity.
          </motion.p>
          {!isAuthenticated && (
            <div style={{ display: 'flex', gap: '16px', justifyContent: 'center' }}>
              <motion.div variants={fadeUpItem} whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                <Link to="/register" style={{ 
                  padding: '16px 36px', 
                  background: 'var(--brand, #9EACCA)', 
                  color: '#FFFFFF', 
                  borderRadius: '8px', 
                  textDecoration: 'none', 
                  fontWeight: '500', 
                  fontSize: '16px', 
                  display: 'inline-block',
                  transition: 'background 0.2s ease',
                  boxShadow: '0 8px 20px rgba(158, 172, 202, 0.3)',
                }}
                onMouseOver={(e) => e.currentTarget.style.background = 'var(--brand-hover, #8A9ABC)'}
                onMouseOut={(e) => e.currentTarget.style.background = 'var(--brand, #9EACCA)'}
                aria-label="Create an Account with TheraVox">
                  Create Account
                </Link>
              </motion.div>
              <motion.div variants={fadeUpItem} whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                <Link to="/text" style={{ 
                  padding: '16px 36px', 
                  background: 'transparent', 
                  color: 'var(--text, #1D1D1B)', 
                  border: '1px solid var(--border-strong, #D1CFC9)',
                  borderRadius: '8px', 
                  textDecoration: 'none', 
                  fontWeight: '500', 
                  fontSize: '16px', 
                  display: 'inline-block',
                  transition: 'all 0.2s ease',
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.background = 'var(--surface, #FFFFFF)';
                  e.currentTarget.style.borderColor = 'var(--brand, #9EACCA)';
                  e.currentTarget.style.color = 'var(--brand, #9EACCA)';
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.background = 'transparent';
                  e.currentTarget.style.borderColor = 'var(--border-strong, #D1CFC9)';
                  e.currentTarget.style.color = 'var(--text, #1D1D1B)';
                }}
                aria-label="Try Text Analysis Modality">
                  Get Started
                </Link>
              </motion.div>
            </div>
          )}
          {isAuthenticated && (
            <motion.div variants={fadeUpItem} whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Link to="/text" style={{ 
                padding: '16px 36px', 
                background: 'var(--brand, #9EACCA)', 
                color: '#FFFFFF', 
                borderRadius: '8px', 
                textDecoration: 'none', 
                fontWeight: '500', 
                fontSize: '16px', 
                display: 'inline-block',
                transition: 'background 0.2s ease',
                boxShadow: '0 8px 20px rgba(158, 172, 202, 0.3)',
              }}
              onMouseOver={(e) => e.currentTarget.style.background = 'var(--brand-hover, #8A9ABC)'}
              onMouseOut={(e) => e.currentTarget.style.background = 'var(--brand, #9EACCA)'}
              aria-label="Try Text Analysis Modality">
                Get Started
              </Link>
            </motion.div>
          )}
        </div>
      </motion.section>

    </motion.div>
  );
}
