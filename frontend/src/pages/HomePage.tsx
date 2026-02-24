import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import HeroSection from '../components/shared/HeroSection';

export default function HomePage() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.1, delayChildren: 0.2 },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.6, ease: 'easeOut' } },
  };

  const featureCardVariants = {
    hidden: { opacity: 0, y: 30 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.5 } },
    hover: { y: -8, transition: { duration: 0.3 } },
  };

  return (
    <>
      {/* Hero Section */}
      <HeroSection
        title="Understand Emotions, Transform Lives"
        subtitle="TheraVox AI uses advanced multimodal emotion analysis to help you understand yourself better through Vision, Text, and Audio insights"
      />

      {/* Features Section */}
      <motion.section
        className="container"
        variants={containerVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, margin: '0px 0px -15% 0px' }}
        style={{ paddingTop: '60px', paddingBottom: '80px' }}
      >
        <motion.div style={{ textAlign: 'center', marginBottom: '64px' }} variants={itemVariants}>
          <h2 style={{ fontSize: 'clamp(28px, 5vw, 44px)', marginBottom: '16px', fontWeight: '700' }}>
            Three Powerful Ways to Analyze Emotions
          </h2>
          <p style={{ fontSize: '18px', color: 'var(--text-secondary)', maxWidth: '600px', margin: '0 auto' }}>
            Comprehensive emotion detection across multiple modalities for deeper insights
          </p>
        </motion.div>

        <motion.div className="grid" variants={containerVariants} style={{ marginBottom: '80px' }}>
          {/* Vision Card */}
          <motion.div
            className="card"
            variants={featureCardVariants}
            whileHover="hover"
            style={{
              background: 'linear-gradient(135deg, var(--surface), var(--brand-lighter))',
              border: '1px solid var(--border)',
              cursor: 'pointer',
              overflow: 'hidden',
              position: 'relative'
            }}
          >
            <div style={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              background: 'radial-gradient(circle at 20% 50%, rgba(217, 119, 87, 0.1), transparent 50%)',
              opacity: 0,
              transition: 'opacity 0.3s ease',
              pointerEvents: 'none'
            }} className="feature-card__bg" />
            <div style={{ position: 'relative', zIndex: 1 }}>
              <div style={{
                fontSize: '56px',
                marginBottom: '20px',
                width: '80px',
                height: '80px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: 'linear-gradient(135deg, #FDF4F1, #FAF8F5)',
                borderRadius: '16px',
                border: '2px solid rgba(217, 119, 87, 0.2)'
              }}>
                👁️
              </div>
              <h3 style={{ fontSize: '22px', fontWeight: '700', marginBottom: '12px', color: 'var(--text)' }}>Vision Analysis</h3>
              <p style={{ color: 'var(--text-secondary)', lineHeight: '1.6', marginBottom: '20px' }}>
                Real-time emotion detection from facial expressions and micro-expressions with AI-powered accuracy
              </p>
              <ul style={{ listStyle: 'none', padding: 0, display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <li style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-secondary)', fontSize: '14px' }}>✨ Live webcam detection</li>
                <li style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-secondary)', fontSize: '14px' }}>✨ Confidence metrics</li>
                <li style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-secondary)', fontSize: '14px' }}>✨ Frame analysis</li>
              </ul>
            </div>
          </motion.div>

          {/* Text Card */}
          <motion.div
            className="card"
            variants={featureCardVariants}
            whileHover="hover"
            style={{
              background: 'linear-gradient(135deg, var(--surface), var(--accent-sage-light))',
              border: '1px solid var(--border)',
              cursor: 'pointer',
              overflow: 'hidden',
              position: 'relative'
            }}
          >
            <div style={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              background: 'radial-gradient(circle at 20% 50%, rgba(122, 154, 140, 0.1), transparent 50%)',
              opacity: 0,
              transition: 'opacity 0.3s ease',
              pointerEvents: 'none'
            }} className="feature-card__bg" />
            <div style={{ position: 'relative', zIndex: 1 }}>
              <div style={{
                fontSize: '56px',
                marginBottom: '20px',
                width: '80px',
                height: '80px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: 'linear-gradient(135deg, #E8F0EC, #F0EDE8)',
                borderRadius: '16px',
                border: '2px solid rgba(122, 154, 140, 0.2)'
              }}>
                📝
              </div>
              <h3 style={{ fontSize: '22px', fontWeight: '700', marginBottom: '12px', color: 'var(--text)' }}>Text Analysis</h3>
              <p style={{ color: 'var(--text-secondary)', lineHeight: '1.6', marginBottom: '20px' }}>
                Advanced NLP sentiment analysis and emotion extraction from written text with contextual understanding
              </p>
              <ul style={{ listStyle: 'none', padding: 0, display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <li style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-secondary)', fontSize: '14px' }}>✨ Sentiment detection</li>
                <li style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-secondary)', fontSize: '14px' }}>✨ Multi-emotion parsing</li>
                <li style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-secondary)', fontSize: '14px' }}>✨ Context analysis</li>
              </ul>
            </div>
          </motion.div>

          {/* Audio Card */}
          <motion.div
            className="card"
            variants={featureCardVariants}
            whileHover="hover"
            style={{
              background: 'linear-gradient(135deg, var(--surface), var(--accent-ochre-light))',
              border: '1px solid var(--border)',
              cursor: 'pointer',
              overflow: 'hidden',
              position: 'relative'
            }}
          >
            <div style={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              background: 'radial-gradient(circle at 20% 50%, rgba(201, 169, 98, 0.1), transparent 50%)',
              opacity: 0,
              transition: 'opacity 0.3s ease',
              pointerEvents: 'none'
            }} className="feature-card__bg" />
            <div style={{ position: 'relative', zIndex: 1 }}>
              <div style={{
                fontSize: '56px',
                marginBottom: '20px',
                width: '80px',
                height: '80px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: 'linear-gradient(135deg, #FBF7ED, #FAF8F5)',
                borderRadius: '16px',
                border: '2px solid rgba(201, 169, 98, 0.2)'
              }}>
                🎤
              </div>
              <h3 style={{ fontSize: '22px', fontWeight: '700', marginBottom: '12px', color: 'var(--text)' }}>Audio Analysis</h3>
              <p style={{ color: 'var(--text-secondary)', lineHeight: '1.6', marginBottom: '20px' }}>
                Voice emotion recognition from speech patterns, tone, and prosody for comprehensive audio insights
              </p>
              <ul style={{ listStyle: 'none', padding: 0, display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <li style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-secondary)', fontSize: '14px' }}>✨ Real-time speech analysis</li>
                <li style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-secondary)', fontSize: '14px' }}>✨ Tone detection</li>
                <li style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-secondary)', fontSize: '14px' }}>✨ Stress indicators</li>
              </ul>
            </div>
          </motion.div>
        </motion.div>
      </motion.section>

      {/* Why TheraVox Section */}
      <motion.section
        className="container"
        variants={containerVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, margin: '0px 0px -15% 0px' }}
        style={{ paddingTop: '60px', paddingBottom: '80px', background: 'linear-gradient(135deg, var(--surface-secondary), var(--surface))', marginTop: '40px', borderRadius: '20px' }}
      >
        <motion.div style={{ textAlign: 'center', marginBottom: '60px' }} variants={itemVariants}>
          <h2 style={{ fontSize: 'clamp(28px, 5vw, 44px)', marginBottom: '16px', fontWeight: '700' }}>
            Why Choose TheraVox?
          </h2>
          <p style={{ fontSize: '18px', color: 'var(--text-secondary)', maxWidth: '600px', margin: '0 auto' }}>
            Complete emotional intelligence platform for self-understanding and wellness
          </p>
        </motion.div>

        <motion.div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '24px' }} variants={containerVariants}>
          {[
            { icon: '🎯', title: 'Accurate Detection', desc: 'Industry-leading AI models for precise emotion recognition' },
            { icon: '🔒', title: 'Privacy First', desc: 'Your data stays secure with local processing options' },
            { icon: '📊', title: 'Deep Analytics', desc: 'Track patterns and trends in your emotional well-being' },
            { icon: '🚀', title: 'Fast & Efficient', desc: 'Real-time analysis with minimal latency' },
            { icon: '🧘', title: 'Wellness Tools', desc: 'Integrated exercises and resources for mental health' },
            { icon: '💡', title: 'Actionable Insights', desc: 'Get personalized recommendations based on your data' },
          ].map((item, idx) => (
            <motion.div
              key={idx}
              variants={itemVariants}
              style={{
                padding: '28px',
                background: 'var(--surface)',
                borderRadius: '16px',
                border: '1px solid var(--border)',
                textAlign: 'center'
              }}
            >
              <div style={{ fontSize: '48px', marginBottom: '16px' }}>{item.icon}</div>
              <h4 style={{ fontSize: '18px', fontWeight: '700', marginBottom: '8px', color: 'var(--text)' }}>{item.title}</h4>
              <p style={{ color: 'var(--text-secondary)', fontSize: '14px', margin: 0 }}>{item.desc}</p>
            </motion.div>
          ))}
        </motion.div>
      </motion.section>

      {/* How It Works Section */}
      <motion.section
        className="container"
        variants={containerVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, margin: '0px 0px -15% 0px' }}
        style={{ paddingTop: '60px', paddingBottom: '80px' }}
      >
        <motion.div style={{ textAlign: 'center', marginBottom: '60px' }} variants={itemVariants}>
          <h2 style={{ fontSize: 'clamp(28px, 5vw, 44px)', marginBottom: '16px', fontWeight: '700' }}>
            How It Works
          </h2>
          <p style={{ fontSize: '18px', color: 'var(--text-secondary)', maxWidth: '600px', margin: '0 auto' }}>
            Simple steps to understand your emotions better
          </p>
        </motion.div>

        <motion.div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '20px' }} variants={containerVariants}>
          {[
            { num: '1', title: 'Choose Your Mode', desc: 'Select from Vision, Text, or Audio analysis' },
            { num: '2', title: 'Provide Input', desc: 'Upload content or start real-time analysis' },
            { num: '3', title: 'Get Results', desc: 'Instant emotion detection with confidence scores' },
            { num: '4', title: 'Take Action', desc: 'Access wellness tools and recommendations' },
          ].map((step, idx) => (
            <motion.div
              key={idx}
              variants={itemVariants}
              style={{
                padding: '32px 24px',
                background: 'linear-gradient(135deg, var(--surface-secondary), var(--surface))',
                borderRadius: '16px',
                border: '2px solid var(--border-subtle)',
                textAlign: 'center',
                position: 'relative'
              }}
            >
              <div style={{
                width: '56px',
                height: '56px',
                background: 'linear-gradient(135deg, var(--brand), var(--accent-ochre))',
                borderRadius: '50%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'white',
                fontSize: '24px',
                fontWeight: '700',
                margin: '0 auto 16px',
                boxShadow: '0 4px 12px rgba(217, 119, 87, 0.3)'
              }}>
                {step.num}
              </div>
              <h4 style={{ fontSize: '18px', fontWeight: '700', marginBottom: '8px', color: 'var(--text)' }}>{step.title}</h4>
              <p style={{ color: 'var(--text-secondary)', fontSize: '14px', margin: 0 }}>{step.desc}</p>
            </motion.div>
          ))}
        </motion.div>
      </motion.section>

      {/* CTA Section */}
      <motion.section
        className="container"
        variants={containerVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, margin: '0px 0px -15% 0px' }}
        style={{
          paddingTop: '80px',
          paddingBottom: '80px',
          textAlign: 'center',
          background: 'linear-gradient(135deg, var(--brand), var(--brand-dark))',
          borderRadius: '20px',
          color: 'white',
          marginTop: '60px'
        }}
      >
        <motion.h2 variants={itemVariants} style={{ fontSize: 'clamp(28px, 5vw, 44px)', marginBottom: '20px' }}>
          Ready to Understand Your Emotions?
        </motion.h2>
        <motion.p variants={itemVariants} style={{ fontSize: '18px', marginBottom: '40px', maxWidth: '600px', margin: '0 auto 40px' }}>
          Start your emotional intelligence journey today with TheraVox AI
        </motion.p>
        <motion.div variants={itemVariants} style={{ display: 'flex', gap: '16px', justifyContent: 'center', flexWrap: 'wrap' }}>
          <Link to="/vision" style={{ padding: '16px 32px', background: 'white', color: 'var(--brand)', borderRadius: '12px', textDecoration: 'none', fontWeight: '600', fontSize: '16px', transition: 'all 0.3s ease', border: '2px solid white' }}
            onMouseOver={(e) => {
              e.currentTarget.style.background = 'transparent';
              e.currentTarget.style.color = 'white';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.background = 'white';
              e.currentTarget.style.color = 'var(--brand)';
            }}>
            🚀 Start Now
          </Link>
          <Link to="/wellness" style={{ padding: '16px 32px', background: 'transparent', color: 'white', borderRadius: '12px', textDecoration: 'none', fontWeight: '600', fontSize: '16px', transition: 'all 0.3s ease', border: '2px solid white' }}
            onMouseOver={(e) => {
              e.currentTarget.style.background = 'white';
              e.currentTarget.style.color = 'var(--brand)';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.background = 'transparent';
              e.currentTarget.style.color = 'white';
            }}>
            🧘 Explore Wellness
          </Link>
        </motion.div>
      </motion.section>
    </>
  );
}
