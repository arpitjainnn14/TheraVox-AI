import type { FormEvent } from 'react';
import { useEffect, useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from '../contexts/AuthContext';
import '../styles/auth.css';

// --- Animation Variants ---
const cardVariants = {
  hidden: { opacity: 0, y: 40, scale: 0.98 },
  visible: { 
    opacity: 1, y: 0, scale: 1,
    transition: { duration: 0.6, ease: [0.22, 1, 0.36, 1] as [number, number, number, number] }
  }
};

const staggerContainer = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.07,
      delayChildren: 0.15
    }
  }
};

const itemVariants = {
  hidden: { opacity: 0, y: 15 },
  visible: { 
    opacity: 1, y: 0, 
    transition: { duration: 0.4, ease: [0.22, 1, 0.36, 1] as [number, number, number, number] } 
  }
};

// --- Password strength scorer ---
function getPasswordStrength(pw: string): { score: number; label: string; color: string } {
  if (!pw) return { score: 0, label: '', color: '' };
  let score = 0;
  if (pw.length >= 8) score++;
  if (pw.length >= 12) score++;
  if (/[A-Z]/.test(pw)) score++;
  if (/[0-9]/.test(pw)) score++;
  if (/[^A-Za-z0-9]/.test(pw)) score++;

  const levels = [
    { score: 0, label: '',            color: 'var(--border)' },
    { score: 1, label: 'Weak',        color: '#EF4444' },
    { score: 2, label: 'Fair',        color: '#F97316' },
    { score: 3, label: 'Good',        color: '#EAB308' },
    { score: 4, label: 'Strong',      color: '#10B981' },
    { score: 5, label: 'Very strong', color: '#059669' },
  ];
  return levels[score] ?? levels[5];
}

// --- Icons ---
const IconUser = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>
  </svg>
);

const IconEmail = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/>
    <polyline points="22,6 12,13 2,6"/>
  </svg>
);

const IconLock = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
    <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
  </svg>
);

const IconEyeOff = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/>
    <line x1="1" y1="1" x2="23" y2="23"/>
  </svg>
);

const IconEye = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>
    <circle cx="12" cy="12" r="3"/>
  </svg>
);

const IconAlert = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
  </svg>
);

const IconCheck = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="20 6 9 17 4 12"/>
  </svg>
);

const IconX = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
    <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
  </svg>
);

const IconGoogle = () => (
  <svg viewBox="0 0 24 24" width="20" height="20">
    <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
    <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
    <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
    <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
  </svg>
);

const IconGitHub = () => (
  <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
    <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/>
  </svg>
);

export default function RegisterPage() {
  const { register, isAuthenticated, isLoading } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const [fullName, setFullName]               = useState('');
  const [email, setEmail]                     = useState('');
  const [password, setPassword]               = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword]       = useState(false);
  const [showConfirm, setShowConfirm]         = useState(false);
  const [acceptTerms, setAcceptTerms]         = useState(false);
  const [error, setError]                     = useState('');
  const [submitting, setSubmitting]           = useState(false);

  useEffect(() => {
    if (!isLoading && isAuthenticated) {
      const from = (location.state as { from?: { pathname: string } })?.from?.pathname ?? '/';
      navigate(from, { replace: true });
    }
  }, [isAuthenticated, isLoading, navigate, location.state]);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');

    if (password.length < 8) {
      setError('Password must be at least 8 characters.');
      return;
    }
    if (password !== confirmPassword) {
      setError('Passwords do not match.');
      return;
    }
    if (!acceptTerms) {
      setError('You must accept the terms and conditions.');
      return;
    }

    setSubmitting(true);
    try {
      await register(fullName.trim(), email.trim(), password);
      const from = (location.state as { from?: { pathname: string } })?.from?.pathname ?? '/';
      navigate(from, { replace: true });
    } catch (err) {
      setError((err as Error).message ?? 'Registration failed. Please try again.');
    } finally {
      setSubmitting(false);
    }
  };

  const strength = getPasswordStrength(password);
  const isFormValid = fullName.trim() && email && password.length >= 8 && confirmPassword && acceptTerms;
  const passwordsMatch = confirmPassword.length > 0 && password === confirmPassword;
  const passwordsMismatch = confirmPassword.length > 0 && password !== confirmPassword;

  return (
    <div className="auth-page auth-page--split">
      <div className="auth-bg">
        <div className="auth-orb auth-orb--1" />
        <div className="auth-orb auth-orb--2" />
        <div className="auth-orb auth-orb--3" />
      </div>

      <div className="auth-split-wrapper">
        <motion.div 
          className="auth-info-panel"
          variants={staggerContainer}
          initial="hidden"
          animate="visible"
        >
          <motion.div variants={itemVariants}>
            <Link to="/" className="auth-brand auth-brand--large" style={{ textDecoration: 'none' }}>
              <motion.div 
                className="auth-brand__icon-wrap"
                whileHover={{ rotate: [0, -10, 10, -5, 5, 0], scale: 1.05 }}
                transition={{ duration: 0.5 }}
              >
                <img src="/logo.png" alt="TheraVox logo" className="auth-brand__logo" />
              </motion.div>
              <span className="auth-brand__text">TheraVox AI</span>
            </Link>
          </motion.div>
          
          <div className="auth-info-content">
            <motion.h2 variants={itemVariants}>Your journey to emotional wellness starts here.</motion.h2>
            <motion.p variants={itemVariants}>TheraVox uses advanced AI to analyze sentiment, provide insights, and foster personal growth through reflective journaling and voice analysis.</motion.p>
            
            <motion.ul className="auth-features" variants={staggerContainer}>
              {[
                { icon: '✨', text: 'Advanced emotional insights' },
                { icon: '🎙️', text: 'Voice and text journal analysis' },
                { icon: '🔒', text: 'Secure and private entries' },
                { icon: '🌱', text: 'Personalized wellness journey' }
              ].map((feature, i) => (
                <motion.li 
                  key={i} 
                  variants={itemVariants}
                  whileHover={{ 
                    scale: 1.05, 
                    x: 10, 
                    backgroundColor: 'rgba(255, 255, 255, 1)', 
                    boxShadow: '0 8px 16px rgba(0,0,0,0.08)',
                    borderColor: 'var(--brand)'
                  }}
                  whileTap={{ scale: 0.98 }}
                  style={{ cursor: 'pointer' }}
                >
                  <span className="auth-feature-icon">{feature.icon}</span> 
                  {feature.text}
                </motion.li>
              ))}
            </motion.ul>
          </div>
        </motion.div>

        <div className="auth-form-panel">
          <motion.div
            className="auth-card"
            variants={cardVariants}
            initial="hidden"
            animate="visible"
          >
            <div className="auth-card-accent" />

            <motion.div variants={staggerContainer} initial="hidden" animate="visible">
              <motion.div variants={itemVariants} className="auth-card-brand">
                <Link to="/" className="auth-brand">
                  <div className="auth-brand__icon-wrap">
                    <img src="/logo.png" alt="TheraVox logo" className="auth-brand__logo" />
                  </div>
                  <span className="auth-brand__text">TheraVox AI</span>
                </Link>
              </motion.div>

              <motion.h1 variants={itemVariants} className="auth-heading">
                Create account
              </motion.h1>
          <motion.p variants={itemVariants} className="auth-subtitle">
            Join TheraVox for your emotional wellness journey
          </motion.p>

          <form className="auth-form" onSubmit={handleSubmit} noValidate>
            <AnimatePresence mode="wait">
              {error && (
                <motion.div
                  key="reg-error"
                  className="auth-error"
                  role="alert"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                >
                  <IconAlert />
                  {error}
                </motion.div>
              )}
            </AnimatePresence>

            <motion.div variants={itemVariants} className="auth-field">
              <label htmlFor="reg-fullname">Full Name</label>
              <div className="auth-input-wrap">
                <span className="auth-input-icon"><IconUser /></span>
                <input
                  id="reg-fullname"
                  type="text"
                  autoComplete="name"
                  placeholder="Jane Smith"
                  value={fullName}
                  onChange={e => setFullName(e.target.value)}
                  required
                  disabled={submitting}
                />
              </div>
            </motion.div>

            <motion.div variants={itemVariants} className="auth-field">
              <label htmlFor="reg-email">Email Address</label>
              <div className="auth-input-wrap">
                <span className="auth-input-icon"><IconEmail /></span>
                <input
                  id="reg-email"
                  type="email"
                  autoComplete="email"
                  placeholder="jane@example.com"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  required
                  disabled={submitting}
                />
              </div>
            </motion.div>

            <motion.div variants={itemVariants} className="auth-field">
              <label htmlFor="reg-password">Password</label>
              <div className="auth-input-wrap">
                <span className="auth-input-icon"><IconLock /></span>
                <input
                  id="reg-password"
                  type={showPassword ? 'text' : 'password'}
                  autoComplete="new-password"
                  placeholder="Min. 8 characters"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  required
                  disabled={submitting}
                  className="auth-input--password"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(v => !v)}
                  className="auth-password-toggle"
                >
                  {showPassword ? <IconEyeOff /> : <IconEye />}
                </button>
              </div>

              <AnimatePresence>
                {password && (
                  <motion.div
                    className="auth-strength"
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                  >
                    <div className="auth-strength__bars">
                      {[1, 2, 3, 4, 5].map(i => (
                        <div
                          key={i}
                          className="auth-strength__bar"
                          style={{ 
                            background: i <= strength.score ? strength.color : undefined,
                            boxShadow: i <= strength.score ? `0 0 8px ${strength.color}40` : 'none'
                          }}
                        />
                      ))}
                    </div>
                    <span className="auth-strength__label" style={{ color: strength.color }}>
                      {strength.label}
                    </span>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>

            <motion.div variants={itemVariants} className="auth-field">
              <label htmlFor="reg-confirm">Confirm Password</label>
              <div className="auth-input-wrap">
                <span className="auth-input-icon"><IconLock /></span>
                <input
                  id="reg-confirm"
                  type={showConfirm ? 'text' : 'password'}
                  autoComplete="new-password"
                  placeholder="Repeat password"
                  value={confirmPassword}
                  onChange={e => setConfirmPassword(e.target.value)}
                  required
                  disabled={submitting}
                  className="auth-input--password"
                />
                <button
                  type="button"
                  onClick={() => setShowConfirm(v => !v)}
                  className="auth-password-toggle"
                >
                  {showConfirm ? <IconEyeOff /> : <IconEye />}
                </button>
              </div>

              <AnimatePresence>
                {(passwordsMatch || passwordsMismatch) && (
                  <motion.div
                    className={`auth-match-indicator ${passwordsMatch ? 'auth-match-indicator--ok' : 'auth-match-indicator--no'}`}
                    initial={{ opacity: 0, y: -4 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                  >
                    {passwordsMatch ? (
                      <><IconCheck /> Passwords match</>
                    ) : (
                      <><IconX /> Passwords don't match</>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>

            <motion.div variants={itemVariants} className="auth-actions">
              <label className="auth-remember">
                <input 
                  type="checkbox" 
                  checked={acceptTerms} 
                  onChange={e => setAcceptTerms(e.target.checked)}
                />
                <span>I accept the <Link to="/terms" target="_blank" rel="noopener noreferrer">Terms</Link> & <a href="#" onClick={e => e.preventDefault()}>Privacy</a></span>
              </label>
            </motion.div>

            <motion.div variants={itemVariants}>
              <button
                type="submit"
                className="auth-btn"
                disabled={submitting || !isFormValid}
              >
                {submitting ? (
                  <>
                    <div className="auth-spinner" />
                    <span>Creating account...</span>
                  </>
                ) : (
                  'Create account'
                )}
              </button>
            </motion.div>
          </form>

          <motion.div variants={itemVariants} className="auth-social">
            <div className="auth-social-title">Or sign up with</div>
            <div className="auth-social-grid">
              <button type="button" className="auth-social-btn" title="Social login is a demo feature">
                <IconGoogle />
                <span>Google</span>
              </button>
              <button type="button" className="auth-social-btn" title="Social login is a demo feature">
                <IconGitHub />
                <span>GitHub</span>
              </button>
            </div>
          </motion.div>

          <motion.p variants={itemVariants} className="auth-footer">
            Already have an account?{' '}
            <Link to="/login" state={location.state}>Sign in</Link>
          </motion.p>
        </motion.div>
      </motion.div>
        </div>
      </div>
    </div>
  );
}
