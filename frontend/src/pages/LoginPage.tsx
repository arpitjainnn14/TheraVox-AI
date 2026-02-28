import type { FormEvent } from 'react';
import { useEffect, useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from '../contexts/AuthContext';
import { IconAlert, IconEmail, IconEye, IconEyeOff, IconGitHub, IconGoogle, IconLock } from '../components/shared/Icons';
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
      staggerChildren: 0.08,
      delayChildren: 0.2
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

export default function LoginPage() {
  const { login, isAuthenticated, isLoading } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (!isLoading && isAuthenticated) {
      const from = (location.state as { from?: { pathname: string } })?.from?.pathname ?? '/';
      navigate(from, { replace: true });
    }
  }, [isAuthenticated, isLoading, navigate, location.state]);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');
    setSubmitting(true);
    try {
      await login(email.trim(), password);
      const from = (location.state as { from?: { pathname: string } })?.from?.pathname ?? '/';
      navigate(from, { replace: true });
    } catch (err) {
      setError((err as Error).message ?? 'Login failed. Please try again.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="auth-page">
      {/* Background with animated orbs */}
      <div className="auth-bg">
        <div className="auth-orb auth-orb--1" />
        <div className="auth-orb auth-orb--2" />
        <div className="auth-orb auth-orb--3" />
      </div>

      <motion.div
        className="auth-card"
        variants={cardVariants}
        initial="hidden"
        animate="visible"
      >
        <div className="auth-card-accent" />

        <motion.div variants={staggerContainer} initial="hidden" animate="visible">
          {/* Brand Logo */}
          <motion.div variants={itemVariants}>
            <Link to="/" className="auth-brand">
              <div className="auth-brand__icon-wrap">
                <img src="/logo.png" alt="TheraVox logo" className="auth-brand__logo" />
              </div>
              <span className="auth-brand__text">TheraVox AI</span>
            </Link>
          </motion.div>

          <motion.h1 variants={itemVariants} className="auth-heading">
            Welcome back
          </motion.h1>
          <motion.p variants={itemVariants} className="auth-subtitle">
            Enter your details to access your account
          </motion.p>

          <form className="auth-form" onSubmit={handleSubmit} noValidate>
            {/* Error Message */}
            <AnimatePresence mode="wait">
              {error && (
                <motion.div
                  key="login-error"
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

            {/* Email Field */}
            <motion.div variants={itemVariants} className="auth-field">
              <label htmlFor="login-email">Email Address</label>
              <div className="auth-input-wrap">
                <span className="auth-input-icon"><IconEmail /></span>
                <input
                  id="login-email"
                  type="email"
                  autoComplete="email"
                  placeholder="name@company.com"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  required
                  disabled={submitting}
                />
              </div>
            </motion.div>

            {/* Password Field */}
            <motion.div variants={itemVariants} className="auth-field">
              <label htmlFor="login-password">Password</label>
              <div className="auth-input-wrap">
                <span className="auth-input-icon"><IconLock /></span>
                <input
                  id="login-password"
                  type={showPassword ? 'text' : 'password'}
                  autoComplete="current-password"
                  placeholder="••••••••••••"
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
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                >
                  {showPassword ? <IconEyeOff /> : <IconEye />}
                </button>
              </div>
            </motion.div>

            {/* Remember Me & Forgot Password */}
            <motion.div variants={itemVariants} className="auth-actions">
            </motion.div>

            {/* Submit Button */}
            <motion.div variants={itemVariants}>
              <button
                type="submit"
                className="auth-btn"
                disabled={submitting || !email || !password}
              >
                {submitting ? (
                  <>
                    <div className="auth-spinner" />
                    <span>Signing in...</span>
                  </>
                ) : (
                  'Sign in'
                )}
              </button>
            </motion.div>
          </form>

          {/* Social Logins */}
          <motion.div variants={itemVariants} className="auth-social">
            <div className="auth-social-title">Or continue with</div>
            <div className="auth-social-grid">
              <button
                type="button"
                className="auth-social-btn"
                onClick={() => { window.location.href = '/api/auth/google/authorize'; }}
              >
                <IconGoogle />
                <span>Google</span>
              </button>
              <button
                type="button"
                className="auth-social-btn"
                onClick={() => { window.location.href = '/api/auth/github/authorize'; }}
              >
                <IconGitHub />
                <span>GitHub</span>
              </button>
            </div>
          </motion.div>

          <motion.p variants={itemVariants} className="auth-footer">
            Don't have an account?{' '}
            <Link to="/register" state={location.state}>Create an account</Link>
          </motion.p>
        </motion.div>
      </motion.div>
    </div>
  );
}
