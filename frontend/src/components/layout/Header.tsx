import { useState } from 'react';
import { useLocation, Link, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from '../../contexts/AuthContext';
import '../../styles/auth.css';

/** Returns up to 2 uppercase initials from a full name. */
function getInitials(name: string): string {
  return name
    .trim()
    .split(/\s+/)
    .slice(0, 2)
    .map(n => n[0]?.toUpperCase() ?? '')
    .join('');
}

// --- Hoverable Nav Menu Item ---
const NavMenuItem = ({
  title,
  path,
  icon,
  description,
  isActive
}: {
  title: string;
  path: string;
  icon: string;
  description: string;
  isActive: boolean;
}) => {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <div
      style={{ position: 'relative', display: 'flex', alignItems: 'center', height: '100%' }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <span
        className={`nav__link ${isActive ? 'active' : ''}`}
        style={{ padding: '10px 16px', display: 'flex', alignItems: 'center', cursor: 'default' }}
      >
        {title}
      </span>

      <AnimatePresence>
        {isHovered && (
          <motion.div
            initial={{ opacity: 0, y: 15, x: '-50%', scale: 0.95 }}
            animate={{ opacity: 1, y: 0, x: '-50%', scale: 1 }}
            exit={{ opacity: 0, y: 10, x: '-50%', scale: 0.95 }}
            transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
            style={{
              position: 'absolute',
              top: '100%',
              left: '50%',
              width: '320px',
              backgroundColor: 'rgba(255, 255, 255, 0.98)',
              backdropFilter: 'blur(20px)',
              border: '1px solid var(--border)',
              borderRadius: '20px',
              padding: '24px',
              boxShadow: '0 24px 48px rgba(0,0,0,0.08), 0 0 0 1px rgba(255,255,255,0.5) inset',
              zIndex: 100,
              marginTop: '4px',
              cursor: 'default',
            }}
          >
            <div style={{ 
              fontSize: '32px', 
              marginBottom: '16px',
              width: '56px',
              height: '56px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: 'var(--surface-secondary)',
              borderRadius: '16px',
              border: '1px solid var(--border-subtle)'
            }}>
              {icon}
            </div>
            <h4 style={{ margin: '0 0 8px 0', fontSize: '18px', fontWeight: 700, color: 'var(--text)', letterSpacing: '-0.01em' }}>
              {title} Analysis
            </h4>
            <p style={{ margin: '0 0 20px 0', fontSize: '14px', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              {description}
            </p>
            <Link
              to={path}
              style={{
                display: 'block',
                padding: '12px 20px',
                backgroundColor: 'var(--brand)',
                color: 'white',
                borderRadius: '12px',
                fontSize: '14px',
                fontWeight: 600,
                textDecoration: 'none',
                textAlign: 'center',
                transition: 'all 0.2s',
                boxShadow: '0 4px 12px rgba(217, 119, 87, 0.2)'
              }}
              onMouseOver={(e) => {
                e.currentTarget.style.backgroundColor = 'var(--brand-hover)';
                e.currentTarget.style.transform = 'translateY(-1px)';
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.backgroundColor = 'var(--brand)';
                e.currentTarget.style.transform = 'none';
              }}
            >
              Open {title} Studio
            </Link>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default function Header() {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, isAuthenticated, logout } = useAuth();

  const isActive = (path: string) => location.pathname === path;

  const handleLogout = async () => {
    await logout();
    navigate('/login', { replace: true });
  };

  return (
    <header className="header" style={{ overflow: 'visible' }}>
      <div className="container" style={{ position: 'relative' }}>
        <div className="header__inner">
          <Link to="/" className="brand has-logo">
            <img
              src="/logo.png"
              alt="TheraVox logo"
              className="brand__logo"
            />
            <span className="brand__text">TheraVox AI</span>
          </Link>

          <nav className="nav" style={{ display: 'flex', alignItems: 'stretch', height: '100%' }}>
            <NavMenuItem
              title="Vision"
              path="/vision"
              icon="👁️"
              description="Analyze real-time facial expressions and micro-expressions to decode emotional states accurately."
              isActive={isActive('/vision')}
            />
            <NavMenuItem
              title="Text"
              path="/text"
              icon="📝"
              description="Extract deep sentiment, tone, and emotional context from written text using advanced NLP."
              isActive={isActive('/text')}
            />
            <NavMenuItem
              title="Audio"
              path="/audio"
              icon="🎤"
              description="Understand vocal tone, stress levels, and emotion through comprehensive speech and audio analysis."
              isActive={isActive('/audio')}
            />
            <NavMenuItem
              title="Wellness"
              path="/wellness"
              icon="✨"
              description="Track your mood, practice breathing exercises, gratitude journaling, and mindfulness tools."
              isActive={isActive('/wellness')}
            />

            {/* Authenticated user menu */}
            {isAuthenticated && user && (
              <div className="nav__user" style={{ marginLeft: '24px', position: 'relative', zIndex: 10 }}>
                <Link
                  to="/profile"
                  className="nav__avatar"
                  title={`View profile: ${user.full_name}`}
                  aria-label={`Profile: ${user.full_name}`}
                  style={{ textDecoration: 'none' }}
                >
                  {getInitials(user.full_name)}
                </Link>
                <button
                  className="nav__logout"
                  onClick={handleLogout}
                  type="button"
                  aria-label="Sign out"
                  style={{ marginLeft: '8px' }}
                >
                  Sign out
                </button>
              </div>
            )}
          </nav>
        </div>
      </div>
    </header>
  );
}
