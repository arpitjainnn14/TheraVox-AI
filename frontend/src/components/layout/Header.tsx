import { useLocation, Link, useNavigate } from 'react-router-dom';
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
    <header className="header">
      <div className="container">
        <div className="header__inner">
          <Link to="/" className="brand has-logo">
            <img
              src="/logo.png"
              alt="TheraVox logo"
              className="brand__logo"
            />
            <span className="brand__text">TheraVox</span>
          </Link>

          <nav className="nav">
            <Link
              to="/vision"
              className={`nav__link ${isActive('/vision') ? 'active' : ''}`}
            >
              Vision
            </Link>
            <Link
              to="/text"
              className={`nav__link ${isActive('/text') ? 'active' : ''}`}
            >
              Text
            </Link>
            <Link
              to="/audio"
              className={`nav__link ${isActive('/audio') ? 'active' : ''}`}
            >
              Audio
            </Link>

            {/* Authenticated user menu */}
            {isAuthenticated && user && (
              <div className="nav__user">
                <div
                  className="nav__avatar"
                  title={user.full_name}
                  aria-label={`Logged in as ${user.full_name}`}
                >
                  {getInitials(user.full_name)}
                </div>
                <span className="nav__username">{user.full_name.split(' ')[0]}</span>
                <button
                  className="nav__logout"
                  onClick={handleLogout}
                  type="button"
                  aria-label="Sign out"
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
