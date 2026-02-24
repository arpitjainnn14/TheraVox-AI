import { useLocation } from 'react-router-dom';
import Header from './Header';
import Footer from './Footer';
import FloatingWellness from './FloatingWellness';
import FloatingFeedback from './FloatingFeedback';
import GreetingBanner from './GreetingBanner';

interface LayoutProps {
  children: React.ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  const location = useLocation();
  const isWellnessPage = location.pathname === '/wellness';

  return (
    <div className="layout">
      <Header />
      <div className="container">
        <GreetingBanner />
      </div>
      <main id="main" className="container">
        {children}
      </main>
      <Footer />
      {!isWellnessPage && <FloatingWellness />}
      <FloatingFeedback />
    </div>
  );
}
