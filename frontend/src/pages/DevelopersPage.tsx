import { motion } from 'framer-motion';
import HeroSection from '../components/shared/HeroSection';

interface TeamMember {
  name: string;
  role: string;
  bio: string;
  linkedin?: string;
  github?: string;
}

const TEAM: TeamMember[] = [
  {
    name: 'Arpit Jain',
    role: 'Full-stack & Vision',
    bio: 'Full-stack developer specializing in computer vision and emotion detection from facial expressions.',
    linkedin: 'https://linkedin.com/in/arpit-jain',
    github: 'https://github.com/arpit-jain',
  },
  {
    name: 'Ajit Dixit',
    role: 'Full-stack',
    bio: 'Full-stack engineer responsible for API development, backend optimization, and audio processing.',
    linkedin: 'https://linkedin.com/in/ajit-dixit',
    github: 'https://github.com/ajit-dixit',
  },
  {
    name: 'Bhoomi Chowksey',
    role: 'Text & Frontend',
    bio: 'Frontend specialist and text emotion analysis expert. Responsible for UI/UX design and text NLP.',
    linkedin: 'https://linkedin.com/in/bhoomi-chowksey',
    github: 'https://github.com/bhoomi-chowksey',
  },
  {
    name: 'Suchita Nandi',
    role: 'QA & Docs',
    bio: 'Quality assurance engineer and technical writer ensuring reliability and comprehensive documentation.',
    linkedin: 'https://linkedin.com/in/suchita-nandi',
    github: 'https://github.com/suchita-nandi',
  },
  {
    name: 'Aditi Bathla',
    role: 'UX & Integration',
    bio: 'UX/UI designer and integration specialist coordinating all components into a seamless experience.',
    linkedin: 'https://linkedin.com/in/aditi-bathla',
    github: 'https://github.com/aditi-bathla',
  },
];

export default function DevelopersPage() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.1, delayChildren: 0.2 },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.5 } },
  };

  return (
    <>
      <HeroSection title="Project Developers" subtitle="Capstone Final Project • 8 Credits" />

      <motion.div
        className="container"
        style={{ maxWidth: '1100px' }}
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        <motion.div className="team-grid" variants={containerVariants}>
          {TEAM.map((member) => (
            <motion.article key={member.name} className="member-card" variants={itemVariants}>
              <div
                className="member-avatar"
                data-initials={member.name
                  .split(' ')
                  .map((n) => n[0])
                  .join('')}
                style={{
                  width: '80px',
                  height: '80px',
                  borderRadius: '50%',
                  backgroundColor: '#d97757',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#fff',
                  fontSize: '24px',
                  fontWeight: 'bold',
                  marginBottom: '16px',
                }}
              />
              <h3 style={{ marginBottom: '4px' }}>{member.name}</h3>
              <p
                style={{
                  color: '#d97757',
                  fontWeight: '600',
                  fontSize: '14px',
                  marginBottom: '12px',
                }}
              >
                {member.role}
              </p>
              <p style={{ fontSize: '14px', lineHeight: '1.6', marginBottom: '16px' }}>
                {member.bio}
              </p>
              <div
                style={{
                  display: 'flex',
                  gap: '12px',
                  justifyContent: 'center',
                }}
              >
                {member.linkedin && (
                  <a
                    href={member.linkedin}
                    target="_blank"
                    rel="noopener noreferrer"
                    title="LinkedIn"
                    style={{
                      display: 'inline-flex',
                      width: '36px',
                      height: '36px',
                      alignItems: 'center',
                      justifyContent: 'center',
                      borderRadius: '50%',
                      backgroundColor: '#f5f2ed',
                      color: '#d97757',
                      textDecoration: 'none',
                      transition: 'all 0.25s',
                    }}
                  >
                    in
                  </a>
                )}
                {member.github && (
                  <a
                    href={member.github}
                    target="_blank"
                    rel="noopener noreferrer"
                    title="GitHub"
                    style={{
                      display: 'inline-flex',
                      width: '36px',
                      height: '36px',
                      alignItems: 'center',
                      justifyContent: 'center',
                      borderRadius: '50%',
                      backgroundColor: '#f5f2ed',
                      color: '#d97757',
                      textDecoration: 'none',
                      transition: 'all 0.25s',
                    }}
                  >
                    gh
                  </a>
                )}
              </div>
            </motion.article>
          ))}
        </motion.div>

        <motion.div
          className="card"
          style={{ gridColumn: '1 / -1', marginTop: '48px' }}
          variants={itemVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
        >
          <h2>About This Project</h2>
          <p>
            TheraVox is an innovative multimodal emotion analysis system that leverages state-of-the-art
            AI models to detect and analyze emotions from vision, text, and audio inputs. Built as a
            capstone final project, it combines computer vision, natural language processing, and audio
            analysis into a unified platform for mental health awareness and emotional intelligence.
          </p>
          <p>
            The project integrates FastAPI for backend services, React with Framer Motion for responsive
            animations, and advanced machine learning models for accurate emotion detection. It includes
            a comprehensive wellness sanctuary with meditation tools, journaling features, mood tracking,
            and personalized insights.
          </p>
        </motion.div>
      </motion.div>
    </>
  );
}
