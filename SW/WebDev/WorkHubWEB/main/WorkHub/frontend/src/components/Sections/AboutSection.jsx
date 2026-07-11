import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import '../../styles/WelcomePage.css';

export default function AboutSection() {
  const [expandedIndex, setExpandedIndex] = useState(null);
  const navigate = useNavigate();

  const aboutItems = [
    {
      id: 'seekers',
      title: 'For Job Seekers',
      content: 'Discover your next opportunity with WorkHub. Browse companies, view team structures, explore job openings, and connect with hiring managers. All in one centralized platform designed for your career growth. Find the perfect role that matches your skills and ambitions.'
    },
    {
      id: 'companies',
      title: 'For Companies',
      content: 'Streamline your recruitment, centralize team management, track productivity, enable internal communication, and gain actionable analytics. WorkHub is your all-in-one solution for organizational management. Scale your team efficiently while maintaining culture and collaboration.'
    },
    {
      id: 'why',
      title: 'Why WorkHub',
      content: 'Say goodbye to scattered tools and fragmented workflows. WorkHub brings everything together: recruitment, team collaboration, task management, analytics, and communication. One platform. Complete control. Measurable results. Join the revolution in organizational management.'
    }
  ];

  const toggleExpand = (index) => {
    setExpandedIndex(expandedIndex === index ? null : index);
  };

  const handleGetStarted = () => {
    // Navigate to ChooseRegister page
    navigate('/choose-register');
  };

  return (
    <div className="about-section">
      <div className="about-intro">
        <p>
          WorkHub is the comprehensive solution that brings organizations and talent together.
          Whether you're a company looking to streamline operations or a professional seeking
          new opportunities, WorkHub provides the tools and connections you need to succeed.
        </p>
      </div>

      <div className="about-items">
        {aboutItems.map((item, index) => (
          <div key={item.id} className="about-item">
            <button 
              className="about-item-header"
              onClick={() => toggleExpand(index)}
            >
              <span className="about-title">{item.title}</span>
              <span className={`expand-icon ${expandedIndex === index ? 'open' : ''}`}>
                ▼
              </span>
            </button>

            {expandedIndex === index && (
              <div className="about-item-content">
                <p>{item.content}</p>
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="about-footer">
        <p>Join thousands of companies and professionals already using WorkHub</p>
        <button className="cta-button" onClick={handleGetStarted}>Get Started Today</button>
      </div>
    </div>
  );
}