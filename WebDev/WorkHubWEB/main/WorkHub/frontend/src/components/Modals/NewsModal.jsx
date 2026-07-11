import React, { useState, useEffect } from 'react';
import { createProfileIcon } from '../../utils/userProfileUtils';
import '../../styles/WelcomePage.css';

export default function NewsModal({ news, onClose, onCompanyClick }) {
  const [showAuthorCard, setShowAuthorCard] = useState(false);
  const [authorCardPosition, setAuthorCardPosition] = useState({ x: 0, y: 0 });
  const [hideTimeout, setHideTimeout] = useState(null);

  useEffect(() => {
    const handleEsc = (e) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', handleEsc);
    return () => {
      window.removeEventListener('keydown', handleEsc);
      if (hideTimeout) {
        clearTimeout(hideTimeout);
      }
    };
  }, [onClose, hideTimeout]);

  const handleAuthorHover = (e) => {
    // Clear any pending hide timeout
    if (hideTimeout) {
      clearTimeout(hideTimeout);
      setHideTimeout(null);
    }

    const rect = e.target.getBoundingClientRect();
    setAuthorCardPosition({
      x: rect.left + rect.width / 2,
      y: rect.bottom + 5 // Changed from 10 to 5 - closer to the name
    });
    setShowAuthorCard(true);
  };

  const handleAuthorLeave = () => {
    // Delay hiding to give user time to move to the card
    const timeout = setTimeout(() => {
      setShowAuthorCard(false);
    }, 1000); // Increased from 300ms to 500ms for more time
    setHideTimeout(timeout);
  };

  const handleCardEnter = () => {
    // Cancel hide when entering the card
    if (hideTimeout) {
      clearTimeout(hideTimeout);
      setHideTimeout(null);
    }
  };

  const handleCardLeave = () => {
    // Hide immediately when leaving the card
    setShowAuthorCard(false);
  };

  const handleCompanyClick = () => {
    if (onCompanyClick) {
      onCompanyClick(news.company.name, news.company.id);
      onClose();
    }
  };

  const handleEmailClick = () => {
    window.location.href = `mailto:${news.author.email || 'contact@company.com'}`;
  };

  // Check if image exists and is not empty
  const hasImage = news.image && news.image.trim() !== '' && news.image !== 'null';

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content news-modal" onClick={e => e.stopPropagation()}>
        {/* Sticky Header */}
        <div className="modal-header sticky-header">
          <div className="author-section">
            <div 
              className="author-info-hover"
              onMouseEnter={handleAuthorHover}
              onMouseLeave={handleAuthorLeave}
            >
              {createProfileIcon(news.author.name.split(' ')[0], news.author.avatarColor, 40)}
              <span className="author-name-modal">{news.author.name}</span>
            </div>
          </div>
          <button className="close-btn" onClick={onClose}>×</button>
        </div>

        {/* Author Hover Card */}
        {showAuthorCard && (
          <div 
            className="author-hover-card"
            style={{
              left: `${authorCardPosition.x}px`,
              top: `${authorCardPosition.y}px`,
              transform: 'translateX(-50%)',
              pointerEvents: 'auto' // Make sure it can receive mouse events
            }}
            onMouseEnter={handleCardEnter}
            onMouseLeave={handleCardLeave}
          >
            <div className="hover-card-content">
              {createProfileIcon(news.author.name.split(' ')[0], news.author.avatarColor, 60)}
              <h4>{news.author.name}</h4>
              <p className="hover-company">{news.company.name} - {news.author.departmentName || 'Department'}</p>
              <button className="contact-email-btn" onClick={handleEmailClick}>
                📧 {news.author.email || 'contact@company.com'}
              </button>
            </div>
          </div>
        )}

        {/* Scrollable Body */}
        <div className="modal-body scrollable-body">
          {/* Only show image if it exists */}
          {hasImage && (
            <div className="news-modal-image">
              <img 
                src={news.image} 
                alt={news.title}
                onError={(e) => {
                  e.target.parentElement.style.display = 'none';
                }}
              />
            </div>
          )}

          <h2 className="news-modal-title">{news.title}</h2>
          
          <div className="news-modal-content">
            {news.content.split('\n').map((paragraph, index) => (
              <p key={index}>{paragraph}</p>
            ))}
          </div>

          <div className="news-modal-meta">
            <div className="meta-left">
              <span className="news-date-modal">
                {new Date(news.date || news.created_at).toLocaleDateString('en-US', { 
                  year: 'numeric', 
                  month: 'long', 
                  day: 'numeric' 
                })}
              </span>
              <span className="company-tag">{news.company.industry}</span>
            </div>
            
            <div className="meta-right" onClick={handleCompanyClick}>
              {news.company.logo_url && (
                <img 
                  src={news.company.logo_url} 
                  alt={news.company.name}
                  className="company-logo-modal"
                  onError={(e) => {
                    e.target.style.display = 'none';
                  }}
                />
              )}
              <span className="company-name-modal">{news.company.name}</span>
            </div>
          </div>
        </div>

        <button className="back-btn-modal" onClick={onClose}>
          Back to News
        </button>
      </div>
    </div>
  );
}