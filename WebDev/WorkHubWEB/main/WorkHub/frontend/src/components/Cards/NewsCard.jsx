import React from 'react';
import { createProfileIcon } from '../../utils/userProfileUtils';

export default function NewsCard({ news, onClick }) {
  const formatDate = (date) => {
    return new Date(date).toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric' 
    });
  };

  const hasImage = news.image_url && news.image_url.trim() !== '';
  const preview = news.content.substring(0, 150) + '...';

  return (
    <div className="news-card" onClick={onClick}>
      <div className={`news-card-layout ${hasImage ? 'with-image' : 'no-image'}`}>
        {hasImage && (
          <div className="news-card-image">
            <img src={news.image_url} alt={news.title} />
          </div>
        )}
        
        <div className="news-card-content">
          <div className="news-card-author">
            {createProfileIcon(news.author.name.split(' ')[0], news.author.avatarColor, 24)}
            <span className="author-name">{news.author.name}</span>
          </div>
          
          <h3 className="news-card-title">{news.title}</h3>
          <p className="news-card-preview">{preview}</p>
          
          <div className="news-card-footer">
            <div className="company-info">
              <span className="news-company">{news.company.name}</span>
            </div>
            <span className="news-date">{formatDate(news.date || news.created_at)}</span>
          </div>
        </div>
      </div>
    </div>
  );
}