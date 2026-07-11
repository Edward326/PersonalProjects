import React from 'react';
import { createProfileIcon } from '../../utils/userProfileUtils';
import '../../styles/HomePage.css';

export default function EventCard({ event }) {
  const formatDate = (date) => {
    return new Date(date).toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const creatorName = event.creator_name || `${event.first_name} ${event.last_name}` || 'Unknown';
  const creatorInitial = creatorName.charAt(0);
  const avatarColor = event.avatar_color || '#667eea';

  return (
    <div className="event-card">
      <div className="event-card-header">
        {createProfileIcon(creatorInitial, avatarColor, 32)}
        <span className="event-creator-name">{creatorName}</span>
      </div>
      
      <h3 className="event-card-title">{event.title}</h3>
      
      <p className="event-card-description">{event.description}</p>
      
      <div className="event-card-footer">
        {event.location && (
          <div className="event-location">
            <span className="event-icon">📍</span>
            <span>{event.location}</span>
          </div>
        )}
        <div className="event-date">
          <span className="event-icon">📅</span>
          <span>{formatDate(event.event_date)}</span>
        </div>
      </div>
    </div>
  );
}