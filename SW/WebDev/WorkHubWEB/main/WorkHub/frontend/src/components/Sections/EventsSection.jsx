import React, { useState, useEffect } from 'react';
import EventCard from '../Cards/EventCard';
import deleteIcon from '../../assets/Icons/delete.png';
import addIcon from '../../assets/Icons/plus.png';
import '../../styles/HomePage.css';
import '../../styles/Notifications.css';

export default function EventsSection({ 
  canPost, 
  organizationId,
  departmentId,
  userId, 
  isCEO,
  sidebarPosition 
}) {
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showCreatePopup, setShowCreatePopup] = useState(false);
  const [showDeletePopup, setShowDeletePopup] = useState(false);
  const [notification, setNotification] = useState({ show: false, message: '', type: '' });
  
  // Form state
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [eventDate, setEventDate] = useState('');
  const [location, setLocation] = useState('');
  
  // Delete state
  const [departmentEvents, setDepartmentEvents] = useState([]);
  const [selectedEventIds, setSelectedEventIds] = useState([]);

  useEffect(() => {
    fetchEvents();
  }, [organizationId, departmentId, isCEO]);

  useEffect(() => {
    if (notification.show) {
      const timer = setTimeout(() => {
        setNotification({ show: false, message: '', type: '' });
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [notification.show]);

  const showNotification = (message, type = 'error') => {
    setNotification({ show: true, message, type });
  };

  const fetchEvents = async () => {
    try {
      setLoading(true);
      let url;
      
      if (isCEO) {
        // CEO sees all events across all departments
        url = `http://localhost:3000/api/events/organization/${organizationId}`;
      } else {
        // Employee sees only department events
        url = `http://localhost:3000/api/events/department/${departmentId}`;
      }

      const response = await fetch(url);
      const result = await response.json();
      
      if (result.success) {
        setEvents(result.data);
      } else {
        showNotification('Failed to load events', 'error');
      }
    } catch (error) {
      console.error('Error fetching events:', error);
      showNotification('Failed to load events', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleCreateEvent = async () => {
    if (!title.trim() || !description.trim() || !eventDate || !location.trim()) {
      showNotification('All fields are required', 'error');
      return;
    }

    try {
      const response = await fetch('http://localhost:3000/api/events', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: title.trim(),
          description: description.trim(),
          event_date: eventDate,
          location: location.trim(),
          is_personal: false,
          created_by: userId,
          department_id: departmentId,
          organization_id: organizationId
        })
      });

      const result = await response.json();

      if (result.success) {
        showNotification('Event created successfully!', 'success');
        setShowCreatePopup(false);
        setTitle('');
        setDescription('');
        setEventDate('');
        setLocation('');
        fetchEvents(); // Refresh events list
      } else {
        showNotification('Failed to create event', 'error');
      }
    } catch (error) {
      console.error('Error creating event:', error);
      showNotification('Failed to create event', 'error');
    }
  };

  const handleOpenDeletePopup = async () => {
    try {
      let url;
      
      if (isCEO) {
        url = `http://localhost:3000/api/events/organization/${organizationId}`;
      } else {
        url = `http://localhost:3000/api/events/department/${departmentId}`;
      }

      const response = await fetch(url);
      const result = await response.json();
      
      if (result.success) {
        setDepartmentEvents(result.data);
        setShowDeletePopup(true);
      } else {
        showNotification('Failed to load events for deletion', 'error');
      }
    } catch (error) {
      console.error('Error fetching events:', error);
      showNotification('Failed to load events for deletion', 'error');
    }
  };

  const handleDeleteEvents = async () => {
    if (selectedEventIds.length === 0) {
      showNotification('Please select at least one event to delete', 'warning');
      return;
    }

    try {
      const deletePromises = selectedEventIds.map(id =>
        fetch(`http://localhost:3000/api/events/${id}`, { method: 'DELETE' })
      );

      await Promise.all(deletePromises);
      
      showNotification(`Successfully deleted ${selectedEventIds.length} event(s)`, 'success');
      setShowDeletePopup(false);
      setSelectedEventIds([]);
      fetchEvents(); // Refresh events list
    } catch (error) {
      console.error('Error deleting events:', error);
      showNotification('Failed to delete events', 'error');
    }
  };

  const toggleEventSelection = (eventId) => {
    setSelectedEventIds(prev => 
      prev.includes(eventId) 
        ? prev.filter(id => id !== eventId)
        : [...prev, eventId]
    );
  };

  if (loading) {
    return (
      <div className="section-loading">
        <div className="loading-spinner"></div>
        <p>Loading events...</p>
      </div>
    );
  }

  return (
    <div className="events-section">
      {notification.show && (
        <div className={`section-notification ${notification.type}`}>
          {notification.message}
        </div>
      )}

      {/* Action Buttons */}
      {canPost && (
        <div className={`action-buttons ${sidebarPosition === 'right' ? 'buttons-left' : 'buttons-right'}`}>
          <button 
            className="action-button-icon plus-button" 
            onClick={() => setShowCreatePopup(true)}
            title="Create Event"
          >
             <img src={addIcon} alt="Add" />
          </button>
          <button 
            className="action-button-icon delete-button" 
            onClick={handleOpenDeletePopup}
            title="Delete Event"
          >
            <img src={deleteIcon} alt="Delete" />
          </button>
        </div>
      )}

      {/* Create Event Popup */}
      {showCreatePopup && (
        <div className="config-popup-overlay">
          <div className="config-popup">
            <h2>Event Configuration</h2>
            <div className="config-form">
              <div className="form-group">
                <label>Title *</label>
                <input 
                  type="text" 
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="Enter event title"
                />
              </div>
              <div className="form-group">
                <label>Description *</label>
                <textarea 
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="Enter event description"
                  rows="4"
                />
              </div>
              <div className="form-group">
                <label>Event Date *</label>
                <input 
                  type="datetime-local" 
                  value={eventDate}
                  onChange={(e) => setEventDate(e.target.value)}
                />
              </div>
              <div className="form-group">
                <label>Location *</label>
                <input 
                  type="text" 
                  value={location}
                  onChange={(e) => setLocation(e.target.value)}
                  placeholder="Enter event location"
                />
              </div>
            </div>
            <div className="config-buttons">
              <button className="config-cancel-button" onClick={() => setShowCreatePopup(false)}>
                Cancel
              </button>
              <button className="config-done-button" onClick={handleCreateEvent}>
                Done
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Delete Event Popup */}
      {showDeletePopup && (
        <div className="config-popup-overlay">
          <div className="config-popup delete-popup">
            <h2>Delete Events</h2>
            <p className="delete-instruction">Select events to delete:</p>
            <div className="delete-list">
              {departmentEvents.length === 0 ? (
                <p className="no-items">No events found</p>
              ) : (
                departmentEvents.map(event => (
                  <div 
                    key={event.id} 
                    className={`delete-item ${selectedEventIds.includes(event.id) ? 'selected' : ''}`}
                    onClick={() => toggleEventSelection(event.id)}
                  >
                    <input 
                      type="checkbox" 
                      checked={selectedEventIds.includes(event.id)}
                      onChange={() => {}}
                    />
                    <div className="delete-item-info">
                      <h4>{event.title}</h4>
                      <p>{new Date(event.event_date).toLocaleDateString()}</p>
                    </div>
                  </div>
                ))
              )}
            </div>
            <div className="config-buttons">
              <button className="config-cancel-button" onClick={() => {
                setShowDeletePopup(false);
                setSelectedEventIds([]);
              }}>
                Cancel
              </button>
              <button 
                className="config-delete-button" 
                onClick={handleDeleteEvents}
                disabled={selectedEventIds.length === 0}
              >
                Delete ({selectedEventIds.length})
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Events List */}
      <div className="events-list">
        {events.length === 0 ? (
          <div className="no-events">
            <p>No events scheduled yet.</p>
            {canPost && <p>Be the first to create one!</p>}
          </div>
        ) : (
          events.map(event => (
            <EventCard key={event.id} event={event} />
          ))
        )}
      </div>
    </div>
  );
}