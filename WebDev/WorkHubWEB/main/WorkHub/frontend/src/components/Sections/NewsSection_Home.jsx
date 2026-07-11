import React, { useState, useEffect } from 'react';
import NewsCard from '../Cards/NewsCard';
import deleteIcon from '../../assets/Icons/delete.png';
import addIcon from '../../assets/Icons/plus.png';
import NewsModal from '../Modals/NewsModal';
import '../../styles/HomePage.css';
import '../../styles/Notifications.css';

export default function NewsSection_Home({ 
  canPost, 
  organizationId, 
  userId, 
  isCEO,
  sidebarPosition 
}) {
  const [news, setNews] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showCreatePopup, setShowCreatePopup] = useState(false);
  const [showDeletePopup, setShowDeletePopup] = useState(false);
  const [notification, setNotification] = useState({ show: false, message: '', type: '' });
  const [selectedNews, setSelectedNews] = useState(null);
  const handleCompanyClick = (companyName, companyId) => {
  };
  
  // Form state
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [imageUrl, setImageUrl] = useState('');
  
  // Delete state
  const [organizationNews, setOrganizationNews] = useState([]);
  const [selectedNewsIds, setSelectedNewsIds] = useState([]);

  useEffect(() => {
    fetchNews();
  }, [organizationId]);

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

  const fetchNews = async () => {
    try {
      setLoading(true);
      const response = await fetch(`http://localhost:3000/api/news`);
      const result = await response.json();
      
      if (result.success) {
        setNews(result.data);
      } else {
        showNotification('Failed to load news', 'error');
      }
    } catch (error) {
      console.error('Error fetching news:', error);
      showNotification('Failed to load news', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleCreateNews = async () => {
    if (!title.trim() || !content.trim()) {
      showNotification('Title and content are required', 'error');
      return;
    }

    try {
      const response = await fetch('http://localhost:3000/api/news', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          organization_id: organizationId,
          author_id: userId,
          title: title.trim(),
          content: content.trim(),
          image_url: imageUrl.trim() || null,
          is_public: true,
          isCEO
        })
      });

      const result = await response.json();

      if (result.success) {
        showNotification('News posted successfully!', 'success');
        setShowCreatePopup(false);
        setTitle('');
        setContent('');
        setImageUrl('');
        fetchNews(); // Refresh news list
      } else {
        showNotification('Failed to create news', 'error');
      }
    } catch (error) {
      console.error('Error creating news:', error);
      showNotification('Failed to create news', 'error');
    }
  };

  const handleOpenDeletePopup = async () => {
    try {
      const response = await fetch(`http://localhost:3000/api/news/organization/${organizationId}`);
      const result = await response.json();
      
      if (result.success) {
        setOrganizationNews(result.data);
        setShowDeletePopup(true);
      } else {
        showNotification('Failed to load news for deletion', 'error');
      }
    } catch (error) {
      console.error('Error fetching news:', error);
      showNotification('Failed to load news for deletion', 'error');
    }
  };

  const handleDeleteNews = async () => {
    if (selectedNewsIds.length === 0) {
      showNotification('Please select at least one news item to delete', 'warning');
      return;
    }

    try {
      const deletePromises = selectedNewsIds.map(id =>
        fetch(`http://localhost:3000/api/news/${id}`, { method: 'DELETE' })
      );

      await Promise.all(deletePromises);
      
      showNotification(`Successfully deleted ${selectedNewsIds.length} news item(s)`, 'success');
      setShowDeletePopup(false);
      setSelectedNewsIds([]);
      fetchNews(); // Refresh news list
    } catch (error) {
      console.error('Error deleting news:', error);
      showNotification('Failed to delete news', 'error');
    }
  };

  const toggleNewsSelection = (newsId) => {
    setSelectedNewsIds(prev => 
      prev.includes(newsId) 
        ? prev.filter(id => id !== newsId)
        : [...prev, newsId]
    );
  };

  if (loading) {
    return (
      <div className="section-loading">
        <div className="loading-spinner"></div>
        <p>Loading news...</p>
      </div>
    );
  }

  return (
    <div className="news-section">
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
            title="Create News"
          >
            <img src={addIcon} alt="Add" />
          </button>
          <button 
            className="action-button-icon delete-button" 
            onClick={handleOpenDeletePopup}
            title="Delete News"
          >
            <img src={deleteIcon} alt="Delete" />
          </button>
        </div>
      )}

      {/* Create News Popup */}
      {showCreatePopup && (
        <div className="config-popup-overlay">
          <div className="config-popup">
            <h2>News Configuration</h2>
            <div className="config-form">
              <div className="form-group">
                <label>Title *</label>
                <input 
                  type="text" 
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="Enter news title"
                />
              </div>
              <div className="form-group">
                <label>Content *</label>
                <textarea 
                  value={content}
                  onChange={(e) => setContent(e.target.value)}
                  placeholder="Enter news content"
                  rows="6"
                />
              </div>
              <div className="form-group">
                <label>Image URL (Optional)</label>
                <input 
                  type="text" 
                  value={imageUrl}
                  onChange={(e) => setImageUrl(e.target.value)}
                  placeholder="https://example.com/image.jpg"
                />
              </div>
            </div>
            <div className="config-buttons">
              <button className="config-cancel-button" onClick={() => setShowCreatePopup(false)}>
                Cancel
              </button>
              <button className="config-done-button" onClick={handleCreateNews}>
                Done
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Delete News Popup */}
      {showDeletePopup && (
        <div className="config-popup-overlay">
          <div className="config-popup delete-popup">
            <h2>Delete News</h2>
            <p className="delete-instruction">Select news items to delete:</p>
            <div className="delete-list">
              {organizationNews.length === 0 ? (
                <p className="no-items">No news items found</p>
              ) : (
                organizationNews.map(newsItem => (
                  <div 
                    key={newsItem.id} 
                    className={`delete-item ${selectedNewsIds.includes(newsItem.id) ? 'selected' : ''}`}
                    onClick={() => toggleNewsSelection(newsItem.id)}
                  >
                    <input 
                      type="checkbox" 
                      checked={selectedNewsIds.includes(newsItem.id)}
                      onChange={() => {}}
                    />
                    <div className="delete-item-info">
                      <h4>{newsItem.title}</h4>
                      <p>{new Date(newsItem.created_at).toLocaleDateString()}</p>
                    </div>
                  </div>
                ))
              )}
            </div>
            <div className="config-buttons">
              <button className="config-cancel-button" onClick={() => {
                setShowDeletePopup(false);
                setSelectedNewsIds([]);
              }}>
                Cancel
              </button>
              <button 
                className="config-delete-button" 
                onClick={handleDeleteNews}
                disabled={selectedNewsIds.length === 0}
              >
                Delete ({selectedNewsIds.length})
              </button>
            </div>
          </div>
        </div>
      )}

      {/* News List */}
      <div className="news-list">
        {news.length === 0 ? (
          <div className="no-news">
            <p>No news available yet.</p>
            {canPost && <p>Be the first to post!</p>}
          </div>
        ) : (
          news.map(newsItem => (
            <NewsCard 
            key={newsItem.id} 
            news={newsItem} 
            onClick={() => setSelectedNews(newsItem)} 
            />
          ))
        )}
      </div>

        {selectedNews && (
        <NewsModal 
          news={selectedNews} 
          onClose={() => setSelectedNews(null)}
        />
      )}
    </div>
  );
}