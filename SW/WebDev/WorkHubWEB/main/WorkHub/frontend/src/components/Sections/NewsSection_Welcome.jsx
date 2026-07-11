import React, { useState, useEffect, useRef, useCallback } from 'react';
import NewsCard from '../Cards/NewsCard';
import NewsModal from '../Modals/NewsModal';
import '../../styles/WelcomePage.css';

export default function NewsSection({ onCompanyClick }) {
  const [news, setNews] = useState([]);
  const [selectedNews, setSelectedNews] = useState(null);
  const [loading, setLoading] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [page, setPage] = useState(0);
  const [dateFilter, setDateFilter] = useState('all');
  const observerTarget = useRef(null);
  const loadedIds = useRef(new Set()); // Track loaded news IDs to prevent duplicates

  const loadMore = useCallback(async () => {
    if (loading || !hasMore) return;
    setLoading(true);
    
    try {
      // Load 5 items per page
      const response = await fetch(`http://localhost:3000/api/news?limit=5&offset=${page * 5}`);
      const result = await response.json();
      
      if (result.success && result.data.length > 0) {
        // Filter out duplicates
        const newItems = result.data.filter(item => !loadedIds.current.has(item.id));
        
        if (newItems.length > 0) {
          // Add new IDs to tracking set
          newItems.forEach(item => loadedIds.current.add(item.id));
          
          setNews(prev => [...prev, ...newItems]);
          setPage(prev => prev + 1);
        }
        
        // Check if there are more items
        if (result.data.length < 5) {
          setHasMore(false);
        }
      } else {
        setHasMore(false);
      }
    } catch (error) {
      console.error('Error loading news:', error);
      setHasMore(false);
    } finally {
      setLoading(false);
    }
  }, [page, loading, hasMore]);

  // Initial load
  useEffect(() => {
    loadMore();
  }, []); // Only run once on mount

  useEffect(() => {
    const observer = new IntersectionObserver(
      entries => {
        if (entries[0].isIntersecting && !loading && hasMore) {
          loadMore();
        }
      },
      { threshold: 0.1 }
    );

    if (observerTarget.current) {
      observer.observe(observerTarget.current);
    }

    return () => {
      if (observerTarget.current) {
        observer.unobserve(observerTarget.current);
      }
    };
  }, [loadMore, loading, hasMore]);

  const filterNewsByDate = (item) => {
    if (dateFilter === 'all') return true;
    
    const itemDate = new Date(item.date || item.created_at);
    const now = new Date();
    const daysDiff = Math.floor((now - itemDate) / (1000 * 60 * 60 * 24));
    
    if (dateFilter === 'today') return daysDiff === 0;
    if (dateFilter === 'week') return daysDiff <= 7;
    if (dateFilter === 'month') return daysDiff <= 30;
    
    return true;
  };

  const filteredNews = news.filter(filterNewsByDate);

  return (
    <div className="news-section">
      <div className="news-header">
        <h2>Latest News</h2>
        <div className="date-filters">
          <button 
            className={`filter-btn ${dateFilter === 'all' ? 'active' : ''}`}
            onClick={() => setDateFilter('all')}
          >
            All
          </button>
          <button 
            className={`filter-btn ${dateFilter === 'today' ? 'active' : ''}`}
            onClick={() => setDateFilter('today')}
          >
            Today
          </button>
          <button 
            className={`filter-btn ${dateFilter === 'week' ? 'active' : ''}`}
            onClick={() => setDateFilter('week')}
          >
            This Week
          </button>
          <button 
            className={`filter-btn ${dateFilter === 'month' ? 'active' : ''}`}
            onClick={() => setDateFilter('month')}
          >
            This Month
          </button>
        </div>
      </div>

      <div className="news-grid">
        {filteredNews.map((item) => (
          <NewsCard 
            key={item.id} 
            news={item} 
            onClick={() => setSelectedNews(item)} 
          />
        ))}
      </div>

      {loading && <div className="loading">Loading more news...</div>}
      {!hasMore && news.length > 0 && <div className="no-more-news">No more news to load</div>}
      
      <div ref={observerTarget} className="observer-target"></div>

      {selectedNews && (
        <NewsModal 
          news={selectedNews} 
          onClose={() => setSelectedNews(null)}
          onCompanyClick={onCompanyClick}
        />
      )}
    </div>
  );
}