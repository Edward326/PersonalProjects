import React, { useEffect, useState } from 'react';
import { Navigate } from 'react-router-dom';

export default function ProtectedRoute({ children }) {
  const [isChecking, setIsChecking] = useState(true);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  useEffect(() => {
    // Check if user is authenticated
    const authStatus = localStorage.getItem('isAuthenticated');
    const authData = localStorage.getItem('authData');
    
    if (authStatus === 'true' && authData) {
      try {
        const parsed = JSON.parse(authData);
        if (parsed.userId) {
          setIsAuthenticated(true);
        }
      } catch (error) {
        console.error('Invalid auth data:', error);
        localStorage.removeItem('isAuthenticated');
        localStorage.removeItem('authData');
      }
    }
    
    setIsChecking(false);
  }, []);

  if (isChecking) {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: '100vh',
        fontSize: '18px',
        color: '#666'
      }}>
        Loading...
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/welcome" replace />;
  }

  return children;
}