import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import Header from '../Shared/Header/Header_Custom';
import Sidebar from '../Shared/Sidebar/Sidebar_Home';
import NewsSection from '../Sections/NewsSection_Home';
import EventsSection from '../Sections/EventsSection';
import '../../styles/HomePage.css';
import '../../styles/Notifications.css';

export default function HomePage() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [notification, setNotification] = useState({ show: false, message: '', type: '' });
  const [userData, setUserData] = useState(null);
  const [departmentData, setDepartmentData] = useState(null);
  const [roleData, setRoleData] = useState(null);
  const [organizationData, setOrganizationData] = useState(null);
  const [activeSection, setActiveSection] = useState('news');
  const [sidebarExpanded, setSidebarExpanded] = useState(false);
  const mainContentRef = useRef(null);

  // Auto-hide notification
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

  const handleLogout = () => {
    localStorage.removeItem('isAuthenticated');
    localStorage.removeItem('authData');
    navigate('/welcome');
  };

  // Fetch all necessary data on mount
  useEffect(() => {
    fetchAllData();
  }, []);

  const fetchAllData = async () => {
    try {
      setLoading(true);
      
      // Get auth data from localStorage
      const authDataString = localStorage.getItem('authData');
      if (!authDataString) {
        showNotification('Session expired. Please log in again.', 'error');
        setTimeout(() => handleLogout(), 5000);
        return;
      }

      const authData = JSON.parse(authDataString);
      const { userId, userType, departmentId, roleId, organizationId } = authData;

      // Fetch user data based on userType
      if (userType === 'ceo') {
        // Fetch organization data for CEO
        const orgResponse = await fetch(`http://localhost:3000/api/organizations/${userId}`);
        const orgResult = await orgResponse.json();

        if (!orgResult.success) {
          throw new Error('Failed to fetch organization data');
        }

        setOrganizationData(orgResult.data);
        setUserData({ ...authData, name: orgResult.data.name, email: orgResult.data.ceo_email });
        
      } else {
        // Fetch employee data
        const userResponse = await fetch(`http://localhost:3000/api/users/${userId}`);
        const userResult = await userResponse.json();

        if (!userResult.success) {
          throw new Error('Failed to fetch user data');
        }

        setUserData(userResult.data);

        // Fetch department data
        const deptResponse = await fetch(`http://localhost:3000/api/departments/${departmentId}`);
        const deptResult = await deptResponse.json();

        if (!deptResult.success) {
          throw new Error('Failed to fetch department data');
        }

        setDepartmentData(deptResult.data);

        // Fetch role data
        const roleResponse = await fetch(`http://localhost:3000/api/roles/${roleId}`);
        const roleResult = await roleResponse.json();

        if (!roleResult.success) {
          throw new Error('Failed to fetch role data');
        }

        setRoleData(roleResult.data);

        // Fetch organization data
        const orgResponse = await fetch(`http://localhost:3000/api/organizations/${organizationId}`);
        const orgResult = await orgResponse.json();

        if (!orgResult.success) {
          throw new Error('Failed to fetch organization data');
        }

        setOrganizationData(orgResult.data);
      }

      setLoading(false);

    } catch (error) {
      console.error('Error fetching data:', error);
      showNotification('Failed to load data. Logging out...', 'error');
      setTimeout(() => handleLogout(), 5000);
    }
  };

  const renderContent = () => {
    if (!userData) return null;

    const isCEO = userData.userType === 'ceo';
    const canPostNews = !isCEO || (roleData && roleData.can_post_news);
    const canPostEvent = !isCEO || (roleData && roleData.can_post_event);

    switch(activeSection) {
      case 'news':
        return (
          <NewsSection 
            canPost={canPostNews}
            organizationId={isCEO?organizationData.id:userData.organization_id}
            userId={userData.id}
            isCEO={isCEO}
            sidebarPosition={departmentData?.sidebar_position || 'left'}
          />
        );
      case 'events':
        return (
          <EventsSection 
            canPost={canPostEvent}
            organizationId={isCEO?organizationData.id:userData.organization_id}
            departmentId={userData.department_id}
            userId={userData.id}
            isCEO={isCEO}
            sidebarPosition={departmentData?.sidebar_position || 'left'}
          />
        );
      case 'requests':
        navigate('/requests');
        return null;
      case 'assignments':
        navigate('/assignments');
        return null;
      case 'organization':
        navigate('/organization');
        return null;
      case 'statistics':
        navigate('/statistics');
        return null;
      case 'settings':
        navigate('/settings');
        return null;
      default:
        return (
          <NewsSection 
            canPost={canPostNews}
            organizationId={userData.organization_id}
            userId={userData.id}
            isCEO={isCEO}
            sidebarPosition={departmentData?.sidebar_position || 'left'}
          />
        );
    }
  };

  if (loading) {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <p>Loading...</p>
      </div>
    );
  }

  const isCEO = userData?.userType === 'ceo';
  const hasHeader = isCEO || departmentData?.has_header !== false;
  const headerConfig = departmentData || {};
  const sidebarPosition = departmentData?.sidebar_position || 'left';

  return (
    <div className="home-page-container">
      {notification.show && (
        <div className={`notification ${notification.type}`}>
          {notification.message}
        </div>
      )}

      {hasHeader && (
        <Header 
          config={headerConfig}
          organizationData={organizationData}
          isCEO={isCEO}
        />
      )}

      <div className={`home-page-body ${!hasHeader ? 'no-header' : ''}`}>
        <Sidebar 
          activeSection={activeSection}
          setActiveSection={setActiveSection}
          sidebarExpanded={sidebarExpanded}
          setSidebarExpanded={setSidebarExpanded}
          sidebarPosition={sidebarPosition}
          roleData={roleData}
          isCEO={isCEO}
          colorConfig={{
            hover: departmentData?.layout_color_hover || '#4a60c172',
            clicked: departmentData?.layout_color_clicked || 'rgba(29, 37, 78, 1)',
            selected: departmentData?.layout_color_selected || '#4a5fc1'
          }}
          onLogout={handleLogout}
        />

        <main className={`main-content ${sidebarPosition === 'right' ? 'sidebar-right' : ''}`} ref={mainContentRef}>
          <div className="content-wrapper">
            {renderContent()}
          </div>
        </main>
      </div>
    </div>
  );
}