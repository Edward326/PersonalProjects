import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Header from '../Shared/Header/Header_Custom';
import Sidebar_Organization from '../Shared/Sidebar/Sidebar_Organization';
import OrganizationTreeSection from '../Sections/OrganizationTreeSection';
import '../../styles/OrganizationPage.css';
import '../../styles/Notifications.css';

export default function OrganizationPage() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [notification, setNotification] = useState({ show: false, message: '', type: '' });
  const [userData, setUserData] = useState(null);
  const [departmentData, setDepartmentData] = useState(null);
  const [roleData, setRoleData] = useState(null);
  const [organizationData, setOrganizationData] = useState(null);
  const [sidebarExpanded, setSidebarExpanded] = useState(false);

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

  const handleLogout = async () => {
    try{
      if(userData!=null){
     await fetch(`http://localhost:3000/api/attendance/clock-out`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
              user_id: userData.id
            })
          });
    localStorage.removeItem('isAuthenticated');
    localStorage.removeItem('authData');
    navigate('/welcome');
    }
    else
    {
    localStorage.removeItem('isAuthenticated');
    localStorage.removeItem('authData');
    navigate('/welcome');
    }
    }catch (error) {
      console.error('Error fetching data:', error);
      showNotification('Failed to load data. Logging out...', 'error');
      setTimeout(() => handleLogout(), 5000);
    }
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
        setUserData({ 
          ...authData, 
          name: orgResult.data.name, 
          email: orgResult.data.ceo_email,
          organization_id: organizationId 
        });
        
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
    <div className="organization-page-container">
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
          userData={userData}
        />
      )}

      <div className={`organization-page-body ${!hasHeader ? 'no-header' : ''}`}>
        <Sidebar_Organization 
          sidebarExpanded={sidebarExpanded}
          setSidebarExpanded={setSidebarExpanded}
          sidebarPosition={sidebarPosition}
          isCEO={isCEO}
          colorConfig={{
            hover: departmentData?.layout_color_hover || '#4a60c172',
            clicked: departmentData?.layout_color_clicked || 'rgba(29, 37, 78, 1)',
            selected: departmentData?.layout_color_selected || '#4a5fc1'
          }}
          onLogout={handleLogout}
          onNavigate={(path) => navigate(path)}
        />

        <main className={`main-content ${sidebarPosition === 'right' ? 'sidebar-right' : ''}`}>
          <div className="content-wrapper">
            <OrganizationTreeSection 
              organizationId={userData.organization_id}
              userId={userData.id}
              isCEO={isCEO}
              canViewStatistics={isCEO || (roleData && roleData.can_view_statistics)}
              onError={showNotification}
            />
          </div>
        </main>
      </div>
    </div>
  );
}