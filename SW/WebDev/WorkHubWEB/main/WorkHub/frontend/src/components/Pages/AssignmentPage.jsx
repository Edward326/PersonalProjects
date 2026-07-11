import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Header from '../Shared/Header/Header_Custom';
import Sidebar_Assignments from '../Shared/Sidebar/Sidebar_Assignments';
import AssignmentsSection from '../Sections/AssignmentsSection';
import AssignmentAdminSection from '../Sections/AssignmentAdminSection';
import '../../styles/AssignmentPage.css';
import '../../styles/Notifications.css';

export default function AssignmentPage() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [notification, setNotification] = useState({ show: false, message: '', type: '' });
  const [userData, setUserData] = useState(null);
  const [departmentData, setDepartmentData] = useState(null);
  const [roleData, setRoleData] = useState(null);
  const [organizationData, setOrganizationData] = useState(null);
  const [activeSection, setActiveSection] = useState('assignments');
  const [sidebarExpanded, setSidebarExpanded] = useState(false);

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

  useEffect(() => {
    fetchAllData();
  }, []);

  const fetchAllData = async () => {
    try {
      setLoading(true);
      
      const authDataString = localStorage.getItem('authData');
      if (!authDataString) {
        showNotification('Session expired. Please log in again.', 'error');
        setTimeout(() => handleLogout(), 2000);
        return;
      }

      const authData = JSON.parse(authDataString);
      const { userId, userType, departmentId, roleId, organizationId } = authData;

      if (userType === 'ceo') {
        showNotification('This page is only available for employees', 'error');
        setTimeout(() => navigate('/'), 2000);
        return;
      }

      // Fetch user data
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

      // Check permissions
      if (!roleResult.data.can_receive_tasks && !roleResult.data.can_assign_tasks) {
        showNotification('You do not have permission to access this page', 'error');
        setTimeout(() => navigate('/'), 2000);
        return;
      }

      // Fetch organization data
      const orgResponse = await fetch(`http://localhost:3000/api/organizations/${organizationId}`);
      const orgResult = await orgResponse.json();

      if (!orgResult.success) {
        throw new Error('Failed to fetch organization data');
      }

      setOrganizationData(orgResult.data);

      // Set default section based on permissions
      if (roleResult.data.can_receive_tasks) {
        setActiveSection('assignments');
      } else if (roleResult.data.can_assign_tasks) {
        setActiveSection('admin');
      }

      setLoading(false);

    } catch (error) {
      console.error('Error fetching data:', error);
      showNotification('Failed to load data. Logging out...', 'error');
      setTimeout(() => handleLogout(), 2000);
    }
  };

  const renderContent = () => {
    if (!userData || !roleData) return null;

    switch(activeSection) {
      case 'assignments':
        return (
          <AssignmentsSection 
            userId={userData.id}
            departmentId={userData.department_id}
            onError={showNotification}
          />
        );
      case 'admin':
        return (
          <AssignmentAdminSection 
            userId={userData.id}
            departmentId={userData.department_id}
            organizationId={userData.organization_id}
            onError={showNotification}
          />
        );
      case 'home':
        navigate('/');
        return null;
      case 'settings':
        navigate('/settings');
        return null;
      default:
        return null;
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

  const hasHeader = departmentData?.has_header !== false;
  const headerConfig = departmentData || {};
  const sidebarPosition = departmentData?.sidebar_position || 'left';

  return (
    <div className="assignment-page-container">
      {notification.show && (
        <div className={`notification ${notification.type}`}>
          {notification.message}
        </div>
      )}

      {hasHeader && (
        <Header 
          config={headerConfig}
          organizationData={organizationData}
          isCEO={false}
          userData={userData}
        />
      )}

      <div className={`assignment-page-body ${!hasHeader ? 'no-header' : ''}`}>
        <Sidebar_Assignments 
          activeSection={activeSection}
          setActiveSection={setActiveSection}
          sidebarExpanded={sidebarExpanded}
          setSidebarExpanded={setSidebarExpanded}
          sidebarPosition={sidebarPosition}
          roleData={roleData}
          colorConfig={{
            hover: departmentData?.layout_color_hover || '#4a60c172',
            clicked: departmentData?.layout_color_clicked || 'rgba(29, 37, 78, 1)',
            selected: departmentData?.layout_color_selected || '#4a5fc1'
          }}
          onLogout={handleLogout}
        />

        <main className={`main-content ${sidebarPosition === 'right' ? 'sidebar-right' : ''}`}>
          <div className="content-wrapper">
            {renderContent()}
          </div>
        </main>
      </div>
    </div>
  );
}