import React, { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import Header from '../Shared/Header/Header_Custom';
import Sidebar_Statistics from '../Shared/Sidebar/Sidebar_Statistics';
import OrganizationStatisticsSection from '../Sections/OrganizationStatisticsSection';
import EmployeeStatisticsSection from '../Sections/EmployeeStatisticsSection';
import TasksStatisticsSection from '../Sections/TasksStatisticsSection';
import '../../styles/StatisticsPage.css';
import '../../styles/Notifications.css';

export default function StatisticsPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [loading, setLoading] = useState(true);
  const [notification, setNotification] = useState({ show: false, message: '', type: '' });
  const [userData, setUserData] = useState(null);
  const [departmentData, setDepartmentData] = useState(null);
  const [roleData, setRoleData] = useState(null);
  const [organizationData, setOrganizationData] = useState(null);
  const [activeSection, setActiveSection] = useState('organization');
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

  // Check URL parameters to determine initial section
  useEffect(() => {
    const userId = searchParams.get('userId');
    if (userId) {
      setActiveSection('employee');
    }
  }, [searchParams]);

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

  const renderContent = () => {
    if (!userData) return null;

    const isCEO = userData.userType === 'ceo';
    const canViewStatistics = isCEO || (roleData && roleData.can_view_statistics);
    const canAssign = isCEO || (roleData && roleData.can_assign_tasks);

    switch(activeSection) {
      case 'organization':
        return (
          <OrganizationStatisticsSection 
            organizationId={userData.organization_id}
            onError={showNotification}
          />
        );
      case 'employee':
        return (
          <EmployeeStatisticsSection 
            organizationId={userData.organization_id}
            isCEO={isCEO}
            canViewStatistics={canViewStatistics}
            onError={showNotification}
          />
        );
      case 'tasks':
        return (
          <TasksStatisticsSection 
            organizationId={userData.organization_id}
            departmentId={userData.department_id}
            isCEO={isCEO}
            canViewStatistics={canViewStatistics}
            canAssign={canAssign}
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
        return (
          <OrganizationStatisticsSection 
            organizationId={userData.organization_id}
            onError={showNotification}
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
    <div className="statistics-page-container">
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

      <div className={`statistics-page-body ${!hasHeader ? 'no-header' : ''}`}>
        <Sidebar_Statistics 
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
          onNavigate={(path) => navigate(path)}
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