import React, { useState } from 'react';
import organizationIcon from '../../../assets/Icons/organization_choose.png';
import employeeIcon from '../../../assets/icons/employee_choose.png';
import tasksIcon from '../../../assets/Icons/clipboard.png';
import homeIcon from '../../../assets/Icons/home.png';
import settingsIcon from '../../../assets/Icons/setting.png';
import logoutIcon from '../../../assets/Icons/logout.png';
import '../../../styles/StatisticsPage.css';
import '../../../styles/Notifications.css';

export default function Sidebar_Statistics({ 
  activeSection, 
  setActiveSection, 
  sidebarExpanded, 
  setSidebarExpanded,
  sidebarPosition = 'left',
  roleData,
  isCEO,
  colorConfig,
  onLogout,
  onNavigate
}) {
  const [isHovered, setIsHovered] = useState(false);
  const [showLogoutConfirm, setShowLogoutConfirm] = useState(false);

  const isOpen = sidebarExpanded || isHovered;

  const toggleSidebar = () => {
    setSidebarExpanded(!sidebarExpanded);
  };

  // Determine which buttons to show based on role
  const showOrganization = isCEO || (roleData && roleData.can_view_statistics);
  const showEmployee = isCEO || (roleData && roleData.can_view_statistics);
  const showTasks = isCEO || (roleData && (roleData.can_view_statistics || roleData.can_assign_tasks));

  // Category 1: Statistics Navigation (Top)
  const topNavItems = [];
  if (showOrganization) {
    topNavItems.push({ id: 'organization', label: 'Organization', icon: organizationIcon });
  }
  if (showEmployee) {
    topNavItems.push({ id: 'employee', label: 'Employee', icon: employeeIcon });
  }
  if (showTasks) {
    topNavItems.push({ id: 'tasks', label: 'Tasks', icon: tasksIcon });
  }

  // Divider between top and bottom sections
  const hasDivider = topNavItems.length > 0;

  // Category 2: Navigation (Bottom)
  const bottomNavItems = [
    { id: 'home', label: 'Home', icon: homeIcon },
    { id: 'settings', label: 'Settings', icon: settingsIcon },
  ];

  const handleLogoutClick = () => {
    setShowLogoutConfirm(true);
  };

  const confirmLogout = () => {
    setShowLogoutConfirm(false);
    onLogout();
  };

  const cancelLogout = () => {
    setShowLogoutConfirm(false);
  };

  const handleNavClick = (id) => {
    if (id === 'home') {
      onNavigate('/');
    } else if (id === 'settings') {
      onNavigate('/settings');
    }
  };

  // Use default colors if CEO, otherwise use department colors
  const navColorConfig = isCEO ? {
    hover: '#4a60c172',
    clicked: 'rgba(29, 37, 78, 1)',
    selected: '#4a5fc1'
  } : colorConfig;

  return (
    <>
      {/* Logout Confirmation Popup */}
      {showLogoutConfirm && (
        <div className="logout-popup-overlay">
          <div className="logout-popup">
            <h2>Confirm Logout</h2>
            <p>Are you sure you want to log out?</p>
            <div className="logout-buttons">
              <button className="logout-cancel-button" onClick={cancelLogout}>
                Cancel
              </button>
              <button className="logout-confirm-button" onClick={confirmLogout}>
                Log Out
              </button>
            </div>
          </div>
        </div>
      )}

      <aside 
        className={`statistics-sidebar ${isOpen ? 'expanded' : 'collapsed'} ${sidebarPosition === 'right' ? 'sidebar-right' : ''}`}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      >
        <div className="statistics-sidebar-toggle-handle" onClick={toggleSidebar}>
          <span className="statistics-toggle-arrow">
            {sidebarPosition === 'left' ? (isOpen ? '◀' : '▶') : (isOpen ? '▶' : '◀')}
          </span>
        </div>

        {/* Category 1: Top Navigation */}
        <nav className="statistics-sidebar-nav statistics-sidebar-top">
          {topNavItems.map((item) => (
            <NavButton
              key={item.id}
              item={item}
              isActive={activeSection === item.id}
              onClick={() => setActiveSection(item.id)}
              colorConfig={navColorConfig}
            />
          ))}
        </nav>

        {/* Divider */}
        {hasDivider && <div className="statistics-sidebar-divider"></div>}

        {/* Category 2: Bottom Navigation */}
        <nav className="statistics-sidebar-nav statistics-sidebar-bottom">
          {bottomNavItems.map((item) => (
            <NavButton
              key={item.id}
              item={item}
              isActive={false}
              onClick={() => handleNavClick(item.id)}
              colorConfig={navColorConfig}
            />
          ))}
          
          {/* Logout Button */}
          <NavButton
            item={{ id: 'logout', label: 'Log Out', icon: logoutIcon }}
            isActive={false}
            onClick={handleLogoutClick}
            colorConfig={{
              hover: '#ff00009a',
              clicked: '#69161672',
              selected: '#ff0000c2'
            }}
          />
        </nav>
      </aside>
    </>
  );
}

// Separate NavButton component for cleaner code
function NavButton({ item, isActive, onClick, colorConfig }) {
  const [isHovering, setIsHovering] = useState(false);

  const getStyle = () => {
    if (isActive) {
      return { backgroundColor: colorConfig.selected };
    }
    if (isHovering) {
      return { backgroundColor: colorConfig.hover };
    }
    return {};
  };

  const handleMouseDown = (e) => {
    e.currentTarget.style.backgroundColor = colorConfig.clicked;
  };

  const handleMouseUp = (e) => {
    e.currentTarget.style.backgroundColor = isActive ? colorConfig.selected : colorConfig.hover;
  };

  return (
    <button
      className={`statistics-nav-item ${isActive ? 'active' : ''}`}
      onClick={onClick}
      onMouseEnter={() => setIsHovering(true)}
      onMouseLeave={() => setIsHovering(false)}
      onMouseDown={handleMouseDown}
      onMouseUp={handleMouseUp}
      style={getStyle()}
    >
      <div className="statistics-nav-icon-box">
        <img src={item.icon} alt={item.label} className="statistics-nav-icon" />
      </div>
      <span className="statistics-nav-label">{item.label}</span>
    </button>
  );
}