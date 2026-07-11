import React, { useState } from 'react';
import assignmentsIcon from '../../../assets/Icons/clipboard.png';
import adminIcon from '../../../assets/Icons/setting.png';
import homeIcon from '../../../assets/Icons/home.png';
import settingsIcon from '../../../assets/Icons/setting.png';
import logoutIcon from '../../../assets/Icons/logout.png';
import '../../../styles/AssignmentPage.css';

export default function Sidebar_Assignments({ 
  activeSection, 
  setActiveSection, 
  sidebarExpanded, 
  setSidebarExpanded,
  sidebarPosition = 'left',
  roleData,
  colorConfig,
  onLogout
}) {
  const [isHovered, setIsHovered] = useState(false);
  const [showLogoutConfirm, setShowLogoutConfirm] = useState(false);

  const isOpen = sidebarExpanded || isHovered;

  const toggleSidebar = () => {
    setSidebarExpanded(!sidebarExpanded);
  };

  const showAssignments = roleData && roleData.can_receive_tasks;
  const showAdmin = roleData && roleData.can_assign_tasks;

  // Top Navigation
  const topNavItems = [];
  if (showAssignments) {
    topNavItems.push({ id: 'assignments', label: 'My assignments', icon: assignmentsIcon });
  }
  if (showAdmin) {
    topNavItems.push({ id: 'admin', label: 'Manage assignments', icon: adminIcon });
  }

  // Bottom Navigation
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

  return (
    <>
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
        className={`assignment-sidebar ${isOpen ? 'expanded' : 'collapsed'} ${sidebarPosition === 'right' ? 'sidebar-right' : ''}`}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      >
        <div className="assignment-sidebar-toggle-handle" onClick={toggleSidebar}>
          <span className="assignment-toggle-arrow">
            {sidebarPosition === 'left' ? (isOpen ? '◀' : '▶') : (isOpen ? '▶' : '◀')}
          </span>
        </div>

        <nav className="assignment-sidebar-nav assignment-sidebar-top">
          {topNavItems.map((item) => (
            <NavButton
              key={item.id}
              item={item}
              isActive={activeSection === item.id}
              onClick={() => setActiveSection(item.id)}
              colorConfig={colorConfig}
            />
          ))}
        </nav>

        <nav className="assignment-sidebar-nav assignment-sidebar-bottom">
          {bottomNavItems.map((item) => (
            <NavButton
              key={item.id}
              item={item}
              isActive={activeSection === item.id}
              onClick={() => setActiveSection(item.id)}
              colorConfig={colorConfig}
            />
          ))}
          
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
      className={`assignment-nav-item ${isActive ? 'active' : ''}`}
      onClick={onClick}
      onMouseEnter={() => setIsHovering(true)}
      onMouseLeave={() => setIsHovering(false)}
      onMouseDown={handleMouseDown}
      onMouseUp={handleMouseUp}
      style={getStyle()}
    >
      <div className="assignment-nav-icon-box">
        <img src={item.icon} alt={item.label} className="assignment-nav-icon" />
      </div>
      <span className="assignment-nav-label">{item.label}</span>
    </button>
  );
}