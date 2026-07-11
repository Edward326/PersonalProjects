import React, { useState } from 'react';
import homeIcon from '../../../assets/Icons/home.png';
import settingsIcon from '../../../assets/Icons/setting.png';
import logoutIcon from '../../../assets/Icons/logout.png';
import '../../../styles/OrganizationPage.css';
import '../../../styles/Notifications.css';

export default function Sidebar_Organization({ 
  sidebarExpanded, 
  setSidebarExpanded,
  sidebarPosition = 'left',
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

  // Bottom Navigation buttons
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
        className={`organization-sidebar ${isOpen ? 'expanded' : 'collapsed'} ${sidebarPosition === 'right' ? 'sidebar-right' : ''}`}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      >
        <div className="organization-sidebar-toggle-handle" onClick={toggleSidebar}>
          <span className="organization-toggle-arrow">
            {sidebarPosition === 'left' ? (isOpen ? '◀' : '▶') : (isOpen ? '▶' : '◀')}
          </span>
        </div>

        {/* Bottom Navigation */}
        <nav className="organization-sidebar-nav organization-sidebar-bottom">
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
      className={`organization-nav-item ${isActive ? 'active' : ''}`}
      onClick={onClick}
      onMouseEnter={() => setIsHovering(true)}
      onMouseLeave={() => setIsHovering(false)}
      onMouseDown={handleMouseDown}
      onMouseUp={handleMouseUp}
      style={getStyle()}
    >
      <div className="organization-nav-icon-box">
        <img src={item.icon} alt={item.label} className="organization-nav-icon" />
      </div>
      <span className="organization-nav-label">{item.label}</span>
    </button>
  );
}