import React, { useState } from 'react';
import accountIcon from '../../../assets/Icons/employee_choose.png';
import administrationIcon from '../../../assets/Icons/setting.png';
import homeIcon from '../../../assets/Icons/home.png';
import logoutIcon from '../../../assets/Icons/logout.png';
import '../../../styles/SettingsPage.css';
import '../../../styles/Notifications.css';

export default function Sidebar_Settings({ 
  activeSection, 
  setActiveSection, 
  sidebarExpanded, 
  setSidebarExpanded,
  sidebarPosition = 'left',
  isCEO,
  isManager,
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

  // Top Navigation - Account is visible for all
  const topNavItems = [
    { id: 'account', label: 'Account', icon: accountIcon }
  ];

  // Add Administration only for CEO or Manager
  if (isCEO || isManager) {
    topNavItems.push({ id: 'administration', label: 'Administration', icon: administrationIcon });
  }

  // Bottom Navigation
  const bottomNavItems = [
    { id: 'home', label: 'Home', icon: homeIcon }
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
        className={`settings-sidebar ${isOpen ? 'expanded' : 'collapsed'} ${sidebarPosition === 'right' ? 'sidebar-right' : ''}`}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      >
        <div className="settings-sidebar-toggle-handle" onClick={toggleSidebar}>
          <span className="settings-toggle-arrow">
            {sidebarPosition === 'left' ? (isOpen ? '◀' : '▶') : (isOpen ? '▶' : '◀')}
          </span>
        </div>

        {/* Top Navigation */}
        <nav className="settings-sidebar-nav settings-sidebar-top">
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
        <div className="settings-sidebar-divider"></div>

        {/* Bottom Navigation */}
        <nav className="settings-sidebar-nav settings-sidebar-bottom">
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

// Separate NavButton component
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
      className={`settings-nav-item ${isActive ? 'active' : ''}`}
      onClick={onClick}
      onMouseEnter={() => setIsHovering(true)}
      onMouseLeave={() => setIsHovering(false)}
      onMouseDown={handleMouseDown}
      onMouseUp={handleMouseUp}
      style={getStyle()}
    >
      <div className="settings-nav-icon-box">
        <img src={item.icon} alt={item.label} className="settings-nav-icon" />
      </div>
      <span className="settings-nav-label">{item.label}</span>
    </button>
  );
}