import React, { useState } from 'react';
import joinRequestIcon from '../../../assets/Icons/request.png';
import reassignIcon from '../../../assets/Icons/clipboard.png';
import homeIcon from '../../../assets/Icons/home.png';
import settingsIcon from '../../../assets/Icons/setting.png';
import logoutIcon from '../../../assets/Icons/logout.png';
import '../../../styles/RequestsPage.css';
import '../../../styles/Notifications.css';

export default function Sidebar_Requests({ 
  activeSection, 
  setActiveSection, 
  sidebarExpanded, 
  setSidebarExpanded,
  sidebarPosition = 'left',
  roleData,
  isCEO,
  colorConfig,
  onLogout
}) {
  const [isHovered, setIsHovered] = useState(false);
  const [showLogoutConfirm, setShowLogoutConfirm] = useState(false);

  const isOpen = sidebarExpanded || isHovered;

  const toggleSidebar = () => {
    setSidebarExpanded(!sidebarExpanded);
  };

  // Determine which buttons to show based on role
  const showJoinRequests = isCEO || (roleData && roleData.can_hire);
  const showReassignment = !isCEO && (roleData && roleData.can_reassign_tasks);

  // Category 1: Request Management (Top)
  const topNavItems = [];
  if (showJoinRequests) {
    topNavItems.push({ id: 'join-requests', label: 'Join Requests', icon: joinRequestIcon });
  }
  if (showReassignment) {
    topNavItems.push({ id: 'reassignment', label: 'Reassignment', icon: reassignIcon });
  }

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
        className={`requests-sidebar ${isOpen ? 'expanded' : 'collapsed'} ${sidebarPosition === 'right' ? 'sidebar-right' : ''}`}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      >
        <div className="requests-sidebar-toggle-handle" onClick={toggleSidebar}>
          <span className="requests-toggle-arrow">
            {sidebarPosition === 'left' ? (isOpen ? '◀' : '▶') : (isOpen ? '▶' : '◀')}
          </span>
        </div>

        {/* Category 1: Top Navigation */}
        <nav className="requests-sidebar-nav requests-sidebar-top">
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

        {/* Category 2: Bottom Navigation */}
        <nav className="requests-sidebar-nav requests-sidebar-bottom">
          {bottomNavItems.map((item) => (
            <NavButton
              key={item.id}
              item={item}
              isActive={activeSection === item.id}
              onClick={() => setActiveSection(item.id)}
              colorConfig={colorConfig}
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
      className={`requests-nav-item ${isActive ? 'active' : ''}`}
      onClick={onClick}
      onMouseEnter={() => setIsHovering(true)}
      onMouseLeave={() => setIsHovering(false)}
      onMouseDown={handleMouseDown}
      onMouseUp={handleMouseUp}
      style={getStyle()}
    >
      <div className="requests-nav-icon-box">
        <img src={item.icon} alt={item.label} className="requests-nav-icon" />
      </div>
      <span className="requests-nav-label">{item.label}</span>
    </button>
  );
}