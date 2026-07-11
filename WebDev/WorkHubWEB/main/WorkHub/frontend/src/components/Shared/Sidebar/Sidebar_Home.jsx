import React, { useState } from 'react';
import newsIcon from '../../../assets/icons/news.png';
import eventsIcon from '../../../assets/Icons/event.png';
import requestsIcon from '../../../assets/Icons/request.png';
import assignmentsIcon from '../../../assets/Icons/clipboard.png';
import organizationIcon from '../../../assets/Icons/organization_choose.png';
import statisticsIcon from '../../../assets/Icons/chart.png';
import settingsIcon from '../../../assets/Icons/setting.png';
import logoutIcon from '../../../assets/Icons/logout.png';
import '../../../styles/HomePage.css';
import '../../../styles/Notifications.css';

export default function Sidebar_Home({ 
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
  const showRequests = isCEO || 
    (roleData && (roleData.can_hire || roleData.can_reassign_tasks));
  
  const showAssignments = !isCEO && 
    (roleData && (roleData.can_assign_tasks || roleData.can_receive_tasks));
  
  const showStatistics = isCEO || 
    (roleData && (roleData.can_view_statistics || roleData.can_assign_tasks));

  // Category 1: Main Navigation (Top)
  const topNavItems = [
    { id: 'news', label: 'News', icon: newsIcon },
    { id: 'events', label: 'Events', icon: eventsIcon },
  ];

  // Category 2: Role-Based (Middle)
  const middleNavItems = [];
  if (showRequests) {
    middleNavItems.push({ id: 'requests', label: 'Requests', icon: requestsIcon });
  }
  if (showAssignments) {
    middleNavItems.push({ id: 'assignments', label: 'Assignments', icon: assignmentsIcon });
  }

  // Category 3: Universal (Bottom)
  const bottomNavItems = [
    { id: 'organization', label: 'Organization', icon: organizationIcon },
  ];
  if (showStatistics) {
    bottomNavItems.push({ id: 'statistics', label: 'Statistics', icon: statisticsIcon });
  }
  bottomNavItems.push({ id: 'settings', label: 'Settings', icon: settingsIcon });

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

  // Dynamic styles based on colorConfig
  const getNavItemStyle = (isActive, isHovering) => {
    if (isActive) {
      return { backgroundColor: colorConfig.selected };
    }
    if (isHovering) {
      return { backgroundColor: colorConfig.hover };
    }
    return {};
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
        className={`home-sidebar ${isOpen ? 'expanded' : 'collapsed'} ${sidebarPosition === 'right' ? 'sidebar-right' : ''}`}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      >
        <div className="home-sidebar-toggle-handle" onClick={toggleSidebar}>
          <span className="home-toggle-arrow">
            {sidebarPosition === 'left' ? (isOpen ? '◀' : '▶') : (isOpen ? '▶' : '◀')}
          </span>
        </div>

        {/* Category 1: Top Navigation */}
        <nav className="home-sidebar-nav home-sidebar-top">
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

        {/* Divider */}
        {middleNavItems.length > 0 && <div className="home-sidebar-divider"></div>}

        {/* Category 2: Middle Navigation */}
        {middleNavItems.length > 0 && (
          <nav className="home-sidebar-nav home-sidebar-middle">
            {middleNavItems.map((item) => (
              <NavButton
                key={item.id}
                item={item}
                isActive={activeSection === item.id}
                onClick={() => setActiveSection(item.id)}
                colorConfig={colorConfig}
              />
            ))}
          </nav>
        )}

        {/* Category 3: Bottom Navigation */}
        <nav className="home-sidebar-nav home-sidebar-bottom">
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
            colorConfig={ {
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
      className={`home-nav-item ${isActive ? 'active' : ''}`}
      onClick={onClick}
      onMouseEnter={() => setIsHovering(true)}
      onMouseLeave={() => setIsHovering(false)}
      onMouseDown={handleMouseDown}
      onMouseUp={handleMouseUp}
      style={getStyle()}
    >
      <div className="home-nav-icon-box">
        <img src={item.icon} alt={item.label} className="home-nav-icon" />
      </div>
      <span className="home-nav-label">{item.label}</span>
    </button>
  );
}