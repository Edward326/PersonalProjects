import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import homeIcon from '../../../assets/Icons/home.png';
import searchIcon from '../../../assets/Icons/search.png';
import aboutIcon from '../../../assets/Icons/info.png';
import statsIcon from '../../../assets/Icons/badge.png';
import loginIcon from '../../../assets/Icons/login.png';
import joinIcon from '../../../assets/Icons/join.png';
import '../../../styles/WelcomePage.css';

export default function Sidebar({ activeSection, setActiveSection, sidebarExpanded, setSidebarExpanded }) {
  const [isHovered, setIsHovered] = useState(false);
  const navigate = useNavigate();

  const isOpen = sidebarExpanded || isHovered;

  const toggleSidebar = () => {
    setSidebarExpanded(!sidebarExpanded);
  };

  // Order: Home, Statistics, Search, About
  const navItems = [
    { id: 'home', label: 'Home', icon: homeIcon },
    { id: 'classement', label: 'Statistics', icon: statsIcon },
    { id: 'search', label: 'Search', icon: searchIcon },
    { id: 'about', label: 'About', icon: aboutIcon },
  ];

  const handleLogin = () => {
    navigate('/login');
  };

  const handleJoin = () => {
    navigate('/choose-register');
  };

  return (
    <aside 
      className={`sidebar ${isOpen ? 'expanded' : 'collapsed'}`}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div className="sidebar-toggle-handle" onClick={toggleSidebar}>
        <span className="toggle-arrow">◀</span>
        <span className="toggle-arrow">▶</span>
      </div>

      <nav className="sidebar-nav">
        {navItems.map((item) => (
          <button
            key={item.id}
            className={`nav-item ${activeSection === item.id ? 'active' : ''}`}
            onClick={() => setActiveSection(item.id)}
          >
            <div className="nav-icon-box">
              <img src={item.icon} alt={item.label} className="nav-icon" />
            </div>
            <span className="nav-label">{item.label}</span>
          </button>
        ))}
      </nav>

      {/* Footer with Login and Join - using nav-item class */}
      <div className="sidebar-footer">
        <button
          className="nav-item"
          onClick={handleLogin}
        >
          <div className="nav-icon-box">
            <img src={loginIcon} alt="Login" className="nav-icon" />
          </div>
          <span className="nav-label">Login</span>
        </button>

        <button
          className="nav-item"
          onClick={handleJoin}
        >
          <div className="nav-icon-box">
            <img src={joinIcon} alt="Join Now" className="nav-icon" />
          </div>
          <span className="nav-label">Join Now</span>
        </button>
      </div>
    </aside>
  );
}