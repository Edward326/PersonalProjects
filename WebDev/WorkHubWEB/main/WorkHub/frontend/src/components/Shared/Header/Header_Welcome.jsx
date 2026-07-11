import React from 'react'
import '../../../styles/WelcomePage.css';

export default function Header() {
  return (
    <header className="header">
      <div className="header-content">
        <div className="logo-section">
          <div className="logo">WorkHub</div>
          <div className="motivation-text">
            Centralize. Organize. Succeed.
          </div>
        </div>
      </div>
    </header>
  );
}