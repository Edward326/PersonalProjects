import React from 'react';
import '../../../styles/HomePage.css';
import '../../../styles/Notifications.css';

export default function Header_Custom({ config, organizationData, isCEO, userData }) {
  const circularLogoStyle = (size) => ({
    height: size,
    width: size,
    borderRadius: '50%',
    objectFit: 'cover',
    flexShrink: 0,
  });
  
  // For CEO, use default styling
  if (isCEO) {
    return (
      <header className="home-header" style={{ backgroundColor: '#667eea' }}>
        <div className="home-header-content">
          <div className="home-logo-section">
            {organizationData?.logo_url && (
              <img 
                src={organizationData.logo_url} 
                alt={organizationData.name} 
                className="home-org-logo"
                style={circularLogoStyle('50px')}
              />
            )}
            <div className="home-logo">{organizationData?.name || 'WorkHub'}</div>
            <div className="home-motivation-text">
              {`Hello, CEO ${userData?.first_name || ''}!`}
            </div>
          </div>
        </div>
      </header>
    );
  }

  // For employees, use department config
  const {
    header_background_color = '#667eea',
    header_display_type = 'both',
    header_position = 'center',
    logo_size = '50px',
    name: departmentName = 'Department'
  } = config;

  const renderHeaderContent = () => {
    switch (header_display_type) {
      case 'both':
        // Show logo + department name
        const hasValidLogo = organizationData?.logo_url && organizationData.logo_url.trim() !== '';
        
        if (!hasValidLogo) {
          // No valid logo, behave like 'dep_name'
          return (
            <div className="home-logo-section">
              <div className="home-logo">{departmentName}</div>
              <div className="home-motivation-text">
                {`Welcome back, ${userData?.first_name || 'User'}!`}
              </div>
            </div>
          );
        }

        return (
          <div className="home-logo-section">
            <img 
              src={organizationData.logo_url} 
              alt={organizationData.name} 
              className="home-org-logo"
              style={circularLogoStyle(logo_size)}
            />
            <div className="home-logo">{departmentName}</div>
            <div className="home-motivation-text">
              {`Welcome back, ${userData?.first_name || 'User'}!`}
            </div>
          </div>
        );

      case 'dep_name':
        // Show only department name
        return (
          <div className="home-logo-section">
            <div className="home-logo">{departmentName}</div>
            <div className="home-motivation-text">
              {`Welcome back, ${userData?.first_name || 'User'}!`}
            </div>
          </div>
        );

      case 'none':
        // Empty header with background color only
        return (
          <div className="home-logo-section">
            <div className="home-motivation-text">
              {`Welcome back, ${userData?.first_name || 'User'}!`}
            </div>
          </div>
        );

      default:
        return (
          <div className="home-logo-section">
            <div className="home-logo">{departmentName}</div>
            <div className="home-motivation-text">
              {`Welcome back, ${userData?.first_name || 'User'}!`}
            </div>
          </div>
        );
    }
  };

  // Determine alignment class
  const alignmentClass = `align-${header_position}`;

  return (
    <header 
      className="home-header" 
      style={{ backgroundColor: header_background_color }}
    >
      <div className={`home-header-content ${alignmentClass}`}>
        {renderHeaderContent()}
      </div>
    </header>
  );
}