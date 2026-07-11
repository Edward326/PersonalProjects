import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

export default function CompanyCard({ company, onBack }) {
  const navigate = useNavigate();

  useEffect(() => {
    const handleEsc = (e) => {
      if (e.key === 'Escape') {
        onBack();
      }
    };
    window.addEventListener('keydown', handleEsc);
    return () => window.removeEventListener('keydown', handleEsc);
  }, [onBack]);

  const handleEmailClick = () => {
    window.location.href = `mailto:${company.contact_email}?subject=Business Inquiry`;
  };

  const handleJoin = () => {
    // Navigate to EmployeeRegister with company ID
    navigate(`/employee-register?orgId=${company.id}&orgName=${encodeURIComponent(company.name)}`);
  };

  const hasLinkedIn = company.linkedin_url && company.linkedin_url.trim() !== '';
  const hasTwitter = company.twitter_url && company.twitter_url.trim() !== '';
  const hasWebsite = company.website && company.website.trim() !== '';

  return (
    <div className="company-card-expanded">
      <button className="back-button" onClick={onBack}>← Back</button>
      
      <div className="company-header">
        <div className="company-logo-large">
          {company.logo_url ? (
            <img 
              src={company.logo_url} 
              alt={company.name}
              onError={(e) => {
                e.target.style.display = 'none';
              }}
            />
          ) : (
            <div style={{ fontSize: '48px' }}>🏢</div>
          )}
        </div>
        <h1>{company.name}</h1>
        <p className="company-industry">{company.industry}</p>
      </div>

      <div className="company-details">
        <div className="detail-section">
          <h3 className="detail-title">Description</h3>
          <p className="detail-value">{company.description}</p>
        </div>
        
        <div className="detail-section">
          <h3 className="detail-title">Founding Date</h3>
          <p className="detail-value">{new Date(company.founding_date).toLocaleDateString('en-US', { 
            year: 'numeric', 
            month: 'long', 
            day: 'numeric' 
          })}</p>
        </div>

        <div className="detail-section">
          <h3 className="detail-title">Employees</h3>
          <p className="detail-value">{company.employees_count || 0}</p>
        </div>

        <div className="detail-section">
          <h3 className="detail-title">Departments</h3>
          <p className="detail-value">{company.departments_count || 0}</p>
        </div>
      </div>

      <div className="company-links">
        {hasWebsite && (
          <a 
            href={company.website} 
            target="_blank" 
            rel="noopener noreferrer" 
            className="link-btn"
          >
            <img 
              src="https://cdn-icons-png.flaticon.com/512/1006/1006771.png" 
              alt="Website" 
              className="link-icon"
            />
            Website
          </a>
        )}
        {hasLinkedIn && (
          <a 
            href={company.linkedin_url} 
            target="_blank" 
            rel="noopener noreferrer" 
            className="link-btn"
          >
            <img 
              src="https://cdn-icons-png.flaticon.com/512/174/174857.png" 
              alt="LinkedIn" 
              className="link-icon"
            />
            LinkedIn
          </a>
        )}
        {hasTwitter && (
          <a 
            href={company.twitter_url} 
            target="_blank" 
            rel="noopener noreferrer" 
            className="link-btn"
          >
            <img 
              src="https://cdn-icons-png.flaticon.com/512/733/733579.png" 
              alt="Twitter" 
              className="link-icon"
            />
            Twitter
          </a>
        )}
      </div>

      <div className="company-actions">
        <button className="contact-btn" onClick={handleEmailClick}>
          📧 Contact Company
        </button>
        <button className="join-btn" onClick={handleJoin}>
          ⭐ Join Company
        </button>
      </div>
    </div>
  );
}