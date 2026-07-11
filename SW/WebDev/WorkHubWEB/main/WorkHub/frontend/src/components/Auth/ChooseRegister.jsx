import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import '../../styles/ChooseRegister.css';
import employeeIcon from '../../assets/icons/employee_choose.png'
import organizationIcon from '../../assets/Icons/organization_choose.png';

export default function ChooseRegister() {
  const navigate = useNavigate();
  const [hoveredCard, setHoveredCard] = useState(null);

  const handleEmployeeClick = () => {
    navigate('/employee-register');
  };

  const handleOrganizationClick = () => {
    navigate('/organization-register');
  };

  return (
    <div className="choose-register-container">
      <div className="choose-register-content">
        <h1 className="choose-register-title">Who Are You?</h1>
        <p className="choose-register-subtitle">Select your path to join WorkHub</p>
        
        <div className="cards-container">
          {/* Employee Card */}
          <div 
            className={`register-card ${hoveredCard === 'employee' ? 'hovered' : ''}`}
            onMouseEnter={() => setHoveredCard('employee')}
            onMouseLeave={() => setHoveredCard(null)}
            onClick={handleEmployeeClick}
          >
            <div className="card-header">
              <div className="card-icon">
                <img src={employeeIcon} alt="Employee" />
              </div>
              <h2 className="card-title">Employee</h2>
            </div>
            
            <div className="card-description">
              <p>
                Join an existing organization and become part of a team. 
                Access assignments, collaborate with colleagues, track your progress, 
                and grow your career within an established company.
              </p>
            </div>
          </div>

          {/* Organization Card */}
          <div 
            className={`register-card ${hoveredCard === 'organization' ? 'hovered' : ''}`}
            onMouseEnter={() => setHoveredCard('organization')}
            onMouseLeave={() => setHoveredCard(null)}
            onClick={handleOrganizationClick}
          >
            <div className="card-header">
              <div className="card-icon">
                <img src={organizationIcon} alt="Organization" />
              </div>
              <h2 className="card-title">Organization</h2>
            </div>
            
            <div className="card-description">
              <p>
                Create and manage your own organization. Build your team, 
                assign tasks, track productivity, manage departments, 
                and scale your business efficiently.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}