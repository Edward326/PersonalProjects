import React, { useState, useEffect } from 'react';
import '../../styles/StatisticsPage.css';

export default function OrganizationStatisticsSection({ organizationId, onError }) {
  const [loading, setLoading] = useState(true);
  const [organizationData, setOrganizationData] = useState(null);
  const [departments, setDepartments] = useState([]);
  const [totalRoles, setTotalRoles] = useState(0);

  useEffect(() => {
    fetchOrganizationStats();
  }, [organizationId]);

  const fetchOrganizationStats = async () => {
    try {
      setLoading(true);

      // Fetch organization data
      const orgResponse = await fetch(`http://localhost:3000/api/organizations/${organizationId}`);
      const orgResult = await orgResponse.json();

      if (!orgResult.success) {
        throw new Error('Failed to fetch organization data');
      }

      setOrganizationData(orgResult.data);

      // Fetch all departments
      const deptResponse = await fetch(`http://localhost:3000/api/departments/organization/${organizationId}`);
      const deptResult = await deptResponse.json();

      if (!deptResult.success) {
        throw new Error('Failed to fetch departments');
      }

      // For each department, fetch additional stats
      const departmentsWithStats = await Promise.all(
        deptResult.data.map(async (dept) => {
          // Fetch manager data if exists
          let manager = null;
          if (dept.manager_id) {
            const managerResponse = await fetch(`http://localhost:3000/api/users/${dept.manager_id}`);
            const managerResult = await managerResponse.json();
            if (managerResult.success) {
              manager = managerResult.data;
            }
          }

          // Fetch employee count
          const employeesResponse = await fetch(`http://localhost:3000/api/users/department/${dept.id}`);
          const employeesResult = await employeesResponse.json();
          const employeeCount = employeesResult.success ? employeesResult.data.length : 0;

          // Fetch roles count
          const rolesResponse = await fetch(`http://localhost:3000/api/roles/department/${dept.id}`);
          const rolesResult = await rolesResponse.json();
          const rolesCount = rolesResult.success ? rolesResult.data.length : 0;
          
          // Find highest priority role (lowest priority number)
          let highestPriorityRole = null;
          if (rolesResult.success && rolesResult.data.length > 0) {
            highestPriorityRole = rolesResult.data.reduce((prev, current) => 
              (prev.priority < current.priority) ? prev : current
            );
          }

          // Fetch attendance stats
          const attendanceResponse = await fetch(`http://localhost:3000/api/department-attendance-stats/department/${dept.id}`);
          const attendanceResult = await attendanceResponse.json();
          const attendanceStats = attendanceResult.success ? attendanceResult.data : [];

          return {
            ...dept,
            manager,
            employeeCount,
            rolesCount,
            highestPriorityRole,
            attendanceStats
          };
        })
      );

      setDepartments(departmentsWithStats);

      // Calculate total roles across all departments
      const totalRolesCount = departmentsWithStats.reduce((sum, dept) => sum + dept.rolesCount, 0);
      setTotalRoles(totalRolesCount);

    } catch (error) {
      console.error('Error fetching organization stats:', error);
      onError('Failed to load organization statistics', 'error');
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (date) => {
    if (!date) return 'N/A';
    return new Date(date).toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'long', 
      day: 'numeric' 
    });
  };

  const circularLogoStyle = {
    width: '40px',
    height: '40px',
    borderRadius: '50%',
    objectFit: 'cover',
    marginLeft: '12px'
  };

  if (loading) {
    return (
      <div className="section-loading">
        <div className="loading-spinner"></div>
        <p>Loading organization statistics...</p>
      </div>
    );
  }

  if (!organizationData) {
    return (
      <div className="no-data">
        <p>No organization data available.</p>
      </div>
    );
  }

  return (
    <div className="organization-statistics-section">
      <div className="section-header">
        <h2>Organization Statistics</h2>
        <p className="section-subtitle">Complete overview of your organization's performance and structure</p>
      </div>

      {/* Company Overview */}
      <div className="stats-card company-overview">
        <div className="company-header-stats">
          <div className="company-name-logo">
            <h3>{organizationData.name}</h3>
            {organizationData.logo_url && (
              <img 
                src={organizationData.logo_url} 
                alt={organizationData.name}
                style={circularLogoStyle}
                onError={(e) => e.target.style.display = 'none'}
              />
            )}
          </div>
          <div className="unique-code">Unique ID: {organizationData.unique_code}</div>
        </div>

        <div className="stats-grid">
          <div className="stat-item">
            <label>Industry:</label>
            <span>{organizationData.industry}</span>
          </div>
          <div className="stat-item">
            <label>Founding Date:</label>
            <span>{formatDate(organizationData.founding_date)}</span>
          </div>
          <div className="stat-item">
            <label>Contact Email:</label>
            <span>{organizationData.contact_email}</span>
          </div>
          <div className="stat-item">
            <label>CEO Email:</label>
            <span>{organizationData.ceo_email}</span>
          </div>
        </div>

        {/* Statistics Numbers Row */}
        <div className="stats-numbers-row">
          <div className="stat-number-box">
            <div className="stat-number-title">Employees</div>
            <div className="stat-number-value">{organizationData.employees_count || 0}</div>
          </div>
          <div className="stat-number-box">
            <div className="stat-number-title">Departments</div>
            <div className="stat-number-value">{organizationData.departments_count || 0}</div>
          </div>
          <div className="stat-number-box">
            <div className="stat-number-title">Roles</div>
            <div className="stat-number-value">{totalRoles}</div>
          </div>
        </div>
      </div>

      {/* Departments Statistics */}
      <div className="departments-statistics">
        <h3 className="subsection-title">Departments Breakdown</h3>
        
        {departments.length === 0 ? (
          <div className="no-departments">
            <p>No departments found.</p>
          </div>
        ) : (
          departments.map((dept, index) => (
            <div key={dept.id} className="stats-card department-card">
              <h4 className="department-name">{dept.name}</h4>
              
              <div className="stats-grid">
                {dept.manager ? (
                  <>
                    <div className="stat-item full-width">
                      <label>Manager:</label>
                      <span>{dept.manager.first_name} {dept.manager.last_name}</span>
                    </div>
                    <div className="stat-item full-width">
                      <label>Manager Email:</label>
                      <span>{dept.manager.email}</span>
                    </div>
                  </>
                ) : (
                  <div className="stat-item full-width">
                    <label>Manager:</label>
                    <span className="no-manager">No manager assigned</span>
                  </div>
                )}
                
                <div className="stat-item">
                  <label>Employees:</label>
                  <span className="stat-number">{dept.employeeCount}</span>
                </div>
                
                <div className="stat-item">
                  <label>Roles:</label>
                  <span className="stat-number">{dept.rolesCount}</span>
                </div>
                
                <div className="stat-item full-width">
                  <label>Highest Priority Role:</label>
                  <span>{dept.highestPriorityRole ? dept.highestPriorityRole.name : 'N/A'}</span>
                </div>
              </div>

              {/* Attendance Statistics */}
              {dept.attendanceStats && dept.attendanceStats.length > 0 && (
                <div className="attendance-stats">
                  <h5>Attendance Statistics</h5>
                  <div className="attendance-table">
                    <table>
                      <thead>
                        <tr>
                          <th>Year-Month-Day</th>
                          <th>Total Attendances</th>
                          <th>Attendance %</th>
                        </tr>
                      </thead>
                      <tbody>
                        {dept.attendanceStats.map((stat, idx) => (
                          <tr key={idx}>
                            <td>{stat.month}</td>
                            <td>{stat.total_attendances}</td>
                            <td>
                              <span className={`percentage-badge ${getPercentageClass(stat.attendance_percentage)}`}>
                                {stat.attendance_percentage ? `${parseFloat(stat.attendance_percentage).toFixed(2)}%` : 'N/A'}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

// Helper function to get percentage badge class
function getPercentageClass(percentage) {
  if (!percentage) return '';
  const value = parseFloat(percentage);
  if (value >= 90) return 'excellent';
  if (value >= 75) return 'good';
  if (value >= 60) return 'average';
  return 'low';
}