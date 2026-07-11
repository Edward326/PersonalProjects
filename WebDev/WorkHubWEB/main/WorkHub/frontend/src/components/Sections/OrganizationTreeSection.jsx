import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import '../../styles/OrganizationPage.css';
import '../../styles/Notifications.css';

export default function OrganizationTreeSection({ organizationId, userId, isCEO, canViewStatistics, onError }) {
  const navigate = useNavigate();
  const [departments, setDepartments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedDepartment, setSelectedDepartment] = useState(null);
  const [departmentUsers, setDepartmentUsers] = useState([]);
  const [treeData, setTreeData] = useState(null);
  const [showUserModal, setShowUserModal] = useState(false);
  const [selectedUser, setSelectedUser] = useState(null);

  useEffect(() => {
    fetchDepartments();
  }, [organizationId]);

  const fetchDepartments = async () => {
    try {
      setLoading(true);
      const response = await fetch(`http://localhost:3000/api/departments/organization/${organizationId}`);
      const result = await response.json();
      
      if (result.success) {
        setDepartments(result.data);
      } else {
        onError('Failed to load departments', 'error');
      }
    } catch (error) {
      console.error('Error fetching departments:', error);
      onError('Failed to load departments', 'error');
    } finally {
      setLoading(false);
    }
  };

  const fetchDepartmentUsers = async (departmentId, managerId) => {
    try {
      // Fetch all users in the department
      const usersResponse = await fetch(`http://localhost:3000/api/users/department/${departmentId}`);
      const usersResult = await usersResponse.json();
      
      if (!usersResult.success) {
        throw new Error('Failed to fetch users');
      }

      const users = usersResult.data;

      // Fetch role for each user to get priority
      const usersWithRoles = await Promise.all(
        users.map(async (user) => {
          const roleResponse = await fetch(`http://localhost:3000/api/roles/${user.role_id}`);
          const roleResult = await roleResponse.json();
          
          return {
            ...user,
            role: roleResult.success ? roleResult.data : null,
            priority: roleResult.success ? roleResult.data.priority : 999
          };
        })
      );

      // Separate manager from other users
      let manager = null;
      let otherUsers = usersWithRoles;

      if (managerId) {
        manager = usersWithRoles.find(u => u.id === managerId);
        otherUsers = usersWithRoles.filter(u => u.id !== managerId);
      }

      // Sort other users by priority (lower priority number = higher importance)
      otherUsers.sort((a, b) => a.priority - b.priority);

      // Group users by priority level
      const priorityLevels = {};
      otherUsers.forEach(user => {
        if (!priorityLevels[user.priority]) {
          priorityLevels[user.priority] = [];
        }
        priorityLevels[user.priority].push(user);
      });

      setDepartmentUsers(usersWithRoles);
      setTreeData({ manager, priorityLevels });
    } catch (error) {
      console.error('Error fetching department users:', error);
      onError('Failed to load department users', 'error');
    }
  };

  const handleDepartmentClick = async (department) => {
    setSelectedDepartment(department);
    await fetchDepartmentUsers(department.id, department.manager_id);
  };

  const handleBackToDepartments = () => {
    setSelectedDepartment(null);
    setTreeData(null);
    setDepartmentUsers([]);
  };

  const handleUserClick = (user) => {
    setSelectedUser(user);
    setShowUserModal(true);
  };

  const handleViewStatistics = () => {
    navigate(`/statistics?userId=${selectedUser.id}`);
  };

  const formatDate = (date) => {
    if (!date) return 'N/A';
    return new Date(date).toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'long', 
      day: 'numeric' 
    });
  };

  if (loading) {
    return (
      <div className="section-loading">
        <div className="loading-spinner"></div>
        <p>Loading organization structure...</p>
      </div>
    );
  }

  // Department Cards View
  if (!selectedDepartment) {
    return (
      <div className="organization-tree-section">
        <div className="section-header">
          <h2>Organization Structure</h2>
          <p className="section-subtitle">View the complete structure of your organization by department</p>
        </div>

        <div className="departments-grid">
          {departments.length === 0 ? (
            <div className="no-departments">
              <p>No departments found in this organization.</p>
            </div>
          ) : (
            departments.map(dept => (
              <div key={dept.id} className="department-card" onClick={() => handleDepartmentClick(dept)}>
                <div className="department-card-header">
                  <h3>{dept.name}</h3>
                </div>
                <div className="department-card-body">
                  <div className="department-info">
                    <span className="info-label">Founded:</span>
                    <span className="info-value">{formatDate(dept.created_at)}</span>
                  </div>
                  {dept.manager_id && (
                    <div className="department-info">
                      <span className="info-label">Manager:</span>
                      <span className="info-value">Assigned</span>
                    </div>
                  )}
                </div>
                <button className="view-tree-button">View Department Tree</button>
              </div>
            ))
          )}
        </div>
      </div>
    );
  }

  // Tree View
  return (
    <div className="organization-tree-section">
      <div className="tree-header">
        <button className="back-to-departments-button" onClick={handleBackToDepartments}>
        Back to Departments
        </button>
        <h2>{selectedDepartment.name} - Organization Tree</h2>
      </div>

      <div className="tree-container">
        <div className="tree-wrapper">
          {/* Manager Node (Root) */}
          {treeData?.manager && (
            <div className="tree-level manager-level">
              <div className="tree-node manager-node" onClick={() => handleUserClick(treeData.manager)}>
                <div className="node-avatar">
                  {treeData.manager.first_name.charAt(0)}{treeData.manager.last_name.charAt(0)}
                </div>
                <div className="node-info">
                  <div className="node-name">{treeData.manager.first_name} {treeData.manager.last_name}</div>
                  <div className="node-role">{treeData.manager.role?.name || 'Manager'}</div>
                </div>
              </div>
              {Object.keys(treeData.priorityLevels).length > 0 && (
                <div className="tree-connector"></div>
              )}
            </div>
          )}

          {/* Priority Levels */}
          {treeData?.priorityLevels && Object.keys(treeData.priorityLevels).length > 0 ? (
            Object.keys(treeData.priorityLevels)
              .sort((a, b) => parseInt(a) - parseInt(b))
              .map((priority, levelIndex) => (
                <div key={priority} className="tree-level">
                  <div className="tree-nodes-row">
                    {treeData.priorityLevels[priority].map(user => (
                      <div key={user.id} className="tree-node" onClick={() => handleUserClick(user)}>
                        <div className="node-avatar">
                          {user.first_name.charAt(0)}{user.last_name.charAt(0)}
                        </div>
                        <div className="node-info">
                          <div className="node-name">{user.first_name} {user.last_name}</div>
                          <div className="node-role">{user.role?.name || 'Employee'}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                  {levelIndex < Object.keys(treeData.priorityLevels).length - 1 && (
                    <div className="tree-connector"></div>
                  )}
                </div>
              ))
          ) : (
            !treeData?.manager && (
              <div className="no-employees">
                <p>No employees found in this department.</p>
              </div>
            )
          )}
        </div>
      </div>

      {/* User Details Modal */}
      {showUserModal && selectedUser && (
        <div className="user-modal-overlay" onClick={() => setShowUserModal(false)}>
          <div className="user-modal" onClick={e => e.stopPropagation()}>
            <h2>Employee Details</h2>
            
            <div className="user-details">
              <div className="user-avatar-large">
                {selectedUser.first_name.charAt(0)}{selectedUser.last_name.charAt(0)}
              </div>
              
              <div className="user-info-grid">
                <div className="user-info-item">
                  <label>Full Name:</label>
                  <span>{selectedUser.first_name} {selectedUser.last_name}</span>
                </div>
                <div className="user-info-item">
                  <label>Email:</label>
                  <span>{selectedUser.email}</span>
                </div>
                <div className="user-info-item">
                  <label>Role:</label>
                  <span>{selectedUser.role?.name || 'N/A'}</span>
                </div>
                <div className="user-info-item">
                  <label>Hire Date:</label>
                  <span>{formatDate(selectedUser.hire_date)}</span>
                </div>
              </div>
            </div>

            <div className="user-modal-buttons">
              {canViewStatistics && (
                <button className="view-statistics-button" onClick={handleViewStatistics}>
                  View More About This Employee
                </button>
              )}
              <button className="close-user-modal-button" onClick={() => setShowUserModal(false)}>
                Back
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}