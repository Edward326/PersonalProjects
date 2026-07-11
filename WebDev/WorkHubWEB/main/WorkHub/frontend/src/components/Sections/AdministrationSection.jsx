import React, { useState, useEffect } from 'react';
import ManageEmployeesView from './ManageEmployeeViews.jsx'
import '../../styles/SettingsPage.css';

export default function AdministrationSection({ 
  userData, 
  organizationData, 
  departmentData, 
  isCEO, 
  isManager, 
  onError, 
  onUpdate 
}) {
  const [activeView, setActiveView] = useState('main');
  const [employees, setEmployees] = useState([]);
  const [departments, setDepartments] = useState([]);
  const [roles, setRoles] = useState([]);
  const [loading, setLoading] = useState(false);

  // Modal states
  const [showEmployeeModal, setShowEmployeeModal] = useState(false);
  const [selectedEmployee, setSelectedEmployee] = useState(null);
  const [showRoleChangeModal, setShowRoleChangeModal] = useState(false);
  const [showDismissModal, setShowDismissModal] = useState(false);
  const [dismissReason, setDismissReason] = useState('');
  
  // Department appearance
  const [showAppearanceModal, setShowAppearanceModal] = useState(false);
  const [appearanceData, setAppearanceData] = useState({});
  const [appearancePassword, setAppearancePassword] = useState('');

  // Organization management
  const [showOrgModal, setShowOrgModal] = useState(false);
  const [orgEditData, setOrgEditData] = useState({});
  const [orgPassword, setOrgPassword] = useState('');
  const [orgUniqueCode, setOrgUniqueCode] = useState('');

  // Department management
  const [showDepartmentModal, setShowDepartmentModal] = useState(false);
  const [selectedDepartment, setSelectedDepartment] = useState(null);
  const [showDepartmentWizard, setShowDepartmentWizard] = useState(false);

  // Role management
  const [showRoleModal, setShowRoleModal] = useState(false);
  const [selectedRole, setSelectedRole] = useState(null);
  const [showRoleWizard, setShowRoleWizard] = useState(false);

  useEffect(() => {
    if (activeView === 'manage-employees') {
      fetchEmployees();
    } else if (activeView === 'manage-departments' && isCEO) {
      fetchDepartments();
    } else if (activeView === 'manage-roles' && isCEO) {
      fetchAllRoles();
    }
  }, [activeView]);

  const fetchEmployees = async () => {
    try {
      setLoading(true);
      let url;
      
      if (isCEO) {
        // CEO sees all organization employees
        url = `http://localhost:3000/api/users/organization/${organizationData.id}`;
      } else if (isManager) {
        // Manager sees only department employees
        url = `http://localhost:3000/api/users/department/${departmentData.id}/exclude/${userData.id}`;
      }

      const response = await fetch(url);
      const result = await response.json();
      
      if (result.success) {
        setEmployees(result.data);
      } else {
        onError('Failed to load employees', 'error');
      }
    } catch (error) {
      console.error('Error fetching employees:', error);
      onError('Failed to load employees', 'error');
    } finally {
      setLoading(false);
    }
  };

  const fetchDepartments = async () => {
    try {
      setLoading(true);
      const response = await fetch(`http://localhost:3000/api/departments/organization/${organizationData.id}`);
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

  const fetchAllRoles = async () => {
    try {
      setLoading(true);
      const response = await fetch(`http://localhost:3000/api/roles/organization/${organizationData.id}`);
      const result = await response.json();
      
      if (result.success) {
        setRoles(result.data);
      } else {
        onError('Failed to load roles', 'error');
      }
    } catch (error) {
      console.error('Error fetching roles:', error);
      onError('Failed to load roles', 'error');
    } finally {
      setLoading(false);
    }
  };

  // Handle employee click
  const handleEmployeeClick = async (employee) => {
    setSelectedEmployee(employee);
    setShowEmployeeModal(true);
  };

  // Handle role change
  const handleRoleChangeClick = async () => {
    try {
      // Fetch available roles for employee's department
      const response = await fetch(`http://localhost:3000/api/roles/department/${selectedEmployee.department_id}`);
      const result = await response.json();
      
      if (result.success) {
        setRoles(result.data);
        setShowRoleChangeModal(true);
      }
    } catch (error) {
      console.error('Error fetching roles:', error);
      onError('Failed to load roles', 'error');
    }
  };

  // Confirm role change
  const handleConfirmRoleChange = async (newRoleId) => {
    try {
      const response = await fetch(`http://localhost:3000/api/users/${selectedEmployee.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          role_id: parseInt(newRoleId),
          updated_at: new Date().toISOString()
        })
      });

      const result = await response.json();

      if (result.success) {
        onError('Role changed successfully!', 'success');
        setShowRoleChangeModal(false);
        setShowEmployeeModal(false);
        fetchEmployees();
      } else {
        onError('Failed to change role', 'error');
      }
    } catch (error) {
      console.error('Error changing role:', error);
      onError('Failed to change role', 'error');
    }
  };

  // Handle dismiss employee
  const handleDismissClick = () => {
    setDismissReason('');
    setShowDismissModal(true);
  };

  // Confirm dismiss - CASCADE DELETE
  const handleConfirmDismiss = async () => {
    if (!dismissReason.trim()) {
      onError('Please provide a reason for dismissal', 'error');
      return;
    }

    try {
      const userId = selectedEmployee.id;

      // 1. Delete assignment_users
      await fetch(`http://localhost:3000/api/assignment-users/user/${userId}/all`, {
        method: 'DELETE'
      });

      // 2. Delete assignment_relocations (from_user_id, to_user_id, reviewed_by)
      const relocationsResponse = await fetch(`http://localhost:3000/api/assignment-relocations/user/${userId}`);
      const relocationsResult = await relocationsResponse.json();
      if (relocationsResult.success && relocationsResult.data.length > 0) {
        for (const relocation of relocationsResult.data) {
          await fetch(`http://localhost:3000/api/assignment-relocations/${relocation.id}`, {
            method: 'DELETE'
          });
        }
      }

      // 3. Delete assignments created by user
      const assignmentsResponse = await fetch(`http://localhost:3000/api/assignments/user/${userId}`);
      const assignmentsResult = await assignmentsResponse.json();
      if (assignmentsResult.success && assignmentsResult.data.length > 0) {
        for (const assignment of assignmentsResult.data) {
          await fetch(`http://localhost:3000/api/assignments/${assignment.id}`, {
            method: 'DELETE'
          });
        }
      }

      // 4. Delete assignment_submissions
      const submissionsResponse = await fetch(`http://localhost:3000/api/assignment-submissions/user/${userId}`);
      const submissionsResult = await submissionsResponse.json();
      if (submissionsResult.success && submissionsResult.data.length > 0) {
        for (const submission of submissionsResult.data) {
          await fetch(`http://localhost:3000/api/assignment-submissions/${submission.id}`, {
            method: 'DELETE'
          });
        }
      }

      // 5. Delete user_statistics
      await fetch(`http://localhost:3000/api/statistics/user/${userId}`, {
        method: 'DELETE'
      });

      // 6. Delete attendance records
      await fetch(`http://localhost:3000/api/attendance/user/${userId}/all`, {
        method: 'DELETE'
      });

      // 7. Delete news_posts
      const newsResponse = await fetch(`http://localhost:3000/api/news/author/${userId}`);
      const newsResult = await newsResponse.json();
      if (newsResult.success && newsResult.data.length > 0) {
        for (const news of newsResult.data) {
          await fetch(`http://localhost:3000/api/news/${news.id}`, {
            method: 'DELETE'
          });
        }
      }

      // 8. Delete events
      const eventsResponse = await fetch(`http://localhost:3000/api/events/creator/${userId}`);
      const eventsResult = await eventsResponse.json();
      if (eventsResult.success && eventsResult.data.length > 0) {
        for (const event of eventsResult.data) {
          await fetch(`http://localhost:3000/api/events/${event.id}`, {
            method: 'DELETE'
          });
        }
      }

      // 9. Finally, delete the user
      const deleteUserResponse = await fetch(`http://localhost:3000/api/users/${userId}`, {
        method: 'DELETE'
      });

      const deleteUserResult = await deleteUserResponse.json();

      if (deleteUserResult.success) {
          organizationData.employees_count--;
           const deleteUserResponse = await fetch(`http://localhost:3000/api/organizations/${organizationData.id}`, {
              method: 'PUT',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                ...organizationData
              })
            });

        onError('Employee dismissed successfully', 'success');
        setShowDismissModal(false);
        setShowEmployeeModal(false);
        fetchEmployees();
      } else {
        onError('Failed to dismiss employee', 'error');
      }

    } catch (error) {
      console.error('Error dismissing employee:', error);
      onError('Failed to dismiss employee', 'error');
    }
  };

  // Handle department appearance
  const handleDepartmentAppearanceClick = () => {
    setAppearanceData({
      header_background_color: departmentData.header_background_color || '#667eea',
      header_display_type: departmentData.header_display_type || 'both',
      header_position: departmentData.header_position || 'center',
      logo_size: departmentData.logo_size || 50,
      sidebar_position: departmentData.sidebar_position || 'left',
      layout_color_hover: departmentData.layout_color_hover || '#3498db',
      layout_color_clicked: departmentData.layout_color_clicked || '#2980b9',
      layout_color_selected: departmentData.layout_color_selected || '#667eea'
    });
    setAppearancePassword('');
    setShowAppearanceModal(true);
  };

  // Confirm appearance changes
  const handleConfirmAppearanceChange = async () => {
    if (!appearancePassword.trim()) {
      onError('Password is required to make changes', 'error');
      return;
    }

    try {
      // Verify password
      const verifyResponse = await fetch('http://localhost:3000/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: userData.email,
          password: appearancePassword
        })
      });

      const verifyResult = await verifyResponse.json();

      if (!verifyResult.success) {
        onError('Incorrect password', 'error');
        return;
      }

      // Update department
      const updateResponse = await fetch(`http://localhost:3000/api/departments/${departmentData.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...appearanceData,
          updated_at: new Date().toISOString()
        })
      });

      const updateResult = await updateResponse.json();

      if (updateResult.success) {
        onError('Department appearance updated successfully!', 'success');
        setShowAppearanceModal(false);
        onUpdate();
      } else {
        onError('Failed to update appearance', 'error');
      }

    } catch (error) {
      console.error('Error updating appearance:', error);
      onError('Failed to update appearance', 'error');
    }
  };

  // Handle organization management
  const handleManageOrganizationClick = () => {
    setOrgEditData({
      unique_id: organizationData.unique_id,
      name: organizationData.name,
      logo_url: organizationData.logo_url || '',
      industry: organizationData.industry,
      description: organizationData.description || '',
      founding_date: organizationData.founding_date ? organizationData.founding_date.split('T')[0] : '',
      website: organizationData.website || '',
      linkedin_url: organizationData.linkedin_url || '',
      twitter_url: organizationData.twitter_url || '',
      contact_email: organizationData.contact_email
    });
    setOrgPassword('');
    setOrgUniqueCode('');
    setShowOrgModal(true);
  };

  // Confirm organization changes
  const handleConfirmOrgChange = async () => {
    if (!orgPassword.trim() || !orgUniqueCode.trim()) {
      onError('CEO password and unique code are required', 'error');
      return;
    }

    try {
      // Verify credentials
      const verifyResponse = await fetch('http://localhost:3000/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: organizationData.ceo_email,
          password: orgPassword,
          uniqueCode: orgUniqueCode
        })
      });

      const verifyResult = await verifyResponse.json();

      if (!verifyResult.success) {
        onError('Incorrect credentials', 'error');
        return;
      }

      // Update organization
      const updateResponse = await fetch(`http://localhost:3000/api/organizations/${organizationData.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...orgEditData,
          updated_at: new Date().toISOString()
        })
      });

      const updateResult = await updateResponse.json();

      if (updateResult.success) {
        onError('Organization updated successfully!', 'success');
        setShowOrgModal(false);
        onUpdate();
      } else {
        onError('Failed to update organization', 'error');
      }

    } catch (error) {
      console.error('Error updating organization:', error);
      onError('Failed to update organization', 'error');
    }
  };

  // Handle department click (CEO)
  const handleDepartmentClick = async (department) => {
    setSelectedDepartment(department);
    setShowDepartmentModal(true);
  };

  // Handle department manager change
  const handleChangeDepartmentManager = async (newManagerId) => {
    try {
      const response = await fetch(`http://localhost:3000/api/departments/${selectedDepartment.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          manager_id: parseInt(newManagerId),
          updated_at: new Date().toISOString()
        })
      });

      const result = await response.json();

      if (result.success) {
        onError('Manager changed successfully!', 'success');
        setShowDepartmentModal(false);
        fetchDepartments();
      } else {
        onError('Failed to change manager', 'error');
      }
    } catch (error) {
      console.error('Error changing manager:', error);
      onError('Failed to change manager', 'error');
    }
  };

  // Handle dismiss department (CASCADE DELETE)
  const handleDismissDepartment = async () => {
    if (!window.confirm('⚠️ WARNING: Deleting this department will permanently delete all employees, roles, and related data. This action cannot be undone. Are you sure?')) {
      return;
    }

    try {
      const deptId = selectedDepartment.id;

      // 1. Get all users from this department
      const usersResponse = await fetch(`http://localhost:3000/api/users/department/${deptId}`);
      const usersResult = await usersResponse.json();

      if (usersResult.success && usersResult.data.length > 0) {
        // Delete each user with cascade
        for (const user of usersResult.data) {
          // Use the same dismiss procedure as for individual employees
          await dismissEmployeeCascade(user.id);
        }
      }

      // 2. Delete all roles of this department
      const rolesResponse = await fetch(`http://localhost:3000/api/roles/department/${deptId}`);
      const rolesResult = await rolesResponse.json();

      if (rolesResult.success && rolesResult.data.length > 0) {
        for (const role of rolesResult.data) {
          await fetch(`http://localhost:3000/api/roles/${role.id}`, {
            method: 'DELETE'
          });
        }
      }

      // 3. Delete the department itself
      const deleteDeptResponse = await fetch(`http://localhost:3000/api/departments/${deptId}`, {
        method: 'DELETE'
      });

      const deleteDeptResult = await deleteDeptResponse.json();

      if (deleteDeptResult.success) {
        onError('Department dismissed successfully', 'success');
        setShowDepartmentModal(false);
        fetchDepartments();
        onUpdate();
      } else {
        onError('Failed to dismiss department', 'error');
      }

    } catch (error) {
      console.error('Error dismissing department:', error);
      onError('Failed to dismiss department', 'error');
    }
  };

  // Helper function for cascade delete of employee
  const dismissEmployeeCascade = async (userId) => {
    // Same logic as handleConfirmDismiss but without UI updates
    try {
      await fetch(`http://localhost:3000/api/assignment-users/user/${userId}/all`, { method: 'DELETE' });
      
      const relocationsResponse = await fetch(`http://localhost:3000/api/assignment-relocations/user/${userId}`);
      const relocationsResult = await relocationsResponse.json();
      if (relocationsResult.success && relocationsResult.data.length > 0) {
        for (const relocation of relocationsResult.data) {
          await fetch(`http://localhost:3000/api/assignment-relocations/${relocation.id}`, { method: 'DELETE' });
        }
      }

      const assignmentsResponse = await fetch(`http://localhost:3000/api/assignments/user/${userId}`);
      const assignmentsResult = await assignmentsResponse.json();
      if (assignmentsResult.success && assignmentsResult.data.length > 0) {
        for (const assignment of assignmentsResult.data) {
          await fetch(`http://localhost:3000/api/assignments/${assignment.id}`, { method: 'DELETE' });
        }
      }

      const submissionsResponse = await fetch(`http://localhost:3000/api/assignment-submissions/user/${userId}`);
      const submissionsResult = await submissionsResponse.json();
      if (submissionsResult.success && submissionsResult.data.length > 0) {
        for (const submission of submissionsResult.data) {
          await fetch(`http://localhost:3000/api/assignment-submissions/${submission.id}`, { method: 'DELETE' });
        }
      }

      await fetch(`http://localhost:3000/api/statistics/user/${userId}`, { method: 'DELETE' });
      await fetch(`http://localhost:3000/api/attendance/user/${userId}/all`, { method: 'DELETE' });

      const newsResponse = await fetch(`http://localhost:3000/api/news/author/${userId}`);
      const newsResult = await newsResponse.json();
      if (newsResult.success && newsResult.data.length > 0) {
        for (const news of newsResult.data) {
          await fetch(`http://localhost:3000/api/news/${news.id}`, { method: 'DELETE' });
        }
      }

      const eventsResponse = await fetch(`http://localhost:3000/api/events/creator/${userId}`);
      const eventsResult = await eventsResponse.json();
      if (eventsResult.success && eventsResult.data.length > 0) {
        for (const event of eventsResult.data) {
          await fetch(`http://localhost:3000/api/events/${event.id}`, { method: 'DELETE' });
        }
      }

      await fetch(`http://localhost:3000/api/users/${userId}`, { method: 'DELETE' });
    } catch (error) {
      console.error('Error in cascade delete:', error);
    }
  };

  // Handle role click (CEO)
  const handleRoleClick = (role) => {
    setSelectedRole(role);
    setShowRoleModal(true);
  };

  // Handle delete role
  const handleDeleteRole = async () => {
    try {
      const response = await fetch(`http://localhost:3000/api/roles/${selectedRole.id}`, {
        method: 'DELETE'
      });

      const result = await response.json();

      if (result.success) {
        onError('Role deleted successfully', 'success');
        setShowRoleModal(false);
        fetchAllRoles();
      } else {
        onError('Failed to delete role', 'error');
      }
    } catch (error) {
      console.error('Error deleting role:', error);
      onError('Failed to delete role', 'error');
    }
  };

  // Render main view with options
  if (activeView === 'main') {
    return (
      <div className="administration-section">
        <div className="section-header">
          <h2>Administration</h2>
          <p className="section-subtitle">Manage your {isCEO ? 'organization' : 'department'}</p>
        </div>

        <div className="admin-options-grid">
          {isManager && !isCEO && (
            <>
              <div className="admin-option-card" onClick={handleDepartmentAppearanceClick}>
                <h3>Department Appearance</h3>
                <p>Customize the look and feel of your department</p>
              </div>

              <div className="admin-option-card" onClick={() => setActiveView('manage-employees')}>
                <h3>Manage Department Employees</h3>
                <p>View and manage employees in your department</p>
              </div>
            </>
          )}

          {isCEO && (
            <>
              <div className="admin-option-card" onClick={() => setActiveView('manage-employees')}>
                <h3>Manage Organization Employees</h3>
                <p>View and manage all employees in your organization</p>
              </div>

              <div className="admin-option-card" onClick={handleManageOrganizationClick}>
                <h3>Manage Organization</h3>
                <p>Edit organization details and settings</p>
              </div>
            </>
          )}
        </div>

        {/* Department Appearance Modal */}
        {showAppearanceModal && (
          <DepartmentAppearanceModal
            appearanceData={appearanceData}
            setAppearanceData={setAppearanceData}
            password={appearancePassword}
            setPassword={setAppearancePassword}
            onConfirm={handleConfirmAppearanceChange}
            onCancel={() => setShowAppearanceModal(false)}
          />
        )}

        {/* Organization Modal */}
        {showOrgModal && (
          <OrganizationModal
            orgData={orgEditData}
            setOrgData={setOrgEditData}
            password={orgPassword}
            setPassword={setOrgPassword}
            uniqueCode={orgUniqueCode}
            setUniqueCode={setOrgUniqueCode}
            onConfirm={handleConfirmOrgChange}
            onCancel={() => setShowOrgModal(false)}
          />
        )}
      </div>
    );
  }

  // Render employees management view
  if (activeView === 'manage-employees') {
    return (
      <ManageEmployeesView
        employees={employees}
        loading={loading}
        isCEO={isCEO}
        onBack={() => setActiveView('main')}
        onEmployeeClick={handleEmployeeClick}
        showEmployeeModal={showEmployeeModal}
        selectedEmployee={selectedEmployee}
        onCloseModal={() => setShowEmployeeModal(false)}
        onRoleChange={handleRoleChangeClick}
        onDismiss={handleDismissClick}
        showRoleChangeModal={showRoleChangeModal}
        roles={roles}
        onConfirmRoleChange={handleConfirmRoleChange}
        onCancelRoleChange={() => setShowRoleChangeModal(false)}
        showDismissModal={showDismissModal}
        dismissReason={dismissReason}
        setDismissReason={setDismissReason}
        onConfirmDismiss={handleConfirmDismiss}
        onCancelDismiss={() => setShowDismissModal(false)}
      />
    );
  }
  return null;
}

// Department Appearance Modal Component
function DepartmentAppearanceModal({ 
  appearanceData, 
  setAppearanceData, 
  password, 
  setPassword, 
  onConfirm, 
  onCancel 
}) {
  return (
    <div className="modal-overlay" onClick={onCancel}>
      <div className="modal-content large-modal" onClick={e => e.stopPropagation()}>
        <h2>Department Appearance</h2>
        
        <div className="appearance-form">
          <div className="form-group">
            <label>Header Background Color</label>
            <input 
              type="color"
              value={appearanceData.header_background_color}
              onChange={(e) => setAppearanceData({...appearanceData, header_background_color: e.target.value})}
            />
          </div>

          <div className="form-group">
            <label>Header Display Type</label>
            <select 
              value={appearanceData.header_display_type}
              onChange={(e) => setAppearanceData({...appearanceData, header_display_type: e.target.value})}
            >
              <option value="both">Logo + Department Name</option>
              <option value="dep_name">Department Name Only</option>
              <option value="none">None</option>
            </select>
          </div>

          <div className="form-group">
            <label>Header Position</label>
            <select 
              value={appearanceData.header_position}
              onChange={(e) => setAppearanceData({...appearanceData, header_position: e.target.value})}
            >
              <option value="left">Left</option>
              <option value="center">Center</option>
              <option value="right">Right</option>
            </select>
          </div>

          <div className="form-group">
            <label>Logo Size (px)</label>
            <input 
              type="number"
              value={appearanceData.logo_size}
              onChange={(e) => setAppearanceData({...appearanceData, logo_size: parseInt(e.target.value)})}
              min="30"
              max="100"
            />
          </div>

          <div className="form-group">
            <label>Sidebar Position</label>
            <select 
              value={appearanceData.sidebar_position}
              onChange={(e) => setAppearanceData({...appearanceData, sidebar_position: e.target.value})}
            >
              <option value="left">Left</option>
              <option value="right">Right</option>
            </select>
          </div>

          <div className="form-group">
            <label>Layout Color - Hover</label>
            <input 
              type="color"
              value={appearanceData.layout_color_hover}
              onChange={(e) => setAppearanceData({...appearanceData, layout_color_hover: e.target.value})}
            />
          </div>

          <div className="form-group">
            <label>Layout Color - Clicked</label>
            <input 
              type="color"
              value={appearanceData.layout_color_clicked}
              onChange={(e) => setAppearanceData({...appearanceData, layout_color_clicked: e.target.value})}
            />
          </div>

          <div className="form-group">
            <label>Layout Color - Selected</label>
            <input 
              type="color"
              value={appearanceData.layout_color_selected}
              onChange={(e) => setAppearanceData({...appearanceData, layout_color_selected: e.target.value})}
            />
          </div>

          <div className="form-group">
            <label>Password (to confirm changes)</label>
            <input 
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter your password"
            />
          </div>
        </div>

        <div className="modal-buttons">
          <button className="modal-cancel-button" onClick={onCancel}>Cancel</button>
          <button className="modal-confirm-button" onClick={onConfirm}>Apply Changes</button>
        </div>
      </div>
    </div>
  );
}

// Organization Modal Component
function OrganizationModal({ 
  orgData, 
  setOrgData, 
  password, 
  setPassword, 
  uniqueCode, 
  setUniqueCode, 
  onConfirm, 
  onCancel 
}) {
  return (
    <div className="modal-overlay" onClick={onCancel}>
      <div className="modal-content large-modal scrollable" onClick={e => e.stopPropagation()}>
        <h2>Manage Organization</h2>
        
        <div className="org-form">
          <div className="form-group">
            <label>Unique ID</label>
            <input 
              type="text"
              value={orgData.unique_id}
              onChange={(e) => setOrgData({...orgData, unique_id: e.target.value})}
            />
          </div>

          <div className="form-group">
            <label>Name</label>
            <input 
              type="text"
              value={orgData.name}
              onChange={(e) => setOrgData({...orgData, name: e.target.value})}
            />
          </div>

          <div className="form-group">
            <label>Logo URL</label>
            <input 
              type="text"
              value={orgData.logo_url}
              onChange={(e) => setOrgData({...orgData, logo_url: e.target.value})}
            />
          </div>

          <div className="form-group">
            <label>Industry</label>
            <input 
              type="text"
              value={orgData.industry}
              onChange={(e) => setOrgData({...orgData, industry: e.target.value})}
            />
          </div>

          <div className="form-group">
            <label>Description</label>
            <textarea 
              value={orgData.description}
              onChange={(e) => setOrgData({...orgData, description: e.target.value})}
              rows="4"
            />
          </div>

          <div className="form-group">
            <label>Founding Date</label>
            <input 
              type="date"
              value={orgData.founding_date}
              onChange={(e) => setOrgData({...orgData, founding_date: e.target.value})}
            />
          </div>

          <div className="form-group">
            <label>Website</label>
            <input 
              type="text"
              value={orgData.website}
              onChange={(e) => setOrgData({...orgData, website: e.target.value})}
            />
          </div>

          <div className="form-group">
            <label>LinkedIn URL</label>
            <input 
              type="text"
              value={orgData.linkedin_url}
              onChange={(e) => setOrgData({...orgData, linkedin_url: e.target.value})}
            />
          </div>

          <div className="form-group">
            <label>Twitter URL</label>
            <input 
              type="text"
              value={orgData.twitter_url}
              onChange={(e) => setOrgData({...orgData, twitter_url: e.target.value})}
            />
          </div>

          <div className="form-group">
            <label>Contact Email</label>
            <input 
              type="email"
              value={orgData.contact_email}
              onChange={(e) => setOrgData({...orgData, contact_email: e.target.value})}
            />
          </div>

          <div className="form-group-divider"></div>

          <div className="form-group">
            <label>CEO Password (to confirm changes)</label>
            <input 
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter CEO password"
            />
          </div>

          <div className="form-group">
            <label>Unique Code (to confirm changes)</label>
            <input 
              type="text"
              value={uniqueCode}
              onChange={(e) => setUniqueCode(e.target.value)}
              placeholder="Enter organization unique code"
            />
          </div>
        </div>

        <div className="modal-buttons">
          <button className="modal-cancel-button" onClick={onCancel}>Cancel</button>
          <button className="modal-confirm-button" onClick={onConfirm}>Save Changes</button>
        </div>
      </div>
    </div>
  );
}