import React from 'react';
import '../../styles/SettingsPage.css';

export default function ManageEmployeesView({
  employees,
  loading,
  isCEO,
  onBack,
  onEmployeeClick,
  showEmployeeModal,
  selectedEmployee,
  onCloseModal,
  onRoleChange,
  onDismiss,
  showRoleChangeModal,
  roles,
  onConfirmRoleChange,
  onCancelRoleChange,
  showDismissModal,
  dismissReason,
  setDismissReason,
  onConfirmDismiss,
  onCancelDismiss
}) {
  if (loading) {
    return (
      <div className="section-loading">
        <div className="loading-spinner"></div>
        <p>Loading employees...</p>
      </div>
    );
  }

  return (
    <div className="manage-employees-view">
      <button className="back-button" onClick={onBack}>
      Back to Administration
      </button>

      <div className="section-header">
        <h2>{isCEO ? 'Organization' : 'Department'} Employees</h2>
        <p className="section-subtitle">Manage employee roles and assignments</p>
      </div>

      {employees.length === 0 ? (
        <div className="no-data">
          <p>No employees found</p>
        </div>
      ) : (
        <div className="employees-table-container">
          <table className="employees-table">
            <thead>
              <tr>
                <th>Full Name</th>
                <th>Email</th>
                <th>Role</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {employees.map((employee) => (
                <tr key={employee.id}>
                  <td>{employee.first_name} {employee.last_name}</td>
                  <td>{employee.email}</td>
                  <td>{employee.role_name || 'No Role'}</td>
                  <td>
                    <button 
                      className="table-action-button"
                      onClick={() => onEmployeeClick(employee)}
                    >
                      Manage
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Employee Modal */}
      {showEmployeeModal && selectedEmployee && (
        <div className="modal-overlay" onClick={onCloseModal}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <h2>Manage Employee</h2>
            
            <div className="employee-info-section">
              <div className="info-row">
                <label>Name:</label>
                <span>{selectedEmployee.first_name} {selectedEmployee.last_name}</span>
              </div>
              <div className="info-row">
                <label>Email:</label>
                <span>{selectedEmployee.email}</span>
              </div>
              <div className="info-row">
                <label>Current Role:</label>
                <span>{selectedEmployee.role_name || 'No Role'}</span>
              </div>
              {selectedEmployee.department_name && (
                <div className="info-row">
                  <label>Department:</label>
                  <span>{selectedEmployee.department_name}</span>
                </div>
              )}
            </div>

            <div className="modal-buttons vertical">
              <button className="modal-action-button" onClick={onRoleChange}>
                Change Role
              </button>
              <button className="modal-dismiss-button" onClick={onDismiss}>
                Dismiss Employee
              </button>
              <button className="modal-cancel-button" onClick={onCloseModal}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Role Change Modal */}
      {showRoleChangeModal && (
        <div className="modal-overlay" onClick={onCancelRoleChange}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <h2>Change Employee Role</h2>
            
            <div className="role-selection">
              <p>Select a new role for {selectedEmployee.first_name} {selectedEmployee.last_name}:</p>
              
              <div className="roles-list">
                {roles.map((role) => (
                  <div 
                    key={role.id} 
                    className={`role-item ${selectedEmployee.role_id === role.id ? 'current-role' : ''}`}
                    onClick={() => {
                      if (selectedEmployee.role_id !== role.id) {
                        onConfirmRoleChange(role.id);
                      }
                    }}
                  >
                    <h4>{role.name}</h4>
                    <p>Priority: {role.priority}</p>
                    {selectedEmployee.role_id === role.id && (
                      <span className="current-badge">Current Role</span>
                    )}
                  </div>
                ))}
              </div>
            </div>

            <div className="modal-buttons">
              <button className="modal-cancel-button" onClick={onCancelRoleChange}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Dismiss Modal */}
      {showDismissModal && (
        <div className="modal-overlay" onClick={onCancelDismiss}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <h2>Dismiss Employee</h2>
            <p className="warning-text">
              ⚠️ This will permanently delete all data associated with this employee, including assignments, submissions, and attendance records.
            </p>

            <div className="form-group">
              <label>Reason for Dismissal *</label>
              <textarea 
                value={dismissReason}
                onChange={(e) => setDismissReason(e.target.value)}
                placeholder="Enter reason for dismissal"
                rows="4"
              />
            </div>

            <div className="modal-buttons">
              <button className="modal-cancel-button" onClick={onCancelDismiss}>
                Cancel
              </button>
              <button className="modal-delete-button" onClick={onConfirmDismiss}>
                Confirm Dismissal
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}