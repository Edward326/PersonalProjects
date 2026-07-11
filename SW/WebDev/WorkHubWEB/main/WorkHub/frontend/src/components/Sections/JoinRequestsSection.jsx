import React, { useState, useEffect } from 'react';
import '../../styles/RequestsPage.css';
import '../../styles/Notifications.css';

export default function JoinRequestsSection({ organizationId, userId, isCEO, sidebarPosition, onError }) {
  const [joinRequests, setJoinRequests] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedRequest, setSelectedRequest] = useState(null);
  const [showModal, setShowModal] = useState(false);
  const [departments, setDepartments] = useState([]);
  const [roles, setRoles] = useState([]);
  const [selectedDepartment, setSelectedDepartment] = useState('');
  const [selectedRole, setSelectedRole] = useState('');
  const [userDetails, setUserDetails] = useState(null);

  useEffect(() => {
    fetchJoinRequests();
    fetchDepartments();
  }, [organizationId]);

  const fetchJoinRequests = async () => {
    try {
      setLoading(true);
      const response = await fetch(`http://localhost:3000/api/join-requests?organizationId=${organizationId}`);
      const result = await response.json();
      
      if (result.success) {
        setJoinRequests(result.data);
      } else {
        onError('Failed to load join requests', 'error');
      }
    } catch (error) {
      console.error('Error fetching join requests:', error);
      onError('Failed to load join requests', 'error');
    } finally {
      setLoading(false);
    }
  };

  const fetchDepartments = async () => {
    try {
      const response = await fetch(`http://localhost:3000/api/departments/organization/${organizationId}`);
      const result = await response.json();
      
      if (result.success) {
        setDepartments(result.data);
      }
    } catch (error) {
      console.error('Error fetching departments:', error);
    }
  };

  const fetchRolesForDepartment = async (departmentId) => {
    try {
      const response = await fetch(`http://localhost:3000/api/roles/department/${departmentId}`);
      const result = await response.json();
      
      if (result.success) {
        setRoles(result.data);
      }
    } catch (error) {
      console.error('Error fetching roles:', error);
    }
  };

  const fetchUserDetails = async (userId) => {
    try {
      const response = await fetch(`http://localhost:3000/api/users/${userId}`);
      const result = await response.json();
      
      if (result.success) {
        setUserDetails(result.data);
      }
    } catch (error) {
      console.error('Error fetching user details:', error);
    }
  };

  const handleRequestClick = async (request) => {
    setSelectedRequest(request);
    await fetchUserDetails(request.user_id);
    setShowModal(true);
    setSelectedDepartment('');
    setSelectedRole('');
    setRoles([]);
  };

  const handleDepartmentChange = (e) => {
    const deptId = e.target.value;
    setSelectedDepartment(deptId);
    setSelectedRole('');
    if (deptId) {
      fetchRolesForDepartment(deptId);
    } else {
      setRoles([]);
    }
  };

  const handleApprove = async () => {
    if (!selectedDepartment || !selectedRole) {
      onError('Please select both department and role', 'warning');
      return;
    }

    try {
      // Update user with department, role, approval status, and hire date
      const updateUserResponse = await fetch(`http://localhost:3000/api/users/${selectedRequest.user_id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          department_id: parseInt(selectedDepartment),
          role_id: parseInt(selectedRole),
          is_approved: true,
          hire_date: new Date().toISOString().split('T')[0]
        })
      });

      const updateResult = await updateUserResponse.json();

      if (!updateResult.success) {
        throw new Error('Failed to update user');
      }

      // Delete the join request
      const deleteResponse = await fetch(`http://localhost:3000/api/join-requests/user/${selectedRequest.user_id}/organization/${selectedRequest.organization_id}`, {
        method: 'DELETE'
      });

      const deleteResult = await deleteResponse.json();

      if (!deleteResult.success) {
        throw new Error('Failed to delete join request');
      }

      onError('Join request approved successfully!', 'success');
      setShowModal(false);
      setSelectedRequest(null);
      setUserDetails(null);
      fetchJoinRequests(); // Refresh the list
    } catch (error) {
      console.error('Error approving join request:', error);
      onError('Failed to approve join request', 'error');
    }
  };

  const handleDeny = async () => {
    try {
      // Delete the join request
      const deleteJoinRequestResponse = await fetch(`http://localhost:3000/api/join-requests/user/${selectedRequest.user_id}/organization/${selectedRequest.organization_id}`, {
        method: 'DELETE'
      });

      const deleteJoinRequestResult = await deleteJoinRequestResponse.json();

      if (!deleteJoinRequestResult.success) {
        throw new Error('Failed to delete join request');
      }

      // Delete the user from users table
      const deleteUserResponse = await fetch(`http://localhost:3000/api/users/${selectedRequest.user_id}`, {
        method: 'DELETE'
      });

      const deleteUserResult = await deleteUserResponse.json();

      if (!deleteUserResult.success) {
        throw new Error('Failed to delete user');
      }

      onError('Join request denied and user removed', 'success');
      setShowModal(false);
      setSelectedRequest(null);
      setUserDetails(null);
      fetchJoinRequests(); // Refresh the list
    } catch (error) {
      console.error('Error denying join request:', error);
      onError('Failed to deny join request', 'error');
    }
  };

  const formatDate = (date) => {
    return new Date(date).toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  if (loading) {
    return (
      <div className="section-loading">
        <div className="loading-spinner"></div>
        <p>Loading join requests...</p>
      </div>
    );
  }

  return (
    <div className="join-requests-section">
      <div className="section-header">
        <h2>Join Requests</h2>
        <p className="section-subtitle">Review and approve employees joining your organization</p>
      </div>

      <div className="requests-list">
        {joinRequests.length === 0 ? (
          <div className="no-requests">
            <p>No pending join requests at the moment.</p>
          </div>
        ) : (
          joinRequests.map(request => (
            <div key={`${request.user_id}-${request.organization_id}`} className="request-item" onClick={() => handleRequestClick(request)}>
              <div className="request-info">
                <h3>{request.first_name} {request.last_name}</h3>
                <p className="request-email">{request.email}</p>
                <p className="request-date">Requested: {formatDate(request.requested_at)}</p>
              </div>
              <button className="review-button">Review</button>
            </div>
          ))
        )}
      </div>

      {/* Modal for reviewing join request */}
      {showModal && userDetails && (
        <div className="request-modal-overlay" onClick={() => setShowModal(false)}>
          <div className="request-modal" onClick={e => e.stopPropagation()}>
            <h2>Review Join Request</h2>
            
            <div className="modal-section">
              <h3>Applicant Information</h3>
              <div className="info-grid">
                <div className="info-item">
                  <label>Name:</label>
                  <span>{userDetails.first_name} {userDetails.last_name}</span>
                </div>
                <div className="info-item">
                  <label>Email:</label>
                  <span>{userDetails.email}</span>
                </div>
                <div className="info-item">
                  <label>Requested:</label>
                  <span>{formatDate(selectedRequest.requested_at)}</span>
                </div>
              </div>
            </div>

            <div className="modal-section">
              <h3>Assign Position</h3>
              <div className="form-group">
                <label>Department *</label>
                <select 
                  value={selectedDepartment} 
                  onChange={handleDepartmentChange}
                  className="modal-select"
                >
                  <option value="">Select Department</option>
                  {departments.map(dept => (
                    <option key={dept.id} value={dept.id}>{dept.name}</option>
                  ))}
                </select>
              </div>

              <div className="form-group">
                <label>Role *</label>
                <select 
                  value={selectedRole} 
                  onChange={(e) => setSelectedRole(e.target.value)}
                  className="modal-select"
                  disabled={!selectedDepartment}
                >
                  <option value="">Select Role</option>
                  {roles.map(role => (
                    <option key={role.id} value={role.id}>{role.name}</option>
                  ))}
                </select>
              </div>
            </div>

            <div className="modal-buttons">
              <button className="deny-button" onClick={handleDeny}>
                Deny Request
              </button>
              <button className="approve-button" onClick={handleApprove}>
                Approve Request
              </button>
            </div>

            <button className="close-modal-button" onClick={() => setShowModal(false)}>
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}