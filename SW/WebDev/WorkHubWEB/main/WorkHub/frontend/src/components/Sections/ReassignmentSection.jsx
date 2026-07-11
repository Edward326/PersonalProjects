import React, { useState, useEffect } from 'react';
import '../../styles/RequestsPage.css';
import '../../styles/Notifications.css';

export default function ReassignmentSection({ departmentId, userId, sidebarPosition, onError }) {
  const [reassignmentRequests, setReassignmentRequests] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedRequest, setSelectedRequest] = useState(null);
  const [showModal, setShowModal] = useState(false);
  const [assignmentDetails, setAssignmentDetails] = useState(null);
  const [fromUserDetails, setFromUserDetails] = useState(null);
  const [toUserDetails, setToUserDetails] = useState(null);

  useEffect(() => {
    fetchReassignmentRequests();
  }, [departmentId]);

  const fetchReassignmentRequests = async () => {
    try {
      setLoading(true);
      const response = await fetch(`http://localhost:3000/api/assignment-relocations/department/${departmentId}?status=pending`);
      const result = await response.json();
      
      if (result.success) {
        setReassignmentRequests(result.data);
      } else {
        onError('Failed to load reassignment requests', 'error');
      }
    } catch (error) {
      console.error('Error fetching reassignment requests:', error);
      onError('Failed to load reassignment requests', 'error');
    } finally {
      setLoading(false);
    }
  };

  const fetchAssignmentDetails = async (assignmentId) => {
    try {
      const response = await fetch(`http://localhost:3000/api/assignments/${assignmentId}`);
      const result = await response.json();
      
      if (result.success) {
        setAssignmentDetails(result.data);
      }
    } catch (error) {
      console.error('Error fetching assignment details:', error);
    }
  };

  const fetchUserDetails = async (userId, isFromUser) => {
    try {
      const response = await fetch(`http://localhost:3000/api/users/${userId}`);
      const result = await response.json();
      
      if (result.success) {
        if (isFromUser) {
          setFromUserDetails(result.data);
        } else {
          setToUserDetails(result.data);
        }
      }
    } catch (error) {
      console.error('Error fetching user details:', error);
    }
  };

  const handleRequestClick = async (request) => {
    setSelectedRequest(request);
    setShowModal(true);
    
    // Fetch all necessary details
    await fetchAssignmentDetails(request.assignment_id);
    await fetchUserDetails(request.from_user_id, true);
    if (request.to_user_id) {
      await fetchUserDetails(request.to_user_id, false);
    }
  };

  const handleApprove = async () => {
    try {
      // 1. Update assignment relocation status to 'approved'
      const updateResponse = await fetch(`http://localhost:3000/api/assignment-relocations/${selectedRequest.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          status: 'approved',
          reviewed_at: new Date().toISOString(),
          reviewed_by: userId
        })
      });

      const updateResult = await updateResponse.json();

      if (!updateResult.success) {
        throw new Error('Failed to update reassignment status');
      }

      // 2. Create new assignment_users record for to_user_id
      if (selectedRequest.to_user_id) {
        const createResponse = await fetch(`http://localhost:3000/api/assignment-users`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            assignment_id: selectedRequest.assignment_id,
            user_id: selectedRequest.to_user_id,
            assigned_at: new Date().toISOString()
          })
        });

        const createResult = await createResponse.json();

        if (!createResult.success) {
          throw new Error('Failed to create new assignment user');
        }
      }

      // 3. Delete assignment_users record for from_user_id
      const deleteResponse = await fetch(
        `http://localhost:3000/api/assignment-users/assignment/${selectedRequest.assignment_id}/user/${selectedRequest.from_user_id}`, 
        {
          method: 'DELETE'
        }
      );

      const deleteResult = await deleteResponse.json();

      if (!deleteResult.success) {
        throw new Error('Failed to remove previous assignment');
      }

      onError('Reassignment request approved successfully!', 'success');
      setShowModal(false);
      clearModalState();
      fetchReassignmentRequests(); // Refresh the list
    } catch (error) {
      console.error('Error approving reassignment:', error);
      onError('Failed to approve reassignment request', 'error');
    }
  };

  const handleDeny = async () => {
    try {
      const updateResponse = await fetch(`http://localhost:3000/api/assignment-relocations/${selectedRequest.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          status: 'rejected',
          reviewed_at: new Date().toISOString(),
          reviewed_by: userId
        })
      });

      const updateResult = await updateResponse.json();

      if (!updateResult.success) {
        throw new Error('Failed to update reassignment status');
      }

      onError('Reassignment request denied', 'success');
      setShowModal(false);
      clearModalState();
      fetchReassignmentRequests(); // Refresh the list
    } catch (error) {
      console.error('Error denying reassignment:', error);
      onError('Failed to deny reassignment request', 'error');
    }
  };

  const clearModalState = () => {
    setSelectedRequest(null);
    setAssignmentDetails(null);
    setFromUserDetails(null);
    setToUserDetails(null);
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

  const getPriorityClass = (priority) => {
    switch(priority) {
      case 'high': return 'priority-high';
      case 'medium': return 'priority-medium';
      case 'low': return 'priority-low';
      default: return '';
    }
  };

  if (loading) {
    return (
      <div className="section-loading">
        <div className="loading-spinner"></div>
        <p>Loading reassignment requests...</p>
      </div>
    );
  }

  return (
    <div className="reassignment-section">
      <div className="section-header">
        <h2>Task Reassignment Requests</h2>
        <p className="section-subtitle">Review and approve task reassignment requests from your team</p>
      </div>

      <div className="requests-list">
        {reassignmentRequests.length === 0 ? (
          <div className="no-requests">
            <p>No pending reassignment requests at the moment.</p>
          </div>
        ) : (
          reassignmentRequests.map(request => (
            <div key={request.id} className="request-item" onClick={() => handleRequestClick(request)}>
              <div className="request-info">
                <h3>Assignment #{request.assignment_id}</h3>
                <p className="request-detail">From: {request.from_user_name || 'Loading...'}</p>
                <p className="request-detail">To: {request.to_user_name || 'Unassigned'}</p>
                <p className="request-date">Requested: {formatDate(request.requested_at)}</p>
              </div>
              <button className="review-button">Review</button>
            </div>
          ))
        )}
      </div>

      {/* Modal for reviewing reassignment request */}
      {showModal && assignmentDetails && fromUserDetails && (
        <div className="request-modal-overlay" onClick={() => setShowModal(false)}>
          <div className="request-modal" onClick={e => e.stopPropagation()}>
            <h2>Review Reassignment Request</h2>
            
            <div className="modal-section">
              <h3>Assignment Details</h3>
              <div className="info-grid">
                <div className="info-item full-width">
                  <label>Title:</label>
                  <span>{assignmentDetails.title}</span>
                </div>
                <div className="info-item full-width">
                  <label>Description:</label>
                  <span className="description-text">{assignmentDetails.description || 'No description provided'}</span>
                </div>
                <div className="info-item">
                  <label>Deadline:</label>
                  <span>{formatDate(assignmentDetails.deadline)}</span>
                </div>
                <div className="info-item">
                  <label>Priority:</label>
                  <span className={`priority-badge ${getPriorityClass(assignmentDetails.priority)}`}>
                    {assignmentDetails.priority}
                  </span>
                </div>
              </div>
            </div>

            <div className="modal-section">
              <h3>Reassignment Information</h3>
              <div className="info-grid">
                <div className="info-item">
                  <label>From:</label>
                  <span>{fromUserDetails.first_name} {fromUserDetails.last_name}</span>
                </div>
                <div className="info-item">
                  <label>To:</label>
                  <span>{toUserDetails ? `${toUserDetails.first_name} ${toUserDetails.last_name}` : 'Unassigned'}</span>
                </div>
                <div className="info-item full-width">
                  <label>Reason:</label>
                  <span className="description-text">{selectedRequest.reason}</span>
                </div>
                <div className="info-item">
                  <label>Requested:</label>
                  <span>{formatDate(selectedRequest.requested_at)}</span>
                </div>
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

            <button className="close-modal-button" onClick={() => {
              setShowModal(false);
              clearModalState();
            }}>
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}