import React, { useState, useEffect } from 'react';
import '../../styles/AssignmentPage.css';

export default function AssignmentsSection({ userId, departmentId, onError }) {
  const [activeTab, setActiveTab] = useState('current');
  const [sortBy, setSortBy] = useState('deadline');
  
  // Data states
  const [submittedAssignments, setSubmittedAssignments] = useState([]);
  const [relocatedAssignments, setRelocatedAssignments] = useState([]);
  const [currentAssignments, setCurrentAssignments] = useState([]);
  
  // UI states
  const [loading, setLoading] = useState(true);
  const [selectedItem, setSelectedItem] = useState(null);
  const [showDetailsModal, setShowDetailsModal] = useState(false);
  const [showSubmitDialog, setShowSubmitDialog] = useState(false);
  const [showRelocateDialog, setShowRelocateDialog] = useState(false);
  const [submitNotes, setSubmitNotes] = useState('');
  const [relocateUsers, setRelocateUsers] = useState([]);
  const [selectedRelocateUser, setSelectedRelocateUser] = useState('');
  const [relocateReason, setRelocateReason] = useState('');

  useEffect(() => {
    fetchAllData();
  }, [userId]);

  const fetchAllData = async () => {
    try {
      setLoading(true);
      await Promise.all([
        fetchSubmitted(),
        fetchRelocated(),
        fetchCurrent()
      ]);
    } catch (error) {
      console.error('Error fetching data:', error);
      onError('Failed to load assignments', 'error');
    } finally {
      setLoading(false);
    }
  };

  const fetchSubmitted = async () => {
    try {
      const response = await fetch(`http://localhost:3000/api/assignment-submissions/user/${userId}`);
      const result = await response.json();
      
      if (result.success) {
        const withDetails = await Promise.all(
          result.data.map(async (submission) => {
            const assignmentResponse = await fetch(`http://localhost:3000/api/assignments/${submission.assignment_id}`);
            const assignmentResult = await assignmentResponse.json();
            
            if (assignmentResult.success) {
              return {
                ...submission,
                assignment: assignmentResult.data
              };
            }
            return null;
          })
        );
        
        setSubmittedAssignments(withDetails.filter(item => item !== null));
      }
    } catch (error) {
      console.error('Error fetching submitted:', error);
    }
  };

  const fetchRelocated = async () => {
    try {
      const response = await fetch(`http://localhost:3000/api/assignment-relocations/user/${userId}`);
      const result = await response.json();
      
      if (result.success) {
        const withDetails = await Promise.all(
          result.data.filter(r => r.from_user_id === userId).map(async (relocation) => {
            const [assignmentRes, toUserRes, reviewedByRes] = await Promise.all([
              fetch(`http://localhost:3000/api/assignments/${relocation.assignment_id}`),
              relocation.to_user_id ? fetch(`http://localhost:3000/api/users/${relocation.to_user_id}`) : null,
              relocation.reviewed_by ? fetch(`http://localhost:3000/api/users/${relocation.reviewed_by}`) : null
            ]);
            
            const assignment = await assignmentRes.json();
            const toUser = toUserRes ? await toUserRes.json() : null;
            const reviewedBy = reviewedByRes ? await reviewedByRes.json() : null;
            
            return {
              ...relocation,
              assignment: assignment.success ? assignment.data : null,
              toUser: toUser?.success ? toUser.data : null,
              reviewedBy: reviewedBy?.success ? reviewedBy.data : null
            };
          })
        );
        
        setRelocatedAssignments(withDetails);
      }
    } catch (error) {
      console.error('Error fetching relocated:', error);
    }
  };

  const fetchCurrent = async () => {
    try {
      const response = await fetch(`http://localhost:3000/api/assignment-users/user/${userId}`);
      const result = await response.json();
      
      if (result.success) {
        const withDetails = await Promise.all(
          result.data.map(async (assignmentUser) => {
            const [assignmentRes, creatorRes] = await Promise.all([
              fetch(`http://localhost:3000/api/assignments/${assignmentUser.assignment_id}`),
              fetch(`http://localhost:3000/api/assignments/${assignmentUser.assignment_id}`)
            ]);
            
            const assignment = await assignmentRes.json();
            
            if (assignment.success) {
              const creatorResponse = await fetch(`http://localhost:3000/api/users/${assignment.data.created_by}`);
              const creator = await creatorResponse.json();
              
              return {
                ...assignmentUser,
                assignment: assignment.data,
                creator: creator.success ? creator.data : null
              };
            }
            return null;
          })
        );
        
        setCurrentAssignments(withDetails.filter(item => item !== null));
      }
    } catch (error) {
      console.error('Error fetching current:', error);
    }
  };

  const sortData = (data, field) => {
    return [...data].sort((a, b) => {
      if (field === 'priority') {
        const priorityOrder = { high: 1, medium: 2, low: 3 };
        const aAssignment = a.assignment || {};
        const bAssignment = b.assignment || {};
        return (priorityOrder[aAssignment.priority] || 4) - (priorityOrder[bAssignment.priority] || 4);
      } else if (field === 'deadline') {
        const aAssignment = a.assignment || {};
        const bAssignment = b.assignment || {};
        return new Date(aAssignment.deadline) - new Date(bAssignment.deadline);
      }
      return 0;
    });
  };

  const handleSubmit = async () => {
    if (!selectedItem) return;
    
    try {
      // Create submission
      const submitResponse = await fetch('http://localhost:3000/api/assignment-submissions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          assignment_id: selectedItem.assignment_id,
          user_id: userId,
          notes: submitNotes || null,
          submitted_at: new Date().toISOString()
        })
      });
      
      const submitResult = await submitResponse.json();
      
      if (!submitResult.success) {
        throw new Error('Failed to submit assignment');
      }
      
      // Delete from assignment_users
      await fetch(`http://localhost:3000/api/assignment-users/assignment/${selectedItem.assignment_id}/user/${userId}`, {
        method: 'DELETE'
      });
      
      onError('Assignment submitted successfully!', 'success');
      setShowDetailsModal(false);
      setShowSubmitDialog(false);
      setSubmitNotes('');
      setSelectedItem(null);
      fetchAllData();
    } catch (error) {
      console.error('Error submitting:', error);
      onError('Failed to submit assignment', 'error');
    }
  };

  const handleRelocate = async () => {
    if (!selectedItem || !selectedRelocateUser || !relocateReason.trim()) {
      onError('Please fill in all fields', 'warning');
      return;
    }
    
    try {
      const response = await fetch('http://localhost:3000/api/assignment-relocations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          assignment_id: selectedItem.assignment_id,
          department_id: departmentId,
          from_user_id: userId,
          to_user_id: selectedRelocateUser,
          reason: relocateReason,
          status: 'pending',
          requested_at: new Date().toISOString()
        })
      });
      
      const result = await response.json();
      
      if (result.success) {
        onError('Relocation request submitted successfully!', 'success');
        setShowDetailsModal(false);
        setShowRelocateDialog(false);
        setSelectedRelocateUser('');
        setRelocateReason('');
        setSelectedItem(null);
        fetchAllData();
      } else {
        throw new Error('Failed to create relocation request');
      }
    } catch (error) {
      console.error('Error relocating:', error);
      onError('Failed to submit relocation request', 'error');
    }
  };

  const fetchRelocateUsers = async (assignmentId) => {
    try {
      const response = await fetch(`http://localhost:3000/api/assignment-operations/department/${departmentId}/unassigned-users/${assignmentId}`);
      const result = await response.json();
      
      if (result.success) {
        const filteredUsers = result.data.filter(user => user.id !== userId);
        setRelocateUsers(filteredUsers);
      }
    } catch (error) {
      console.error('Error fetching relocate users:', error);
    }
  };

  const handleOpenDetails = (item) => {
    setSelectedItem(item);
    setShowDetailsModal(true);
    if (activeTab === 'current') {
      fetchRelocateUsers(item.assignment_id);
    }
  };

  const formatDate = (date) => {
    if (!date) return 'N/A';
    return new Date(date).toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric' 
    });
  };

  const formatDateTime = (date) => {
    if (!date) return 'N/A';
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

  const getStatusClass = (status) => {
    switch(status) {
      case 'approved': return 'status-approved';
      case 'rejected': return 'status-rejected';
      case 'pending': return 'status-pending';
      default: return '';
    }
  };

  const renderSubmittedTable = () => {
    const sortedData = sortData(submittedAssignments, sortBy);
    
    return (
      <div className="assignments-table-container">
        {sortedData.length === 0 ? (
          <div className="no-assignments">No submitted assignments</div>
        ) : (
          <table className="assignments-table">
            <thead>
              <tr>
                <th>Category</th>
                <th>Priority</th>
                <th>Deadline</th>
                <th>Submitted At</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {sortedData.map((item) => (
                <tr key={item.id}>
                  <td>{item.assignment?.category || 'N/A'}</td>
                  <td>
                    <span className={`priority-badge ${getPriorityClass(item.assignment?.priority)}`}>
                      {item.assignment?.priority || 'N/A'}
                    </span>
                  </td>
                  <td>{formatDate(item.assignment?.deadline)}</td>
                  <td>{formatDateTime(item.submitted_at)}</td>
                  <td>
                    <button 
                      className="view-button"
                      onClick={() => handleOpenDetails(item)}
                    >
                      View
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    );
  };

  const renderRelocatedTable = () => {
    const sortedData = sortData(relocatedAssignments, sortBy);
    
    return (
      <div className="assignments-table-container">
        {sortedData.length === 0 ? (
          <div className="no-assignments">No relocated assignments</div>
        ) : (
          <table className="assignments-table">
            <thead>
              <tr>
                <th>To User</th>
                <th>Category</th>
                <th>Priority</th>
                <th>Deadline</th>
                <th>Status</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {sortedData.map((item) => (
                <tr key={item.id}>
                  <td>{item.toUser ? `${item.toUser.first_name} ${item.toUser.last_name}` : 'Unassigned'}</td>
                  <td>{item.assignment?.category || 'N/A'}</td>
                  <td>
                    <span className={`priority-badge ${getPriorityClass(item.assignment?.priority)}`}>
                      {item.assignment?.priority || 'N/A'}
                    </span>
                  </td>
                  <td>{formatDate(item.assignment?.deadline)}</td>
                  <td>
                    <span className={`status-badge ${getStatusClass(item.status)}`}>
                      {item.status}
                    </span>
                  </td>
                  <td>
                    <button 
                      className="view-button"
                      onClick={() => handleOpenDetails(item)}
                    >
                      More
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    );
  };

  const renderCurrentTable = () => {
    const sortedData = sortData(currentAssignments, sortBy);
    
    return (
      <div className="assignments-table-container">
        {sortedData.length === 0 ? (
          <div className="no-assignments">No current assignments</div>
        ) : (
          <table className="assignments-table">
            <thead>
              <tr>
                <th>Created By</th>
                <th>Category</th>
                <th>Priority</th>
                <th>Deadline</th>
                <th>Created At</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {sortedData.map((item) => (
                <tr key={item.id}>
                  <td>{item.creator ? `${item.creator.first_name} ${item.creator.last_name}` : 'N/A'}</td>
                  <td>{item.assignment?.category || 'N/A'}</td>
                  <td>
                    <span className={`priority-badge ${getPriorityClass(item.assignment?.priority)}`}>
                      {item.assignment?.priority || 'N/A'}
                    </span>
                  </td>
                  <td>{formatDate(item.assignment?.deadline)}</td>
                  <td>{formatDate(item.assignment?.created_at)}</td>
                  <td>
                    <button 
                      className="view-button"
                      onClick={() => handleOpenDetails(item)}
                    >
                      View
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    );
  };

  if (loading) {
    return (
      <div className="section-loading">
        <div className="loading-spinner"></div>
        <p>Loading assignments...</p>
      </div>
    );
  }

  return (
    <div className="assignments-section">
      <div className="section-header">
        <h2>My Assignments</h2>
        <p className="section-subtitle">Manage your assigned tasks and submissions</p>
      </div>

      {/* Tabs */}
      <div className="assignments-tabs">
        <button 
          className={`tab-button ${activeTab === 'current' ? 'active' : ''}`}
          onClick={() => setActiveTab('current')}
        >
          Current
        </button>
        <button 
          className={`tab-button ${activeTab === 'submitted' ? 'active' : ''}`}
          onClick={() => setActiveTab('submitted')}
        >
          Submitted
        </button>
        <button 
          className={`tab-button ${activeTab === 'relocated' ? 'active' : ''}`}
          onClick={() => setActiveTab('relocated')}
        >
          Relocated
        </button>
      </div>

      {/* Sort Options */}
      <div className="sort-options">
        <label>Sort by:</label>
        <button 
          className={`sort-button ${sortBy === 'priority' ? 'active' : ''}`}
          onClick={() => setSortBy('priority')}
        >
          Priority
        </button>
        <button 
          className={`sort-button ${sortBy === 'deadline' ? 'active' : ''}`}
          onClick={() => setSortBy('deadline')}
        >
          Deadline
        </button>
      </div>

      {/* Tables */}
      {activeTab === 'submitted' && renderSubmittedTable()}
      {activeTab === 'relocated' && renderRelocatedTable()}
      {activeTab === 'current' && renderCurrentTable()}

      {/* Details Modal - Submitted */}
      {showDetailsModal && activeTab === 'submitted' && selectedItem && (
        <div className="modal-overlay" onClick={() => setShowDetailsModal(false)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <h2>Assignment Details</h2>
            
            <div className="detail-item">
              <label>Title:</label>
              <span>{selectedItem.assignment?.title}</span>
            </div>
            
            <div className="detail-item">
              <label>Notes:</label>
              <div className="notes-box">{selectedItem.notes || 'No notes provided'}</div>
            </div>
            
            <button className="close-button" onClick={() => setShowDetailsModal(false)}>
              Close
            </button>
          </div>
        </div>
      )}

      {/* Details Modal - Relocated */}
      {showDetailsModal && activeTab === 'relocated' && selectedItem && (
        <div className="modal-overlay" onClick={() => setShowDetailsModal(false)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <h2>Relocation Details</h2>
            
            <div className="detail-item">
              <label>Title:</label>
              <span>{selectedItem.assignment?.title}</span>
            </div>
            
            <div className="detail-item">
              <label>To User:</label>
              <span>{selectedItem.toUser ? `${selectedItem.toUser.first_name} ${selectedItem.toUser.last_name}` : 'Unassigned'}</span>
            </div>
            
            <div className="detail-item">
              <label>Reason:</label>
              <div className="notes-box">{selectedItem.reason}</div>
            </div>
            
            <div className="detail-item">
              <label>Requested At:</label>
              <span>{formatDateTime(selectedItem.requested_at)}</span>
            </div>
            
            {selectedItem.reviewed_by && (
              <>
                <div className="detail-item">
                  <label>Reviewed By:</label>
                  <span>
                    {selectedItem.reviewedBy ? `${selectedItem.reviewedBy.first_name} ${selectedItem.reviewedBy.last_name}` : 'N/A'}
                  </span>
                </div>
                
                <div className="detail-item">
                  <label>Reviewed At:</label>
                  <span>{formatDateTime(selectedItem.reviewed_at)}</span>
                </div>
              </>
            )}
            
            <button className="close-button" onClick={() => setShowDetailsModal(false)}>
              Close
            </button>
          </div>
        </div>
      )}

      {/* Details Modal - Current */}
      {showDetailsModal && activeTab === 'current' && selectedItem && !showSubmitDialog && !showRelocateDialog && (
        <div className="modal-overlay" onClick={() => setShowDetailsModal(false)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <h2>Assignment Details</h2>
            
            <div className="detail-item">
              <label>Title:</label>
              <span>{selectedItem.assignment?.title}</span>
            </div>
            
            <div className="detail-item">
              <label>Description:</label>
              <div className="description-box">{selectedItem.assignment?.description || 'No description'}</div>
            </div>
            
            <div className="detail-item">
              <label>Deadline:</label>
              <span>{formatDate(selectedItem.assignment?.deadline)}</span>
            </div>
            
            <div className="detail-item">
              <label>Priority:</label>
              <span className={`priority-badge ${getPriorityClass(selectedItem.assignment?.priority)}`}>
                {selectedItem.assignment?.priority}
              </span>
            </div>
            
            <div className="modal-actions">
              <button 
                className="relocate-button"
                onClick={() => setShowRelocateDialog(true)}
              >
                Relocate
              </button>
              <button 
                className="close-button"
                onClick={() => setShowDetailsModal(false)}
              >
                Close
              </button>
              <button 
                className="submit-button"
                onClick={() => setShowSubmitDialog(true)}
              >
                Submit
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Submit Dialog */}
      {showSubmitDialog && (
        <div className="dialog-overlay">
          <div className="dialog-content slide-left">
            <h3>Submit Assignment</h3>
            
            <div className="form-group">
              <label>Notes (Optional):</label>
              <textarea
                value={submitNotes}
                onChange={(e) => setSubmitNotes(e.target.value)}
                placeholder="Add any notes about your submission..."
                rows="4"
              />
            </div>
            
            <div className="dialog-actions">
              <button 
                className="cancel-button"
                onClick={() => {
                  setShowSubmitDialog(false);
                  setSubmitNotes('');
                }}
              >
                Cancel
              </button>
              <button 
                className="confirm-button"
                onClick={handleSubmit}
              >
                Submit
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Relocate Dialog */}
      {showRelocateDialog && (
        <div className="dialog-overlay">
          <div className="dialog-content slide-right">
            <h3>Relocate Assignment</h3>
            
            <div className="form-group">
              <label>Select User:</label>
              <select
                value={selectedRelocateUser}
                onChange={(e) => setSelectedRelocateUser(e.target.value)}
              >
                <option value="">Choose a user...</option>
                {relocateUsers.map(user => (
                  <option key={user.id} value={user.id}>
                    {user.first_name} {user.last_name}
                  </option>
                ))}
              </select>
            </div>
            
            <div className="form-group">
              <label>Reason:</label>
              <textarea
                value={relocateReason}
                onChange={(e) => setRelocateReason(e.target.value)}
                placeholder="Explain why you want to relocate this assignment..."
                rows="4"
              />
            </div>
            
            <div className="dialog-actions">
              <button 
                className="cancel-button"
                onClick={() => {
                  setShowRelocateDialog(false);
                  setSelectedRelocateUser('');
                  setRelocateReason('');
                }}
              >
                Cancel
              </button>
              <button 
                className="confirm-button"
                onClick={handleRelocate}
              >
                Submit Request
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}