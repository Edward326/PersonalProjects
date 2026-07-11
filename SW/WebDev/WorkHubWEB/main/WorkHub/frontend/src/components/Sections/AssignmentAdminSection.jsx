import React, { useState, useEffect } from 'react';
import addIcon from '../../assets/Icons/plus.png';
import '../../styles/AssignmentPage.css';

export default function AssignmentAdminSection({ userId, departmentId, organizationId, onError }) {
  const [assignments, setAssignments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [sortBy, setSortBy] = useState('deadline');
  
  // Modals and dialogs
  const [selectedAssignment, setSelectedAssignment] = useState(null);
  const [showDetailsModal, setShowDetailsModal] = useState(false);
  const [showModifyDialog, setShowModifyDialog] = useState(false);
  const [showAssignedDialog, setShowAssignedDialog] = useState(false);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [createStep, setCreateStep] = useState(1); // 1: Details, 2: Assign Users
  
  // Modify form
  const [modifyForm, setModifyForm] = useState({
    title: '',
    description: '',
    category: '',
    deadline: '',
    priority: 'medium'
  });
  
  // Create form
  const [createForm, setCreateForm] = useState({
    title: '',
    description: '',
    category: '',
    deadline: '',
    priority: 'medium'
  });
  
  // Assignment users
  const [assignedUsers, setAssignedUsers] = useState([]);
  const [unassignedUsers, setUnassignedUsers] = useState([]);
  const [selectedUsersToRemove, setSelectedUsersToRemove] = useState([]);
  const [selectedUsersToAdd, setSelectedUsersToAdd] = useState([]);
  const [newAssignmentId, setNewAssignmentId] = useState(null);
  const [allDepartmentUsers, setAllDepartmentUsers] = useState([]);
  const [selectedNewUsers, setSelectedNewUsers] = useState([]);

  useEffect(() => {
    fetchAssignments();
    fetchDepartmentUsers();
  }, [departmentId]);

  const fetchAssignments = async () => {
    try {
      setLoading(true);
      const response = await fetch(`http://localhost:3000/api/assignment-operations/department/${departmentId}/with-counts`);
      const result = await response.json();
      
      if (result.success) {
        setAssignments(result.data);
      }
    } catch (error) {
      console.error('Error fetching assignments:', error);
      onError('Failed to load assignments', 'error');
    } finally {
      setLoading(false);
    }
  };

  const fetchDepartmentUsers = async () => {
    try {
      const response = await fetch(`http://localhost:3000/api/users/department/${departmentId}`);
      const result = await response.json();
      
      if (result.success) {
        // Exclude current user
        const filtered = result.data.filter(user => user.id !== userId);
        setAllDepartmentUsers(filtered);
      }
    } catch (error) {
      console.error('Error fetching users:', error);
    }
  };

  const fetchAssignedUsers = async (assignmentId) => {
    try {
      const response = await fetch(`http://localhost:3000/api/assignment-users/assignment/${assignmentId}`);
      const result = await response.json();
      
      if (result.success) {
        setAssignedUsers(result.data);
      }
    } catch (error) {
      console.error('Error fetching assigned users:', error);
    }
  };

  const fetchUnassignedUsers = async (assignmentId) => {
    try {
      const response = await fetch(`http://localhost:3000/api/assignment-operations/department/${departmentId}/unassigned-users/${assignmentId}`);
      const result = await response.json();
      
      if (result.success) {
        // Exclude current user
        const filtered = result.data.filter(user => user.id !== userId);
        setUnassignedUsers(filtered);
      }
    } catch (error) {
      console.error('Error fetching unassigned users:', error);
    }
  };

  const sortData = (data, field) => {
    return [...data].sort((a, b) => {
      if (field === 'priority') {
        const priorityOrder = { high: 1, medium: 2, low: 3 };
        return (priorityOrder[a.priority] || 4) - (priorityOrder[b.priority] || 4);
      } else if (field === 'deadline') {
        return new Date(a.deadline) - new Date(b.deadline);
      }
      return 0;
    });
  };

  const handleOpenDetails = async (assignment) => {
    setSelectedAssignment(assignment);
    setShowDetailsModal(true);
    setModifyForm({
      title: assignment.title,
      description: assignment.description || '',
      category: assignment.category || '',
      deadline: assignment.deadline ? assignment.deadline.split('T')[0] : '',
      priority: assignment.priority || 'medium'
    });
  };

  const handleDelete = async () => {
    if (!selectedAssignment) return;
    
    if (!window.confirm('Are you sure you want to delete this assignment? This will remove it from all assigned users.')) {
      return;
    }
    
    try {
      const response = await fetch(`http://localhost:3000/api/assignment-operations/${selectedAssignment.id}/cascade`, {
        method: 'DELETE'
      });
      
      const result = await response.json();
      
      if (result.success) {
        onError('Assignment deleted successfully', 'success');
        setShowDetailsModal(false);
        setSelectedAssignment(null);
        fetchAssignments();
      } else {
        throw new Error('Failed to delete assignment');
      }
    } catch (error) {
      console.error('Error deleting assignment:', error);
      onError('Failed to delete assignment', 'error');
    }
  };

  const handleModify = async () => {
    if (!modifyForm.title.trim() || !modifyForm.deadline) {
      onError('Title and deadline are required', 'warning');
      return;
    }
    
    try {
      const response = await fetch(`http://localhost:3000/api/assignments/${selectedAssignment.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: modifyForm.title.trim(),
          description: modifyForm.description.trim() || null,
          category: modifyForm.category.trim() || null,
          deadline: modifyForm.deadline,
          priority: modifyForm.priority
        })
      });
      
      const result = await response.json();
      
      if (result.success) {
        onError('Assignment updated successfully', 'success');
        setShowModifyDialog(false);
        setShowDetailsModal(false);
        setSelectedAssignment(null);
        fetchAssignments();
      } else {
        throw new Error('Failed to update assignment');
      }
    } catch (error) {
      console.error('Error updating assignment:', error);
      onError('Failed to update assignment', 'error');
    }
  };

  const handleOpenAssignedDialog = async () => {
    await fetchAssignedUsers(selectedAssignment.id);
    await fetchUnassignedUsers(selectedAssignment.id);
    setShowAssignedDialog(true);
  };

  const handleUpdateAssignments = async () => {
    try {
      // Remove selected users
      if (selectedUsersToRemove.length > 0) {
        await fetch(`http://localhost:3000/api/assignment-operations/${selectedAssignment.id}/remove-users`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ userIds: selectedUsersToRemove })
        });
      }
      
      // Add selected users
      if (selectedUsersToAdd.length > 0) {
        await fetch(`http://localhost:3000/api/assignment-operations/${selectedAssignment.id}/add-users`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ userIds: selectedUsersToAdd })
        });
      }
      
      onError('Assignment users updated successfully', 'success');
      setShowAssignedDialog(false);
      setSelectedUsersToRemove([]);
      setSelectedUsersToAdd([]);
      fetchAssignments();
    } catch (error) {
      console.error('Error updating assignments:', error);
      onError('Failed to update assignments', 'error');
    }
  };

  const handleCreateAssignment = async () => {
    if (!createForm.title.trim() || !createForm.deadline) {
      onError('Title and deadline are required', 'warning');
      return;
    }
    
    try {
      const response = await fetch('http://localhost:3000/api/assignments', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          organization_id: organizationId,
          department_id: departmentId,
          created_by: userId,
          title: createForm.title.trim(),
          description: createForm.description.trim() || null,
          category: createForm.category.trim() || null,
          deadline: createForm.deadline,
          priority: createForm.priority,
          status: 'pending',
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        })
      });
      
      const result = await response.json();
      
      if (result.success) {
        setNewAssignmentId(result.data.id);
        setCreateStep(2);
      } else {
        throw new Error('Failed to create assignment');
      }
    } catch (error) {
      console.error('Error creating assignment:', error);
      onError('Failed to create assignment', 'error');
    }
  };

  const handleFinalizeCreate = async () => {
    if (selectedNewUsers.length === 0) {
      onError('Please select at least one user', 'warning');
      return;
    }
    
    try {
      await fetch(`http://localhost:3000/api/assignment-operations/${newAssignmentId}/add-users`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userIds: selectedNewUsers })
      });
      
      onError('Assignment created and assigned successfully', 'success');
      setShowCreateModal(false);
      setCreateStep(1);
      setCreateForm({
        title: '',
        description: '',
        category: '',
        deadline: '',
        priority: 'medium'
      });
      setSelectedNewUsers([]);
      setNewAssignmentId(null);
      fetchAssignments();
    } catch (error) {
      console.error('Error finalizing create:', error);
      onError('Failed to assign users', 'error');
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
        <p>Loading assignments...</p>
      </div>
    );
  }

  const sortedAssignments = sortData(assignments, sortBy);

  return (
    <div className="assignment-admin-section">
      <div className="section-header">
        <h2>Assignment Administration</h2>
        <p className="section-subtitle">Create and manage department assignments</p>
      </div>

      {/* Add Button */}
      <button 
        className="floating-add-button"
        onClick={() => setShowCreateModal(true)}
        title="Create Assignment"
      >
        <img src={addIcon} alt="Add" />
      </button>

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

      {/* Assignments Table */}
      <div className="assignments-table-container">
        {sortedAssignments.length === 0 ? (
          <div className="no-assignments">No assignments found</div>
        ) : (
          <table className="assignments-table">
            <thead>
              <tr>
                <th>Created By</th>
                <th>Category</th>
                <th>Priority</th>
                <th>Deadline</th>
                <th>Employees Assigned</th>
                <th>Created At</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {sortedAssignments.map((assignment) => (
                <tr key={assignment.id}>
                  <td>{`${assignment.creator_first_name} ${assignment.creator_last_name}`}</td>
                  <td>{assignment.category || 'N/A'}</td>
                  <td>
                    <span className={`priority-badge ${getPriorityClass(assignment.priority)}`}>
                      {assignment.priority}
                    </span>
                  </td>
                  <td>{formatDate(assignment.deadline)}</td>
                  <td>{assignment.employee_count}</td>
                  <td>{formatDate(assignment.created_at)}</td>
                  <td>
                    <button 
                      className="view-button"
                      onClick={() => handleOpenDetails(assignment)}
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

      {/* Details Modal */}
      {showDetailsModal && selectedAssignment && !showModifyDialog && !showAssignedDialog && (
        <div className="modal-overlay" onClick={() => setShowDetailsModal(false)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <h2>Assignment Details</h2>
            
            <div className="detail-item">
              <label>Title:</label>
              <span>{selectedAssignment.title}</span>
            </div>
            
            <div className="detail-item">
              <label>Description:</label>
              <div className="description-box">{selectedAssignment.description || 'No description'}</div>
            </div>
            
            <div className="modal-actions">
              <button 
                className="delete-button"
                onClick={handleDelete}
              >
                Delete
              </button>
              <button 
                className="close-button"
                onClick={() => setShowDetailsModal(false)}
              >
                Close
              </button>
              <button 
                className="modify-button"
                onClick={() => setShowModifyDialog(true)}
              >
                Modify
              </button>
            </div>
            
            <button 
              className="assigned-button"
              onClick={handleOpenAssignedDialog}
            >
              Assigned To ({selectedAssignment.employee_count})
            </button>
          </div>
        </div>
      )}

      {/* Modify Dialog */}
      {showModifyDialog && (
        <div className="dialog-overlay">
          <div className="dialog-content slide-left">
            <h3>Modify Assignment</h3>
            
            <div className="form-group">
              <label>Title:</label>
              <input
                type="text"
                value={modifyForm.title}
                onChange={(e) => setModifyForm({...modifyForm, title: e.target.value})}
                placeholder="Assignment title"
              />
            </div>
            
            <div className="form-group">
              <label>Description:</label>
              <textarea
                value={modifyForm.description}
                onChange={(e) => setModifyForm({...modifyForm, description: e.target.value})}
                placeholder="Assignment description"
                rows="3"
              />
            </div>
            
            <div className="form-group">
              <label>Category:</label>
              <input
                type="text"
                value={modifyForm.category}
                onChange={(e) => setModifyForm({...modifyForm, category: e.target.value})}
                placeholder="e.g., Development, Design"
              />
            </div>
            
            <div className="form-group">
              <label>Deadline:</label>
              <input
                type="date"
                value={modifyForm.deadline}
                onChange={(e) => setModifyForm({...modifyForm, deadline: e.target.value})}
              />
            </div>
            
            <div className="form-group">
              <label>Priority:</label>
              <select
                value={modifyForm.priority}
                onChange={(e) => setModifyForm({...modifyForm, priority: e.target.value})}
              >
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
              </select>
            </div>
            
            <div className="dialog-actions">
              <button 
                className="cancel-button"
                onClick={() => setShowModifyDialog(false)}
              >
                Cancel
              </button>
              <button 
                className="confirm-button"
                onClick={handleModify}
              >
                Save Changes
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Assigned To Dialog */}
      {showAssignedDialog && (
        <div className="dialog-overlay">
          <div className="dialog-content slide-left large">
            <h3>Manage Assigned Users</h3>
            
            {/* Currently Assigned */}
            <div className="section-group">
              <h4>Currently Assigned ({assignedUsers.length})</h4>
              <div className="user-list">
                {assignedUsers.length === 0 ? (
                  <p className="no-users">No users assigned</p>
                ) : (
                  assignedUsers.map(user => (
                    <div key={user.id} className="user-item">
                      <input
                        type="checkbox"
                        checked={selectedUsersToRemove.includes(user.user_id)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedUsersToRemove([...selectedUsersToRemove, user.user_id]);
                          } else {
                            setSelectedUsersToRemove(selectedUsersToRemove.filter(id => id !== user.user_id));
                          }
                        }}
                      />
                      <span>{user.first_name} {user.last_name}</span>
                    </div>
                  ))
                )}
              </div>
            </div>
            
            {/* Available to Assign */}
            <div className="section-group">
              <h4>Available to Assign ({unassignedUsers.length})</h4>
              <div className="user-list">
                {unassignedUsers.length === 0 ? (
                  <p className="no-users">No available users</p>
                ) : (
                  unassignedUsers.map(user => (
                    <div key={user.id} className="user-item">
                      <input
                        type="checkbox"
                        checked={selectedUsersToAdd.includes(user.id)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedUsersToAdd([...selectedUsersToAdd, user.id]);
                          } else {
                            setSelectedUsersToAdd(selectedUsersToAdd.filter(id => id !== user.id));
                          }
                        }}
                      />
                      <span>{user.first_name} {user.last_name}</span>
                    </div>
                  ))
                )}
              </div>
            </div>
            
            <div className="dialog-actions">
              <button 
                className="cancel-button"
                onClick={() => {
                  setShowAssignedDialog(false);
                  setSelectedUsersToRemove([]);
                  setSelectedUsersToAdd([]);
                }}
              >
                Cancel
              </button>
              <button 
                className="confirm-button"
                onClick={handleUpdateAssignments}
              >
                Update
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Create Assignment Modal */}
      {showCreateModal && createStep === 1 && (
        <div className="modal-overlay" onClick={() => setShowCreateModal(false)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <h2>Create New Assignment</h2>
            
            <div className="form-group">
              <label>Title *</label>
              <input
                type="text"
                value={createForm.title}
                onChange={(e) => setCreateForm({...createForm, title: e.target.value})}
                placeholder="Assignment title"
              />
            </div>
            
            <div className="form-group">
              <label>Description</label>
              <textarea
                value={createForm.description}
                onChange={(e) => setCreateForm({...createForm, description: e.target.value})}
                placeholder="Assignment description"
                rows="4"
              />
            </div>
            
            <div className="form-group">
              <label>Category</label>
              <input
                type="text"
                value={createForm.category}
                onChange={(e) => setCreateForm({...createForm, category: e.target.value})}
                placeholder="e.g., Development, Design"
              />
            </div>
            
            <div className="form-group">
              <label>Deadline *</label>
              <input
                type="date"
                value={createForm.deadline}
                onChange={(e) => setCreateForm({...createForm, deadline: e.target.value})}
                min={new Date().toISOString().split('T')[0]}
              />
            </div>
            
            <div className="form-group">
              <label>Priority</label>
              <select
                value={createForm.priority}
                onChange={(e) => setCreateForm({...createForm, priority: e.target.value})}
              >
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
              </select>
            </div>
            
            <div className="modal-actions">
              <button 
                className="cancel-button"
                onClick={() => {
                  setShowCreateModal(false);
                  setCreateForm({
                    title: '',
                    description: '',
                    category: '',
                    deadline: '',
                    priority: 'medium'
                  });
                }}
              >
                Cancel
              </button>
              <button 
                className="confirm-button"
                onClick={handleCreateAssignment}
              >
                Next
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Create Assignment - Step 2: Assign Users */}
      {showCreateModal && createStep === 2 && (
        <div className="modal-overlay">
          <div className="modal-content large" onClick={e => e.stopPropagation()}>
            <h2>Assign Users to Assignment</h2>
            
            <div className="user-list">
              {allDepartmentUsers.length === 0 ? (
                <p className="no-users">No users available in department</p>
              ) : (
                allDepartmentUsers.map(user => (
                  <div key={user.id} className="user-item">
                    <input
                      type="checkbox"
                      checked={selectedNewUsers.includes(user.id)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedNewUsers([...selectedNewUsers, user.id]);
                        } else {
                          setSelectedNewUsers(selectedNewUsers.filter(id => id !== user.id));
                        }
                      }}
                    />
                    <span>{user.first_name} {user.last_name}</span>
                  </div>
                ))
              )}
            </div>
            
            <div className="modal-actions">
              <button 
                className="cancel-button"
                onClick={() => {
                  setShowCreateModal(false);
                  setCreateStep(1);
                  setCreateForm({
                    title: '',
                    description: '',
                    category: '',
                    deadline: '',
                    priority: 'medium'
                  });
                  setSelectedNewUsers([]);
                  setNewAssignmentId(null);
                }}
              >
                Cancel
              </button>
              <button 
                className="confirm-button"
                onClick={handleFinalizeCreate}
              >
                Create Assignment
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}