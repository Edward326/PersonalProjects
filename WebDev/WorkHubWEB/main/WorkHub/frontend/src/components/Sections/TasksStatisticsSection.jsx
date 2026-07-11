import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import '../../styles/StatisticsPage.css';

export default function TasksStatisticsSection({ organizationId, departmentId, isCEO, canViewStatistics, canAssign, onError }) {
  const [searchParams] = useSearchParams();
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [selectedEmployee, setSelectedEmployee] = useState(null);
  const [showResults, setShowResults] = useState(false);
  const [activeTab, setActiveTab] = useState('present');
  const [pastAssignments, setPastAssignments] = useState([]);
  const [presentAssignments, setPresentAssignments] = useState([]);
  const [showModal, setShowModal] = useState(false);
  const [selectedAssignment, setSelectedAssignment] = useState(null);
  const urlUserId = searchParams.get('userId');

  useEffect(() => {
    // If userId in URL, fetch that employee directly
    if (urlUserId) {
      fetchEmployeeById(urlUserId);
    }
  }, [urlUserId]);

  useEffect(() => {
    if (searchQuery.trim().length > 0) {
      searchEmployees();
    } else {
      setSearchResults([]);
      setShowResults(false);
    }
  }, [searchQuery]);

  useEffect(() => {
    if (selectedEmployee) {
      fetchEmployeeAssignments();
    }
  }, [selectedEmployee]);

  const fetchEmployeeById = async (userId) => {
    try {
      const response = await fetch(`http://localhost:3000/api/users/${userId}`);
      const result = await response.json();
      
      if (result.success) {
        setSelectedEmployee(result.data);
      } else {
        onError('Employee not found', 'error');
      }
    } catch (error) {
      console.error('Error fetching employee:', error);
      onError('Failed to load employee data', 'error');
    }
  };

  const searchEmployees = async () => {
    try {
      let url;
      // If can only assign (not CEO or view statistics), search only in department
      if (canAssign && !isCEO && !canViewStatistics) {
        url = `http://localhost:3000/api/users/department/${departmentId}`;
      } else {
        // CEO or can view statistics - search all departments
        url = `http://localhost:3000/api/users/organization/${organizationId}`;
      }

      const response = await fetch(url);
      const result = await response.json();
      
      if (result.success) {
        const filtered = result.data.filter(user => {
          const fullName = `${user.first_name} ${user.last_name}`.toLowerCase();
          return fullName.includes(searchQuery.toLowerCase());
        });
        setSearchResults(filtered);
        setShowResults(true);
      }
    } catch (error) {
      console.error('Error searching employees:', error);
      onError('Failed to search employees', 'error');
    }
  };

  const fetchEmployeeAssignments = async () => {
    try {
      // Fetch past assignments (from assignment_submissions)
      const pastResponse = await fetch(`http://localhost:3000/api/assignment-submissions/user/${selectedEmployee.id}`);
      const pastResult = await pastResponse.json();
      
      if (pastResult.success) {
        // Fetch full assignment details for each submission
        const pastWithDetails = await Promise.all(
          pastResult.data.map(async (submission) => {
            const assignmentResponse = await fetch(`http://localhost:3000/api/assignments/${submission.assignment_id}`);
            const assignmentResult = await assignmentResponse.json();
            
            if (assignmentResult.success) {
              const assignment = assignmentResult.data;
              
              // Fetch creator details
              const creatorResponse = await fetch(`http://localhost:3000/api/users/${assignment.created_by}`);
              const creatorResult = await creatorResponse.json();
              
              return {
                ...submission,
                assignment,
                creator: creatorResult.success ? creatorResult.data : null
              };
            }
            return null;
          })
        );
        
        setPastAssignments(pastWithDetails.filter(item => item !== null));
      }

      // Fetch present assignments (from assignment_users)
      const presentResponse = await fetch(`http://localhost:3000/api/assignment-users/user/${selectedEmployee.id}`);
      const presentResult = await presentResponse.json();
      
      if (presentResult.success) {
        // Fetch full assignment details
        const presentWithDetails = await Promise.all(
          presentResult.data.map(async (assignmentUser) => {
            const assignmentResponse = await fetch(`http://localhost:3000/api/assignments/${assignmentUser.assignment_id}`);
            const assignmentResult = await assignmentResponse.json();
            
            if (assignmentResult.success) {
              const assignment = assignmentResult.data;
              
              // Fetch creator details
              const creatorResponse = await fetch(`http://localhost:3000/api/users/${assignment.created_by}`);
              const creatorResult = await creatorResponse.json();
              
              return {
                ...assignmentUser,
                assignment,
                creator: creatorResult.success ? creatorResult.data : null
              };
            }
            return null;
          })
        );
        
        setPresentAssignments(presentWithDetails.filter(item => item !== null));
      }

    } catch (error) {
      console.error('Error fetching assignments:', error);
      onError('Failed to load assignments', 'error');
    }
  };

  const handleEmployeeSelect = (employee) => {
    setSelectedEmployee(employee);
    setSearchQuery('');
    setShowResults(false);
    setSearchResults([]);
  };

  const handleBackToSearch = () => {
    setSelectedEmployee(null);
    setPastAssignments([]);
    setPresentAssignments([]);
    setSearchQuery('');
  };

  const handleViewAssignment = (assignmentData, isPast = false) => {
    setSelectedAssignment({ ...assignmentData, isPast });
    setShowModal(true);
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

  if (!selectedEmployee) {
    return (
      <div className="tasks-statistics-section">
        <div className="section-header">
          <h2>Tasks Statistics</h2>
          <p className="section-subtitle">Search and view assignment history for any employee</p>
        </div>

        <div className="employee-search-container">
          <div className="search-box">
            <input
              type="text"
              className="search-input"
              placeholder="Search employee by name..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onFocus={() => setShowResults(true)}
            />
            
            {showResults && searchResults.length > 0 && (
              <div className="search-results-dropdown">
                {searchResults.map(employee => (
                  <div
                    key={employee.id}
                    className="search-result-item"
                    onClick={() => handleEmployeeSelect(employee)}
                  >
                    <div className="result-avatar">
                      {employee.first_name.charAt(0)}{employee.last_name.charAt(0)}
                    </div>
                    <div className="result-info">
                      <div className="result-name">{employee.first_name} {employee.last_name}</div>
                      <div className="result-email">{employee.email}</div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {showResults && searchQuery.trim().length > 0 && searchResults.length === 0 && (
              <div className="search-results-dropdown">
                <div className="no-results">No employees found</div>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="tasks-statistics-section">
      <button className="back-to-search-button" onClick={handleBackToSearch}>
      Back to Search
      </button>

      <div className="employee-name-header">
        <h3>{selectedEmployee.first_name} {selectedEmployee.last_name}'s Assignments</h3>
      </div>

      {/* Tabs */}
      <div className="assignments-tabs">
        <button 
          className={`tab-button ${activeTab === 'present' ? 'active' : ''}`}
          onClick={() => setActiveTab('present')}
        >
          Present Assignments
        </button>
        <button 
          className={`tab-button ${activeTab === 'past' ? 'active' : ''}`}
          onClick={() => setActiveTab('past')}
        >
          Past Assignments
        </button>
      </div>

      {/* Present Assignments Table */}
      {activeTab === 'present' && (
        <div className="assignments-table-container">
          {presentAssignments.length === 0 ? (
            <div className="no-assignments">No present assignments</div>
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
                {presentAssignments.map((item, index) => (
                  <tr key={index}>
                    <td>
                      {item.creator ? `${item.creator.first_name} ${item.creator.last_name}` : 'N/A'}
                    </td>
                    <td>{item.assignment.category || 'N/A'}</td>
                    <td>
                      <span className={`priority-badge ${getPriorityClass(item.assignment.priority)}`}>
                        {item.assignment.priority}
                      </span>
                    </td>
                    <td>{formatDate(item.assignment.deadline)}</td>
                    <td>{formatDate(item.assignment.created_at)}</td>
                    <td>
                      <button 
                        className="view-assignment-button"
                        onClick={() => handleViewAssignment(item, false)}
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
      )}

      {/* Past Assignments Table */}
      {activeTab === 'past' && (
        <div className="assignments-table-container">
          {pastAssignments.length === 0 ? (
            <div className="no-assignments">No past assignments</div>
          ) : (
            <table className="assignments-table">
              <thead>
                <tr>
                  <th>Created By</th>
                  <th>Category</th>
                  <th>Priority</th>
                  <th>Deadline</th>
                  <th>Created At</th>
                  <th>Submitted At</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {pastAssignments.map((item, index) => (
                  <tr key={index}>
                    <td>
                      {item.creator ? `${item.creator.first_name} ${item.creator.last_name}` : 'N/A'}
                    </td>
                    <td>{item.assignment.category || 'N/A'}</td>
                    <td>
                      <span className={`priority-badge ${getPriorityClass(item.assignment.priority)}`}>
                        {item.assignment.priority}
                      </span>
                    </td>
                    <td>{formatDate(item.assignment.deadline)}</td>
                    <td>{formatDate(item.assignment.created_at)}</td>
                    <td>{formatDateTime(item.submitted_at)}</td>
                    <td>
                      <button 
                        className="view-assignment-button"
                        onClick={() => handleViewAssignment(item, true)}
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
      )}

      {/* Assignment Details Modal */}
      {showModal && selectedAssignment && (
        <div className="assignment-modal-overlay" onClick={() => setShowModal(false)}>
          <div className="assignment-modal" onClick={e => e.stopPropagation()}>
            <h2>Assignment Details</h2>
            
            <div className="modal-section">
              <div className="assignment-detail-item">
                <label>Title:</label>
                <span>{selectedAssignment.assignment.title}</span>
              </div>
              
              <div className="assignment-detail-item full-width">
                <label>Description:</label>
                <div className="description-box">
                  {selectedAssignment.assignment.description || 'No description provided'}
                </div>
              </div>

              {selectedAssignment.isPast && selectedAssignment.notes && (
                <div className="assignment-detail-item full-width">
                  <label>Submission Notes:</label>
                  <div className="description-box">
                    {selectedAssignment.notes}
                  </div>
                </div>
              )}
            </div>

            <button className="close-assignment-modal-button" onClick={() => setShowModal(false)}>
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}