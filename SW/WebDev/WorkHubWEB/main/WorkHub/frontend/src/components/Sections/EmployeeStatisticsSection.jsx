import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import EmployeeCard from '../Cards/EmployeeCard';
import '../../styles/StatisticsPage.css';

export default function EmployeeStatisticsSection({ organizationId, isCEO, canViewStatistics, onError }) {
  const [searchParams] = useSearchParams();
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [selectedEmployee, setSelectedEmployee] = useState(null);
  const [showResults, setShowResults] = useState(false);
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
      const response = await fetch(`http://localhost:3000/api/users/organization/${organizationId}`);
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

  const handleEmployeeSelect = (employee) => {
    setSelectedEmployee(employee);
    setSearchQuery('');
    setShowResults(false);
    setSearchResults([]);
  };

  const handleBackToSearch = () => {
    setSelectedEmployee(null);
    setSearchQuery('');
  };

  if (selectedEmployee) {
    return (
      <div className="employee-statistics-section">
        <button className="back-to-search-button" onClick={handleBackToSearch}>
          Back to Search
        </button>
        <EmployeeCard 
          employee={selectedEmployee}
          isCEO={isCEO}
          canViewStatistics={canViewStatistics}
          onError={onError}
        />
      </div>
    );
  }

  return (
    <div className="employee-statistics-section">
      <div className="section-header">
        <h2>Employee Statistics</h2>
        <p className="section-subtitle">Search and view detailed statistics for any employee</p>
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