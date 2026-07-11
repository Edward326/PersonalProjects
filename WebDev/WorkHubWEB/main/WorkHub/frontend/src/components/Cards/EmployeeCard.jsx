import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import '../../styles/StatisticsPage.css';

export default function EmployeeCard({ employee, isCEO, canViewStatistics, onError }) {
  const navigate = useNavigate();
  const [departmentData, setDepartmentData] = useState(null);
  const [roleData, setRoleData] = useState(null);
  const [completedAssignments, setCompletedAssignments] = useState(0);
  const [pendingAssignments, setPendingAssignments] = useState(0);
  const [relocationRequests, setRelocationRequests] = useState(0);
  const [attendanceRecords, setAttendanceRecords] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchEmployeeData();
  }, [employee.id]);

  const fetchEmployeeData = async () => {
    try {
      setLoading(true);

      // Fetch department data
      if (employee.department_id) {
        const deptResponse = await fetch(`http://localhost:3000/api/departments/${employee.department_id}`);
        const deptResult = await deptResponse.json();
        if (deptResult.success) {
          setDepartmentData(deptResult.data);
        }
      }

      // Fetch role data
      if (employee.role_id) {
        const roleResponse = await fetch(`http://localhost:3000/api/roles/${employee.role_id}`);
        const roleResult = await roleResponse.json();
        if (roleResult.success) {
          setRoleData(roleResult.data);
        }
      }

      // Fetch completed assignments
      const completedResponse = await fetch(`http://localhost:3000/api/assignment-submissions/user/${employee.id}`);
      const completedResult = await completedResponse.json();
      if (completedResult.success) {
        setCompletedAssignments(completedResult.data.length);
      }

      // Fetch pending assignments
      const pendingResponse = await fetch(`http://localhost:3000/api/assignment-users/user/${employee.id}`);
      const pendingResult = await pendingResponse.json();
      if (pendingResult.success) {
        setPendingAssignments(pendingResult.data.length);
      }

      // Fetch relocation requests
      const relocationResponse = await fetch(`http://localhost:3000/api/assignment-relocations/user/${employee.id}`);
      const relocationResult = await relocationResponse.json();
      if (relocationResult.success) {
        setRelocationRequests(relocationResult.data.length);
      }

      // Fetch attendance records (is_complete=true)
      const attendanceResponse = await fetch(`http://localhost:3000/api/attendance/user/${employee.id}?isComplete=true`);
      const attendanceResult = await attendanceResponse.json();
      if (attendanceResult.success) {
        setAttendanceRecords(attendanceResult.data);
      }

    } catch (error) {
      console.error('Error fetching employee data:', error);
      onError('Failed to load employee data', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleViewTasks = () => {
    navigate(`/statistics?userId=${employee.id}`, { state: { section: 'tasks' } });
    window.location.reload(); // Force reload to trigger tasks section
  };

  const formatDate = (date) => {
    if (!date) return 'N/A';
    return new Date(date).toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'long', 
      day: 'numeric' 
    });
  };

  const calculateHoursWorked = (checkIn, checkOut) => {
    if (!checkIn || !checkOut) return 0;
    const checkInTime = new Date(checkIn);
    const checkOutTime = new Date(checkOut);
    const diffMs = checkOutTime - checkInTime;
    const hours = diffMs / (1000 * 60 * 60);
    return hours.toFixed(2);
  };

  const renderAttendanceCalendar = () => {
    if (attendanceRecords.length === 0) {
      return <div className="no-attendance">No attendance records found</div>;
    }

    // Group attendance by date
    const attendanceByDate = {};
    attendanceRecords.forEach(record => {
      const date = new Date(record.date).toISOString().split('T')[0];
      const hours = calculateHoursWorked(record.check_in_time, record.check_out_time);
      
      if (!attendanceByDate[date]) {
        attendanceByDate[date] = 0;
      }
      attendanceByDate[date] += parseFloat(hours);
    });

    // Get current month and year
    const today = new Date();
    const currentMonth = today.getMonth();
    const currentYear = today.getFullYear();

    // Get first day of month and number of days
    const firstDay = new Date(currentYear, currentMonth, 1);
    const lastDay = new Date(currentYear, currentMonth + 1, 0);
    const daysInMonth = lastDay.getDate();
    const startingDayOfWeek = firstDay.getDay();

    const monthNames = ['January', 'February', 'March', 'April', 'May', 'June',
      'July', 'August', 'September', 'October', 'November', 'December'];

    const calendar = [];
    let week = [];

    // Add empty cells for days before month starts
    for (let i = 0; i < startingDayOfWeek; i++) {
      week.push(<div key={`empty-${i}`} className="calendar-day empty"></div>);
    }

    // Add days of month
    for (let day = 1; day <= daysInMonth; day++) {
      const dateStr = `${currentYear}-${String(currentMonth + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
      const hours = attendanceByDate[dateStr];

      week.push(
        <div key={day} className={`calendar-day ${hours ? 'has-attendance' : ''}`}>
          <div className="day-number">{day}</div>
          {hours && <div className="day-hours">{hours.toFixed(1)}h</div>}
        </div>
      );

      if (week.length === 7) {
        calendar.push(<div key={`week-${calendar.length}`} className="calendar-week">{week}</div>);
        week = [];
      }
    }

    // Add remaining week
    if (week.length > 0) {
      while (week.length < 7) {
        week.push(<div key={`empty-end-${week.length}`} className="calendar-day empty"></div>);
      }
      calendar.push(<div key={`week-${calendar.length}`} className="calendar-week">{week}</div>);
    }

    return (
      <div className="attendance-calendar">
        <h4>{monthNames[currentMonth]} {currentYear}</h4>
        <div className="calendar-header">
          <div className="calendar-day-label">Sun</div>
          <div className="calendar-day-label">Mon</div>
          <div className="calendar-day-label">Tue</div>
          <div className="calendar-day-label">Wed</div>
          <div className="calendar-day-label">Thu</div>
          <div className="calendar-day-label">Fri</div>
          <div className="calendar-day-label">Sat</div>
        </div>
        {calendar}
      </div>
    );
  };

  if (loading) {
    return (
      <div className="section-loading">
        <div className="loading-spinner"></div>
        <p>Loading employee data...</p>
      </div>
    );
  }

  return (
    <div className="employee-card">
      {/* Status Indicator */}
      <div className={`status-indicator ${employee.is_active ? 'active' : 'inactive'}`}>
        <span className="status-dot"></span>
        <span className="status-text">{employee.is_active ? 'Active' : 'Inactive'}</span>
      </div>

      {/* Employee Info */}
      <div className="employee-info-section">
        <div className="employee-avatar-large">
          {employee.first_name.charAt(0)}{employee.last_name.charAt(0)}
        </div>
        <h3>{employee.first_name} {employee.last_name}</h3>
        <p className="employee-email">{employee.email}</p>
      </div>

      {/* Details Grid */}
      <div className="employee-details-grid">
        <div className="detail-item">
          <label>Department:</label>
          <span>{departmentData?.name || 'N/A'}</span>
        </div>
        <div className="detail-item">
          <label>Role:</label>
          <span>{roleData?.name || 'N/A'}</span>
        </div>
        <div className="detail-item">
          <label>Hire Date:</label>
          <span>{formatDate(employee.hire_date)}</span>
        </div>
      </div>

      {/* Permissions */}
      {roleData && (
        <div className="permissions-section">
          <h4>Role Permissions</h4>
          <div className="permissions-grid-card">
            <div className={`permission-badge ${roleData.can_assign_tasks ? 'enabled' : 'disabled'}`}>
              Assign Tasks
            </div>
            <div className={`permission-badge ${roleData.can_reassign_tasks ? 'enabled' : 'disabled'}`}>
              Reassign Tasks
            </div>
            <div className={`permission-badge ${roleData.can_hire ? 'enabled' : 'disabled'}`}>
              Hire
            </div>
            <div className={`permission-badge ${roleData.can_view_statistics ? 'enabled' : 'disabled'}`}>
              View Statistics
            </div>
            <div className={`permission-badge ${roleData.can_accept_join_requests ? 'enabled' : 'disabled'}`}>
              Accept Requests
            </div>
            <div className={`permission-badge ${roleData.can_post_news ? 'enabled' : 'disabled'}`}>
              Post News
            </div>
            <div className={`permission-badge ${roleData.can_post_events ? 'enabled' : 'disabled'}`}>
              Post Events
            </div>
            <div className={`permission-badge ${roleData.receive_tasks ? 'enabled' : 'disabled'}`}>
              Receive Tasks
            </div>
          </div>
        </div>
      )}

      {/* Assignment Stats */}
      <div className="assignment-stats-section">
        <h4>Assignment Statistics</h4>
        <div className="stats-row">
          <div className="stat-box completed">
            <div className="stat-value">{completedAssignments}</div>
            <div className="stat-label">Completed</div>
          </div>
          <div className="stat-box pending">
            <div className="stat-value">{pendingAssignments}</div>
            <div className="stat-label">Pending</div>
          </div>
          <div className="stat-box relocation">
            <div className="stat-value">{relocationRequests}</div>
            <div className="stat-label">Relocation Requests</div>
          </div>
        </div>
      </div>

      {/* Attendance Calendar */}
      <div className="attendance-section">
        <h4>Attendance Calendar</h4>
        {renderAttendanceCalendar()}
      </div>
    </div>
  );
}