import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import warningIcon from '../../assets/icons/warning.png';
import '../../styles/EmployeeRegister.css';
import '../../styles/Notifications.css';

export default function EmployeeRegister() {
  const navigate = useNavigate();
  const location = useLocation();
  
  // Get orgId from URL if present
  const urlParams = new URLSearchParams(location.search);
  const initialOrgId = urlParams.get('orgId') || '';

  // Form fields
  const [organizationCode, setOrganizationCode] = useState(initialOrgId);
  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [selectedDepartment, setSelectedDepartment] = useState('');
  const [selectedRole, setSelectedRole] = useState('');

  // Organization search
  const [organizations, setOrganizations] = useState([]);
  const [filteredOrgs, setFilteredOrgs] = useState([]);
  const [showOrgDropdown, setShowOrgDropdown] = useState(false);
  const [selectedOrg, setSelectedOrg] = useState(null);

  // Departments and roles
  const [departments, setDepartments] = useState([]);
  const [roles, setRoles] = useState([]);

  // Error states
  const [errors, setErrors] = useState({});
  const [wiggleFields, setWiggleFields] = useState({});

  // Notification
  const [notification, setNotification] = useState({ show: false, message: '', type: '' });

  // Success popup
  const [showSuccessPopup, setShowSuccessPopup] = useState(false);

  // Load organizations on mount
  useEffect(() => {
    loadOrganizations();
  }, []);

  // If orgId is in URL, load that organization
  useEffect(() => {
    if (initialOrgId && organizations.length > 0) {
      const org = organizations.find(o => o.id === parseInt(initialOrgId));
      if (org) {
        setSelectedOrg(org);
        setOrganizationCode(org.unique_code);
        loadDepartments(org.id);
      }
    }
  }, [initialOrgId, organizations]);

  // Filter organizations as user types
  useEffect(() => {
    if (organizationCode && !selectedOrg) {
      const filtered = organizations.filter(org => 
        org.unique_code.toLowerCase().includes(organizationCode.toLowerCase()) ||
        org.name.toLowerCase().includes(organizationCode.toLowerCase())
      );
      setFilteredOrgs(filtered);
      setShowOrgDropdown(filtered.length > 0);
    } else {
      setShowOrgDropdown(false);
    }
  }, [organizationCode, organizations, selectedOrg]);

  // Load departments when organization changes
  useEffect(() => {
    if (selectedDepartment && selectedOrg) {
      loadRoles(selectedDepartment);
    } else {
      setRoles([]);
      setSelectedRole('');
    }
  }, [selectedDepartment]);

  // Auto-hide notification
  useEffect(() => {
    if (notification.show) {
      const timer = setTimeout(() => {
        setNotification({ show: false, message: '', type: '' });
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [notification.show]);

  const loadOrganizations = async () => {
    try {
      const response = await fetch('http://localhost:3000/api/organizations');
      const result = await response.json();
      if (result.success) {
        setOrganizations(result.data);
      }
    } catch (error) {
      console.error('Error loading organizations:', error);
      showNotification('Failed to load organizations. Please refresh the page.', 'error');
    }
  };

  const loadDepartments = async (orgId) => {
    try {
      const response = await fetch(`http://localhost:3000/api/departments?organizationId=${orgId}`);
      const result = await response.json();
      if (result.success) {
        setDepartments(result.data);
      }
    } catch (error) {
      console.error('Error loading departments:', error);
    }
  };

  const loadRoles = async (deptId) => {
    try {
      const response = await fetch(`http://localhost:3000/api/roles?departmentId=${deptId}`);
      const result = await response.json();
      if (result.success) {
        setRoles(result.data);
      }
    } catch (error) {
      console.error('Error loading roles:', error);
    }
  };

  const showNotification = (message, type = 'error') => {
    setNotification({ show: true, message, type });
  };

  const handleOrgSelect = (org) => {
    setSelectedOrg(org);
    setOrganizationCode(org.unique_code);
    setShowOrgDropdown(false);
    setErrors({ ...errors, organizationCode: false });
    setWiggleFields({ ...wiggleFields, organizationCode: false });
    loadDepartments(org.id);
    // Reset department and role when changing organization
    setSelectedDepartment('');
    setSelectedRole('');
    setDepartments([]);
    setRoles([]);
  };

  const handleOrgCodeChange = (e) => {
    const value = e.target.value;
    setOrganizationCode(value);
    setSelectedOrg(null); // Clear selected org when manually editing
    setErrors({ ...errors, organizationCode: false });
    setWiggleFields({ ...wiggleFields, organizationCode: false });
    // Clear departments when changing org
    setDepartments([]);
    setSelectedDepartment('');
    setRoles([]);
    setSelectedRole('');
  };

  const handleFieldChange = (field, value) => {
    switch(field) {
      case 'firstName':
        setFirstName(value);
        break;
      case 'lastName':
        setLastName(value);
        break;
      case 'email':
        setEmail(value);
        break;
      case 'password':
        setPassword(value);
        break;
      case 'confirmPassword':
        setConfirmPassword(value);
        break;
      case 'department':
        setSelectedDepartment(value);
        setSelectedRole(''); // Reset role when department changes
        break;
      case 'role':
        setSelectedRole(value);
        break;
    }
    setErrors({ ...errors, [field]: false });
    setWiggleFields({ ...wiggleFields, [field]: false });
  };

  const validateForm = () => {
    const newErrors = {};
    const newWiggles = {};

    // Required fields
    if (!organizationCode.trim()) {
      newErrors.organizationCode = true;
      newWiggles.organizationCode = true;
    }
    if (!firstName.trim()) {
      newErrors.firstName = true;
      newWiggles.firstName = true;
    }
    if (!lastName.trim()) {
      newErrors.lastName = true;
      newWiggles.lastName = true;
    }
    if (!email.trim()) {
      newErrors.email = true;
      newWiggles.email = true;
    }
    if (!password.trim()) {
      newErrors.password = true;
      newWiggles.password = true;
    }
    if (!confirmPassword.trim()) {
      newErrors.confirmPassword = true;
      newWiggles.confirmPassword = true;
    }

    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      setTimeout(() => setWiggleFields({}), 500);
      showNotification('Please fill in all required fields to continue.', 'error');
      return false;
    }

    return true;
  };

  const validateData = async () => {
    const newErrors = {};
    const newWiggles = {};

    // Validate first name starts with a letter
    if (!/^[A-Za-z]/.test(firstName.trim())) {
      newErrors.firstName = true;
      newWiggles.firstName = true;
      setFirstName('');
      setErrors(newErrors);
      setWiggleFields(newWiggles);
      setTimeout(() => setWiggleFields({}), 500);
      showNotification('First name must start with a letter.', 'error');
      return false;
    }

    // Validate last name starts with a letter
    if (!/^[A-Za-z]/.test(lastName.trim())) {
      newErrors.lastName = true;
      newWiggles.lastName = true;
      setLastName('');
      setErrors(newErrors);
      setWiggleFields(newWiggles);
      setTimeout(() => setWiggleFields({}), 500);
      showNotification('Last name must start with a letter.', 'error');
      return false;
    }

    // Verify organization exists
    const org = organizations.find(o => o.unique_code === organizationCode);
    if (!org) {
      newErrors.organizationCode = true;
      newWiggles.organizationCode = true;
      setOrganizationCode('');
      setErrors(newErrors);
      setWiggleFields(newWiggles);
      setTimeout(() => setWiggleFields({}), 500);
      showNotification('Invalid organization code. Please select a valid organization from the list.', 'error');
      return false;
    }

    // Validate first name starts with a letter
    if (!/^[a-zA-Z]/.test(firstName.trim())) {
      newErrors.firstName = true;
      newWiggles.firstName = true;
      setFirstName('');
      setErrors(newErrors);
      setWiggleFields(newWiggles);
      setTimeout(() => setWiggleFields({}), 500);
      showNotification('First name must start with a letter (A-Z).', 'error');
      return false;
    }

    // Validate last name starts with a letter
    if (!/^[a-zA-Z]/.test(lastName.trim())) {
      newErrors.lastName = true;
      newWiggles.lastName = true;
      setLastName('');
      setErrors(newErrors);
      setWiggleFields(newWiggles);
      setTimeout(() => setWiggleFields({}), 500);
      showNotification('Last name must start with a letter (A-Z).', 'error');
      return false;
    }

    // Email format validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      newErrors.email = true;
      newWiggles.email = true;
      setEmail('');
      setErrors(newErrors);
      setWiggleFields(newWiggles);
      setTimeout(() => setWiggleFields({}), 500);
      showNotification('Please enter a valid email address.', 'error');
      return false;
    }

    // Check if email already exists
    try {
      const response = await fetch('http://localhost:3000/api/auth/check-email', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email })
      });
      const result = await response.json();
      
      if (result.exists) {
        newErrors.email = true;
        newWiggles.email = true;
        setEmail('');
        setErrors(newErrors);
        setWiggleFields(newWiggles);
        setTimeout(() => setWiggleFields({}), 500);
        showNotification('This email is already registered. Please use a different email or login to your existing account.', 'error');
        return false;
      }
    } catch (error) {
      console.error('Error checking email:', error);
      showNotification('Error validating email. Please try again.', 'error');
      return false;
    }

    // Password match validation
    if (password !== confirmPassword) {
      newErrors.password = true;
      newErrors.confirmPassword = true;
      newWiggles.password = true;
      newWiggles.confirmPassword = true;
      setPassword('');
      setConfirmPassword('');
      setErrors(newErrors);
      setWiggleFields(newWiggles);
      setTimeout(() => setWiggleFields({}), 500);
      showNotification('Passwords do not match. Please re-enter your password.', 'error');
      return false;
    }

    // Password strength validation (at least 6 characters)
    if (password.length < 6) {
      newErrors.password = true;
      newErrors.confirmPassword = true;
      newWiggles.password = true;
      newWiggles.confirmPassword = true;
      setPassword('');
      setConfirmPassword('');
      setErrors(newErrors);
      setWiggleFields(newWiggles);
      setTimeout(() => setWiggleFields({}), 500);
      showNotification('Password must be at least 6 characters long.', 'error');
      return false;
    }

    return true;
  };

  const generateRandomColor = () => {
    const colors = [
      '#FF5733', '#3498DB', '#27AE60', '#9B59B6', '#E74C3C', 
      '#F39C12', '#16A085', '#8E44AD', '#C0392B', '#D68910',
      '#2ECC71', '#3498DB', '#667eea', '#764ba2', '#f093fb',
      '#4facfe', '#43e97b', '#fa709a', '#fee140', '#30cfd0'
    ];
    return colors[Math.floor(Math.random() * colors.length)];
  };

  const handleRegister = async (e) => {
    e.preventDefault();

    // Step 1: Validate all fields are filled
    if (!validateForm()) {
      return;
    }

    // Step 2: Validate data correctness
    if (!await validateData()) {
      return;
    }

    // Step 3: Create user and join request
    try {
      const org = organizations.find(o => o.unique_code === organizationCode);
      const avatarColor = generateRandomColor();

      // Create user
      const userResponse = await fetch('http://localhost:3000/api/users', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          first_name: firstName,
          last_name: lastName,
          email: email,
          password_hash: password, // In production, this should be hashed
          avatar_color: avatarColor,
          organization_id: org.id,
          department_id: null,
          role_id: null,
          is_approved: false,
          is_active: true,
          hire_date: null
        })
      });

      const userResult = await userResponse.json();

      if (!userResult.success) {
        showNotification('Failed to create account. Please try again.', 'error');
        return;
      }

      const userId = userResult.data.id;

      // Create join request
      const joinRequestResponse = await fetch('http://localhost:3000/api/join-requests', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          organization_id: org.id,
          requested_department_id: selectedDepartment || null,
          requested_role_id: selectedRole || null,
          status: 'pending',
          requested_at: new Date().toISOString(),
          reviewed_at: null,
          reviewed_by: null
        })
      });

      const joinRequestResult = await joinRequestResponse.json();

      if (!joinRequestResult.success) {
        showNotification('Account created but failed to submit join request. Please contact support.', 'error');
        return;
      }

      // Show success popup
      setShowSuccessPopup(true);

    } catch (error) {
      console.error('Registration error:', error);
      showNotification('An error occurred during registration. Please try again.', 'error');
    }
  };

  const handleSuccessOk = () => {
    navigate('/');
  };

  const handleSignInClick = () => {
    navigate('/login');
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleRegister(e);
    }
  };

  return (
    <div className="employee-register-container">
      {/* Notification */}
      {notification.show && (
        <div className={`notification ${notification.type}`}>
          {notification.message}
        </div>
      )}

      {/* Success Popup */}
      {showSuccessPopup && (
        <div className="success-popup-overlay">
          <div className="success-popup">
            <div className="success-icon">✓</div>
            <h2>Request Submitted Successfully!</h2>
            <p>
              Your join request has been sent and is awaiting review by an administrator.
              You will be redirected to the welcome page. Once your request is approved,
              you will be able to log in to your account.
            </p>
            <button className="popup-ok-button" onClick={handleSuccessOk}>
              OK
            </button>
          </div>
        </div>
      )}

      <div className="employee-register-content">
        <h1 className="employee-register-title">Employee Account Registration</h1>

        <div className="register-box">
          <form onSubmit={handleRegister}>
            {/* Organization Code */}
            <div className="form-group">
              <label htmlFor="organizationCode">
                Organization Code <span className="required">*</span>
              </label>
              <div className="org-search-wrapper">
                <div className={`input-wrapper ${errors.organizationCode ? 'error' : ''} ${wiggleFields.organizationCode ? 'wiggle' : ''}`}>
                  <input
                    type="text"
                    id="organizationCode"
                    value={organizationCode}
                    onChange={handleOrgCodeChange}
                    onFocus={() => setShowOrgDropdown(filteredOrgs.length > 0)}
                    placeholder="Enter organization code or name"
                    autoComplete="off"
                  />
                  {errors.organizationCode && (
                    <img src={warningIcon} alt="Error" className="warning-icon" />
                  )}
                </div>
                
                {/* Organization Dropdown */}
                {showOrgDropdown && (
                  <div className="org-dropdown">
                    {filteredOrgs.map(org => (
                      <div 
                        key={org.id}
                        className="org-dropdown-item"
                        onClick={() => handleOrgSelect(org)}
                      >
                        <span className="org-code">{org.unique_code}</span>
                        <span className="org-name">{org.name}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* First Name */}
            <div className="form-group">
              <label htmlFor="firstName">
                First Name <span className="required">*</span>
              </label>
              <div className={`input-wrapper ${errors.firstName ? 'error' : ''} ${wiggleFields.firstName ? 'wiggle' : ''}`}>
                <input
                  type="text"
                  id="firstName"
                  value={firstName}
                  onChange={(e) => handleFieldChange('firstName', e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Enter your first name"
                />
                {errors.firstName && (
                  <img src={warningIcon} alt="Error" className="warning-icon" />
                )}
              </div>
            </div>

            {/* Last Name */}
            <div className="form-group">
              <label htmlFor="lastName">
                Last Name <span className="required">*</span>
              </label>
              <div className={`input-wrapper ${errors.lastName ? 'error' : ''} ${wiggleFields.lastName ? 'wiggle' : ''}`}>
                <input
                  type="text"
                  id="lastName"
                  value={lastName}
                  onChange={(e) => handleFieldChange('lastName', e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Enter your last name"
                />
                {errors.lastName && (
                  <img src={warningIcon} alt="Error" className="warning-icon" />
                )}
              </div>
            </div>

            {/* Email */}
            <div className="form-group">
              <label htmlFor="email">
                Email <span className="required">*</span>
              </label>
              <div className={`input-wrapper ${errors.email ? 'error' : ''} ${wiggleFields.email ? 'wiggle' : ''}`}>
                <input
                  type="email"
                  id="email"
                  value={email}
                  onChange={(e) => handleFieldChange('email', e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Enter your email address"
                />
                {errors.email && (
                  <img src={warningIcon} alt="Error" className="warning-icon" />
                )}
              </div>
            </div>

            {/* Password */}
            <div className="form-group">
              <label htmlFor="password">
                Password <span className="required">*</span>
              </label>
              <div className={`input-wrapper ${errors.password ? 'error' : ''} ${wiggleFields.password ? 'wiggle' : ''}`}>
                <input
                  type="password"
                  id="password"
                  value={password}
                  onChange={(e) => handleFieldChange('password', e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Create a password (min 6 characters)"
                />
                {errors.password && (
                  <img src={warningIcon} alt="Error" className="warning-icon" />
                )}
              </div>
            </div>

            {/* Confirm Password */}
            <div className="form-group">
              <label htmlFor="confirmPassword">
                Confirm Password <span className="required">*</span>
              </label>
              <div className={`input-wrapper ${errors.confirmPassword ? 'error' : ''} ${wiggleFields.confirmPassword ? 'wiggle' : ''}`}>
                <input
                  type="password"
                  id="confirmPassword"
                  value={confirmPassword}
                  onChange={(e) => handleFieldChange('confirmPassword', e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Re-enter your password"
                />
                {errors.confirmPassword && (
                  <img src={warningIcon} alt="Error" className="warning-icon" />
                )}
              </div>
            </div>

            {/* Department (Optional) */}
            {departments.length > 0 && (
              <div className="form-group">
                <label htmlFor="department">
                  Preferred Department <span className="optional">(Optional)</span>
                </label>
                <div className="input-wrapper">
                  <select
                    id="department"
                    value={selectedDepartment}
                    onChange={(e) => handleFieldChange('department', e.target.value)}
                  >
                    <option value="">Select a department</option>
                    {departments.map(dept => (
                      <option key={dept.id} value={dept.id}>
                        {dept.name}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            )}

            {/* Role (Optional - only if department selected) */}
            {roles.length > 0 && (
              <div className="form-group">
                <label htmlFor="role">
                  Preferred Role <span className="optional">(Optional)</span>
                </label>
                <div className="input-wrapper">
                  <select
                    id="role"
                    value={selectedRole}
                    onChange={(e) => handleFieldChange('role', e.target.value)}
                  >
                    <option value="">Select a role</option>
                    {roles.map(role => (
                      <option key={role.id} value={role.id}>
                        {role.name}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            )}

            {/* Register Button */}
            <button type="submit" className="register-button">
              Register
            </button>

            {/* Sign In Link */}
            <div className="register-links">
              <button 
                type="button" 
                className="link-button sign-in-link"
                onClick={handleSignInClick}
              >
                Already have an account? Sign in
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}