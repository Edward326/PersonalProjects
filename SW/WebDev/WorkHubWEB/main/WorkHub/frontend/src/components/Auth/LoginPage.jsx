import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import warningIcon from '../../assets/icons/warning.png';
import employeeIcon from '../../assets/icons/employee_choose.png'
import organizationIcon from '../../assets/Icons/organization_choose.png';
import '../../styles/LoginPage.css';
import '../../styles/Notifications.css';

export default function LoginPage() {
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [uniqueCode, setUniqueCode] = useState('');
  const [emailError, setEmailError] = useState(false);
  const [passwordError, setPasswordError] = useState(false);
  const [uniqueCodeError, setUniqueCodeError] = useState(false);
  const [notification, setNotification] = useState({ show: false, message: '', type: '' });
  const [wiggleEmail, setWiggleEmail] = useState(false);
  const [wigglePassword, setWigglePassword] = useState(false);
  const [wiggleUniqueCode, setWiggleUniqueCode] = useState(false);
  const [showUniqueCodeField, setShowUniqueCodeField] = useState(false);
  const [fieldsLocked, setFieldsLocked] = useState(false);
  const [pendingOrgData, setPendingOrgData] = useState(null);
  const [showApprovalPopup, setShowApprovalPopup] = useState(false);

  // Check if already logged in
  useEffect(() => {
    const isAuthenticated = localStorage.getItem('isAuthenticated');
    if (isAuthenticated === 'true') {
      navigate('/');
    }
  }, [navigate]);

  useEffect(() => {
    if (notification.show) {
      const timer = setTimeout(() => {
        setNotification({ show: false, message: '', type: '' });
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [notification.show]);

  const showNotification = (message, type = 'error') => {
    setNotification({ show: true, message, type });
  };

  const resetErrors = () => {
    setEmailError(false);
    setPasswordError(false);
    setUniqueCodeError(false);
  };

  const handleEmailChange = (e) => {
    if (!fieldsLocked) {
      setEmail(e.target.value);
      setEmailError(false);
      setWiggleEmail(false);
    }
  };

  const handlePasswordChange = (e) => {
    if (!fieldsLocked) {
      setPassword(e.target.value);
      setPasswordError(false);
      setWigglePassword(false);
    }
  };

  const handleUniqueCodeChange = (e) => {
    setUniqueCode(e.target.value);
    setUniqueCodeError(false);
    setWiggleUniqueCode(false);
  };

  const resetToEmailPassword = () => {
    setShowUniqueCodeField(false);
    setFieldsLocked(false);
    setUniqueCode('');
    setUniqueCodeError(false);
    setPendingOrgData(null);
  };

  const handleForgotPassword = async () => {
    if (!email || email.trim() === '') {
      setEmailError(true);
      showNotification('Please provide an email address first to reset your password.');
      return;
    }

    try {
      const response = await fetch('http://localhost:3000/api/auth/check-email', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email })
      });

      const result = await response.json();

      if (!result.exists) {
        setEmailError(true);
        setWiggleEmail(true);
        setTimeout(() => setWiggleEmail(false), 500);
        showNotification('This email address is not registered in our system. Please check the email and try again, or register a new account.');
      } else {
        showNotification(`Password reset instructions have been sent to ${email}. Please check your inbox.`, 'success');
      }
    } catch (error) {
      console.error('Error checking email:', error);
      showNotification('An error occurred while processing your request. Please try again later.');
    }
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    resetErrors();

    // Handle unique code verification for CEO login
    if (showUniqueCodeField) {
      if (!uniqueCode || uniqueCode.trim() === '') {
        setUniqueCodeError(true);
        showNotification('Please enter the organization unique code to continue.');
        return;
      }

      if (uniqueCode !== pendingOrgData.unique_code) {
        setUniqueCodeError(true);
        setWiggleUniqueCode(true);
        setTimeout(() => setWiggleUniqueCode(false), 500);
        setUniqueCode('');
        showNotification('The organization code you entered is incorrect. Please try again or contact your organization administrator.');
        
        setTimeout(() => {
          resetToEmailPassword();
        }, 2000);
        return;
      }

      // CEO login successful - store minimal auth data (only ID and type)
      const authData = {
        userId: pendingOrgData.id,
        userType: 'ceo',
        organizationId: pendingOrgData.id
      };
      
      localStorage.setItem('authData', JSON.stringify(authData));
      localStorage.setItem('isAuthenticated', 'true');

      showNotification('Login successful! Welcome back.', 'success');
      setTimeout(() => {
        navigate('/');
      }, 1500);
      return;
    }

    // Validate email and password fields
    let hasError = false;
    if (!email || email.trim() === '') {
      setEmailError(true);
      hasError = true;
    }
    if (!password || password.trim() === '') {
      setPasswordError(true);
      hasError = true;
    }

    if (hasError) {
      showNotification('Please fill in all required fields to continue.');
      return;
    }

    // Check if CEO first
    try {
      const orgsResponse = await fetch('http://localhost:3000/api/organizations');
      const orgsResult = await orgsResponse.json();

      if (orgsResult.success) {
        const matchingOrg = orgsResult.data.find(org => org.ceo_email === email);

        if (matchingOrg) {
          // Verify password first
          if (password !== matchingOrg.ceo_password_hash) {
            setPasswordError(true);
            setWigglePassword(true);
            setTimeout(() => setWigglePassword(false), 500);
            setPassword('');
            showNotification('The password you entered is incorrect. Please try again.');
            return;
          }

          // Password correct, now ask for unique code
          setPendingOrgData(matchingOrg);
          setShowUniqueCodeField(true);
          setFieldsLocked(true);
          showNotification('CEO credentials verified. Please enter your organization unique code.', 'success');
          return;
        }
      }

      // Not a CEO, check if employee exists
      const emailCheckResponse = await fetch('http://localhost:3000/api/auth/check-email', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email })
      });

      const emailCheckResult = await emailCheckResponse.json();

      if (!emailCheckResult.exists) {
        setEmailError(true);
        setWiggleEmail(true);
        setTimeout(() => setWiggleEmail(false), 500);
        showNotification('No account found with this email address. Please check the email and try again, or register a new account if you don\'t have one.');
        return;
      }

      // Get employee data directly
      const employeeResponse = await fetch('http://localhost:3000/api/users');
      const employeeResult = await employeeResponse.json();

      if (employeeResult.success) {
        const matchingEmployee = employeeResult.data.find(user => user.email === email);

        if (matchingEmployee) {
          // Verify password
          if (password !== matchingEmployee.password_hash) {
            setPasswordError(true);
            setWigglePassword(true);
            setTimeout(() => setWigglePassword(false), 500);
            setPassword('');
            showNotification('The password you entered is incorrect. Please try again or use the "Forgot password" option.');
            return;
          }

          // Check if approved
          if (!matchingEmployee.is_approved) {
            setShowApprovalPopup(true);
            return;
          }

          await fetch(`http://localhost:3000/api/attendance/clock-in`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
              user_id: matchingEmployee.id,
              organization_id: matchingEmployee.organization_id,
              department_id: matchingEmployee.department_id
            })
          });

          // Employee login successful - store minimal auth data (only ID and type)
          const authData = {
            userId: matchingEmployee.id,
            userType: 'employee',
            departmentId: matchingEmployee.department_id,
            roleId: matchingEmployee.role_id,
            organizationId: matchingEmployee.organization_id
          };

          
          localStorage.setItem('authData', JSON.stringify(authData));
          localStorage.setItem('isAuthenticated', 'true');

          showNotification('Login successful! Redirecting to your dashboard...', 'success');
          setTimeout(() => {
            navigate('/');
          }, 1500);
          return;
        }
      }

      // Shouldn't reach here, but just in case
      showNotification('An error occurred during login. Please try again.');
      
    } catch (error) {
      console.error('Login error:', error);
      showNotification('An error occurred while trying to log in. Please check your internet connection and try again.');
    }
  };

  const handleRegisterClick = () => {
    navigate('/choose-register');
  };

  const handleCEORegister = () => {
    navigate('/organization-register');
  };

  const handleEmployeeRegister = () => {
    navigate('/employee-register');
  };

  const handleApprovalRetry = () => {
    setShowApprovalPopup(false);
    setEmail('');
    setPassword('');
    setEmailError(false);
    setPasswordError(false);
  };

  const handleApprovalGoHome = () => {
    setShowApprovalPopup(false);
    navigate('/');
  };

  return (
    <div className="login-page-container">
      {notification.show && (
        <div className={`notification ${notification.type}`}>
          {notification.message}
        </div>
      )}

      {/* Approval Pending Popup */}
      {showApprovalPopup && (
        <div className="approval-popup-overlay">
          <div className="approval-popup">
            <div className="approval-icon warning">⚠</div>
            <h2>Account Pending Approval</h2>
            <p>
              Your account is currently pending approval from your organization administrator. 
              You will be able to access the system once your request has been reviewed and approved.
            </p>
            <p className="approval-note">
              Please check back later or contact your administrator for more information.
            </p>
            <div className="approval-buttons">
              <button className="approval-retry-button" onClick={handleApprovalRetry}>
                Try Another Account
              </button>
              <button className="approval-home-button" onClick={handleApprovalGoHome}>
                Go to Home
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="login-content">
        <h1 className="login-page-title">Log In to WorkHub</h1>

        <div className="login-box-wrapper">
          {showUniqueCodeField && (
            <div className="unique-code-field-container">
              <div className="form-group">
                <label htmlFor="uniqueCode">Organization Code</label>
                <div className={`input-wrapper ${uniqueCodeError ? 'error' : ''} ${wiggleUniqueCode ? 'wiggle' : ''}`}>
                  <input
                    type="password"
                    id="uniqueCode"
                    value={uniqueCode}
                    onChange={handleUniqueCodeChange}
                    onKeyPress={(e) => {
                      if (e.key === 'Enter') {
                        handleLogin(e);
                      }
                    }}
                    placeholder="Enter unique code"
                    autoFocus
                  />
                  {uniqueCodeError && (
                    <img src={warningIcon} alt="Error" className="warning-icon" />
                  )}
                </div>
              </div>
            </div>
          )}

          <div className="login-box">
            <form onSubmit={handleLogin}>
              <div className="form-group">
                <label htmlFor="email">Email</label>
                <div className={`input-wrapper ${emailError ? 'error' : ''} ${wiggleEmail ? 'wiggle' : ''} ${fieldsLocked ? 'locked' : ''}`}>
                  <input
                    type="email"
                    id="email"
                    value={email}
                    onChange={handleEmailChange}
                    placeholder="Enter your email"
                    disabled={fieldsLocked}
                  />
                  {emailError && (
                    <img src={warningIcon} alt="Error" className="warning-icon" />
                  )}
                </div>
              </div>

              <div className="form-group">
                <label htmlFor="password">Password</label>
                <div className={`input-wrapper ${passwordError ? 'error' : ''} ${wigglePassword ? 'wiggle' : ''} ${fieldsLocked ? 'locked' : ''}`}>
                  <input
                    type="password"
                    id="password"
                    value={password}
                    onChange={handlePasswordChange}
                    placeholder="Enter your password"
                    disabled={fieldsLocked}
                  />
                  {passwordError && (
                    <img src={warningIcon} alt="Error" className="warning-icon" />
                  )}
                </div>
              </div>

              <button type="submit" className="login-button">
                {showUniqueCodeField ? 'Verify Code & Login' : 'Log In'}
              </button>

              {!showUniqueCodeField && (
                <div className="login-links">
                  <button 
                    type="button" 
                    className="link-button forgot-password"
                    onClick={handleForgotPassword}
                  >
                    Forgot password?
                  </button>
                  <button 
                    type="button" 
                    className="link-button register-link"
                    onClick={handleRegisterClick}
                  >
                    Register an account
                  </button>
                </div>
              )}

              {showUniqueCodeField && (
                <div className="login-links">
                  <button 
                    type="button" 
                    className="link-button back-to-login"
                    onClick={resetToEmailPassword}
                  >
                    Back to login
                  </button>
                </div>
              )}
            </form>
          </div>
        </div>

        <div className="bottom-actions">
          <button className="action-button ceo-button" onClick={handleCEORegister}>
            <img src={organizationIcon} alt="CEO" className="button-icon-img" />
            <span className="button-text">
              <strong>You are a CEO?</strong>
              <span>Register your organization today</span>
            </span>
          </button>

          <button className="action-button employee-button" onClick={handleEmployeeRegister}>
            <img src={employeeIcon} alt="CEO" className="button-icon-img" />
            <span className="button-text">
              <strong>You are an employee?</strong>
              <span>Join your organization today</span>
            </span>
          </button>
        </div>
      </div>
    </div>
  );
}