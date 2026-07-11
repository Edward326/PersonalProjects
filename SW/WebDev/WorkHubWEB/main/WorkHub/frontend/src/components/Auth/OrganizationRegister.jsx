import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import warningIcon from '../../assets/icons/warning.png';
import '../../styles/OrganizationRegister.css';
import '../../styles/Notifications.css';

export default function OrganizationRegister() {
  const navigate = useNavigate();
  
  // Step management
  const [currentStep, setCurrentStep] = useState(1); // 1: Org Info, 2: Departments, 3: Roles
  
  // Organization fields
  const [uniqueId, setUniqueId] = useState('');
  const [orgName, setOrgName] = useState('');
  const [logoUrl, setLogoUrl] = useState('');
  const [industry, setIndustry] = useState('');
  const [description, setDescription] = useState('');
  const [foundingDate, setFoundingDate] = useState('');
  const [website, setWebsite] = useState('');
  const [linkedin, setLinkedin] = useState('');
  const [twitter, setTwitter] = useState('');
  const [contactEmail, setContactEmail] = useState('');
  const [ceoEmail, setCeoEmail] = useState('');
  const [ceoPassword, setCeoPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');

  // Departments array
  const [departments, setDepartments] = useState([]);
  const [showDeptForm, setShowDeptForm] = useState(false);
  const [currentDept, setCurrentDept] = useState({ name: '', description: '' });

  // Roles array
  const [roles, setRoles] = useState([]);
  const [showRoleForm, setShowRoleForm] = useState(false);
  const [currentRole, setCurrentRole] = useState({
    departmentIndex: 0,
    name: '',
    priority: 1,
    can_post_news: false,
    can_post_event: false,
    can_assign_tasks: false,
    can_receive_tasks: false,
    can_view_statistics: false,
    can_hire: false,
    can_reassign_tasks: false
  });

  // Error states
  const [errors, setErrors] = useState({});
  const [wiggleFields, setWiggleFields] = useState({});

  // Notification
  const [notification, setNotification] = useState({ show: false, message: '', type: '' });

  // Success popup
  const [showSuccessPopup, setShowSuccessPopup] = useState(false);
  const [generatedCode, setGeneratedCode] = useState('');
  const [registeredOrgId, setRegisteredOrgId] = useState(null);

  // Auto-hide notification
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

  const handleFieldChange = (field, value) => {
    const setters = {
      uniqueId: setUniqueId,
      orgName: setOrgName,
      logoUrl: setLogoUrl,
      industry: setIndustry,
      description: setDescription,
      foundingDate: setFoundingDate,
      website: setWebsite,
      linkedin: setLinkedin,
      twitter: setTwitter,
      contactEmail: setContactEmail,
      ceoEmail: setCeoEmail,
      ceoPassword: setCeoPassword,
      confirmPassword: setConfirmPassword
    };

    if (setters[field]) {
      setters[field](value);
    }
    
    setErrors({ ...errors, [field]: false });
    setWiggleFields({ ...wiggleFields, [field]: false });
  };

  const validateOrgInfo = async () => {
    const newErrors = {};
    const newWiggles = {};
    const emptyFields = [];

    // Check all required fields first
    if (!uniqueId.trim()) {
      newErrors.uniqueId = true;
      newWiggles.uniqueId = true;
      emptyFields.push('Unique Organization ID');
    }

    if (!orgName.trim()) {
      newErrors.orgName = true;
      newWiggles.orgName = true;
      emptyFields.push('Organization Name');
    }

    if (!industry.trim()) {
      newErrors.industry = true;
      newWiggles.industry = true;
      emptyFields.push('Industry');
    }

    if (!description.trim()) {
      newErrors.description = true;
      newWiggles.description = true;
      emptyFields.push('Description');
    }

    if (!foundingDate.trim()) {
      newErrors.foundingDate = true;
      newWiggles.foundingDate = true;
      emptyFields.push('Founding Date');
    }

    if (!website.trim()) {
      newErrors.website = true;
      newWiggles.website = true;
      emptyFields.push('Website');
    }

    if (!contactEmail.trim()) {
      newErrors.contactEmail = true;
      newWiggles.contactEmail = true;
      emptyFields.push('Contact Email');
    }

    if (!ceoEmail.trim()) {
      newErrors.ceoEmail = true;
      newWiggles.ceoEmail = true;
      emptyFields.push('CEO Email');
    }

    if (!ceoPassword.trim()) {
      newErrors.ceoPassword = true;
      newWiggles.ceoPassword = true;
      emptyFields.push('Password');
    }

    if (!confirmPassword.trim()) {
      newErrors.confirmPassword = true;
      newWiggles.confirmPassword = true;
      emptyFields.push('Confirm Password');
    }

    // If there are empty required fields, show error and stop
    if (emptyFields.length > 0) {
      setErrors(newErrors);
      setWiggleFields(newWiggles);
      setTimeout(() => setWiggleFields({}), 500);
      
      if (emptyFields.length === 1) {
        showNotification(`Please complete the following field: ${emptyFields[0]}`, 'error');
      } else if (emptyFields.length === 2) {
        showNotification(`Please complete the following fields: ${emptyFields.join(' and ')}`, 'error');
      } else {
        showNotification(`Please complete the following fields: ${emptyFields.slice(0, -1).join(', ')}, and ${emptyFields[emptyFields.length - 1]}`, 'error');
      }
      return false;
    }

    // Unique ID validation: ORG-[A-Z]+-[A-Z0-9]+
    const uniqueIdRegex = /^ORG-[A-Z]+-[A-Z0-9]+$/;
    if (!uniqueIdRegex.test(uniqueId.trim())) {
      newErrors.uniqueId = true;
      newWiggles.uniqueId = true;
      setUniqueId('');
      setErrors(newErrors);
      setWiggleFields(newWiggles);
      setTimeout(() => setWiggleFields({}), 500);
      showNotification('Unique ID must be in format: ORG-LETTERS-ALPHANUMERIC (e.g., ORG-TECH-A7K4M)', 'error');
      return false;
    }

    // Email format validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(contactEmail.trim())) {
      newErrors.contactEmail = true;
      newWiggles.contactEmail = true;
      setContactEmail('');
      setErrors(newErrors);
      setWiggleFields(newWiggles);
      setTimeout(() => setWiggleFields({}), 500);
      showNotification('Please enter a valid contact email address.', 'error');
      return false;
    }

    if (!emailRegex.test(ceoEmail.trim())) {
      newErrors.ceoEmail = true;
      newWiggles.ceoEmail = true;
      setCeoEmail('');
      setErrors(newErrors);
      setWiggleFields(newWiggles);
      setTimeout(() => setWiggleFields({}), 500);
      showNotification('Please enter a valid CEO email address.', 'error');
      return false;
    }

    // Check if CEO email already exists
    try {
      const response = await fetch('http://localhost:3000/api/auth/check-email', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: ceoEmail })
      });
      const result = await response.json();
      
      if (result.exists) {
        newErrors.ceoEmail = true;
        newWiggles.ceoEmail = true;
        setCeoEmail('');
        setErrors(newErrors);
        setWiggleFields(newWiggles);
        setTimeout(() => setWiggleFields({}), 500);
        showNotification('This email is already registered. Please use a different email.', 'error');
        return false;
      }
    } catch (error) {
      console.error('Error checking email:', error);
      showNotification('Error validating email. Please try again.', 'error');
      return false;
    }

    // Password validation
    if (ceoPassword !== confirmPassword) {
      newErrors.ceoPassword = true;
      newErrors.confirmPassword = true;
      newWiggles.ceoPassword = true;
      newWiggles.confirmPassword = true;
      setCeoPassword('');
      setConfirmPassword('');
      setErrors(newErrors);
      setWiggleFields(newWiggles);
      setTimeout(() => setWiggleFields({}), 500);
      showNotification('Passwords do not match. Please re-enter your password.', 'error');
      return false;
    }

    if (ceoPassword.length < 6) {
      newErrors.ceoPassword = true;
      newErrors.confirmPassword = true;
      newWiggles.ceoPassword = true;
      newWiggles.confirmPassword = true;
      setCeoPassword('');
      setConfirmPassword('');
      setErrors(newErrors);
      setWiggleFields(newWiggles);
      setTimeout(() => setWiggleFields({}), 500);
      showNotification('Password must be at least 6 characters long.', 'error');
      return false;
    }

    // Optional URL validations
    if (logoUrl && logoUrl.trim()) {
      try {
        new URL(logoUrl);
      } catch {
        newErrors.logoUrl = true;
        newWiggles.logoUrl = true;
        setLogoUrl('');
        setErrors(newErrors);
        setWiggleFields(newWiggles);
        setTimeout(() => setWiggleFields({}), 500);
        showNotification('Please enter a valid logo URL.', 'error');
        return false;
      }
    }

    if (linkedin && linkedin.trim()) {
      try {
        new URL(linkedin);
      } catch {
        newErrors.linkedin = true;
        newWiggles.linkedin = true;
        setLinkedin('');
        setErrors(newErrors);
        setWiggleFields(newWiggles);
        setTimeout(() => setWiggleFields({}), 500);
        showNotification('Please enter a valid LinkedIn URL.', 'error');
        return false;
      }
    }

    if (twitter && twitter.trim()) {
      try {
        new URL(twitter);
      } catch {
        newErrors.twitter = true;
        newWiggles.twitter = true;
        setTwitter('');
        setErrors(newErrors);
        setWiggleFields(newWiggles);
        setTimeout(() => setWiggleFields({}), 500);
        showNotification('Please enter a valid Twitter URL.', 'error');
        return false;
      }
    }

    return true;
  };

  const handleNextFromOrg = async (e) => {
    e.preventDefault();
    
    if (!await validateOrgInfo()) {
      return;
    }

    // Move to departments step
    setCurrentStep(2);
    showNotification('Organization info saved! Now set up your departments.', 'success');
  };

  const handleAddDepartment = () => {
    if (!currentDept.name.trim()) {
      showNotification('Department name is required.', 'error');
      return;
    }

    const newDept = {
      name: currentDept.name.trim(),
      description: currentDept.description.trim(),
      has_header: true,
      header_background_color: '#2ecc71',
      header_display_type: 'both',
      header_position: 'center',
      logo_size: 'medium',
      sidebar_position: 'left',
      layout_color_hover: '#3498db',
      layout_color_clicked: '#2980b9',
      layout_color_selected: '#667eea'
    };

    setDepartments([...departments, newDept]);
    setCurrentDept({ name: '', description: '' });
    setShowDeptForm(false);
    showNotification(`Department "${newDept.name}" added successfully!`, 'success');
  };

  const handleNextFromDepartments = () => {
    if (departments.length === 0) {
      // No departments, register organization only
      handleFinalRegister();
    } else {
      // Has departments, move to roles step
      setCurrentStep(3);
      showNotification('Departments saved! Now define roles for your departments.', 'success');
    }
  };

  const handleAddRole = () => {
    if (!currentRole.name.trim()) {
      showNotification('Role name is required.', 'error');
      return;
    }

    if (currentRole.priority < 1) {
      showNotification('Priority must be at least 1.', 'error');
      return;
    }

    const selectedDept = departments[currentRole.departmentIndex];
    
    const newRole = {
      departmentName: selectedDept.name,
      name: currentRole.name.trim(),
      priority: currentRole.priority,
      can_post_news: currentRole.can_post_news,
      can_post_event: currentRole.can_post_event,
      can_assign_tasks: currentRole.can_assign_tasks,
      can_receive_tasks: currentRole.can_receive_tasks,
      can_view_statistics: currentRole.can_view_statistics,
      can_hire: currentRole.can_hire,
      can_reassign_tasks: currentRole.can_reassign_tasks
    };

    setRoles([...roles, newRole]);
    setCurrentRole({
      departmentIndex: 0,
      name: '',
      priority: 1,
      can_post_news: false,
      can_post_event: false,
      can_assign_tasks: false,
      can_receive_tasks: false,
      can_view_statistics: false,
      can_hire: false,
      can_reassign_tasks: false
    });
    setShowRoleForm(false);
    showNotification(`Role "${newRole.name}" added to ${selectedDept.name}!`, 'success');
  };

  const generateUniqueCode = () => {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
    let code = '';
    for (let i = 0; i < 5; i++) {
      code += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return code;
  };

  const handleFinalRegister = async () => {
    try {
      const uniqueCode = generateUniqueCode();

      console.log('Starting registration...', { 
        uniqueId, 
        orgName, 
        departments: departments.length,
        roles: roles.length 
      });

      // Create organization
      const orgResponse = await fetch('http://localhost:3000/api/organizations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          unique_id: uniqueId.trim(),
          unique_code: uniqueCode,
          name: orgName.trim(),
          logo_url: logoUrl.trim() || null,
          industry: industry.trim(),
          description: description.trim(),
          founding_date: foundingDate.trim(),
          website: website.trim(),
          linkedin_url: linkedin.trim() || null,
          twitter_url: twitter.trim() || null,
          contact_email: contactEmail.trim(),
          ceo_email: ceoEmail.trim(),
          ceo_password_hash: ceoPassword,
          employees_count: 0,
          departments_count: departments.length,
          is_active: true,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        })
      });

      const orgResult = await orgResponse.json();
      console.log('Organization response:', orgResult);

      if (!orgResult.success) {
        showNotification(orgResult.message || 'Failed to create organization. Please try again.', 'error');
        return;
      }

      const orgId = orgResult.data.id;
      console.log('Organization created with ID:', orgId);

      // Store org data for after popup
      setOrgName(orgName);  // Keep the name
      
      // Create departments if any
      if (departments.length > 0) {
        console.log('Creating departments...');
        const deptPromises = departments.map(dept =>
          fetch('http://localhost:3000/api/departments', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              organization_id: orgId,
              name: dept.name,
              description: dept.description,
              has_header: dept.has_header,
              header_background_color: dept.header_background_color,
              header_display_type: dept.header_display_type,
              header_position: dept.header_position,
              logo_size: dept.logo_size,
              sidebar_position: dept.sidebar_position,
              layout_color_hover: dept.layout_color_hover,
              layout_color_clicked: dept.layout_color_clicked,
              layout_color_selected: dept.layout_color_selected,
              created_at: new Date().toISOString(),
              updated_at: new Date().toISOString()
            })
          }).then(res => res.json())
        );

        const deptResults = await Promise.all(deptPromises);
        console.log('Department results:', deptResults);

        // Check if any department creation failed
        const failedDepts = deptResults.filter(result => !result.success);
        if (failedDepts.length > 0) {
          console.error('Some departments failed to create:', failedDepts);
          showNotification('Some departments failed to create. Please check the data.', 'error');
          return;
        }

        // Create roles if any
        if (roles.length > 0) {
          console.log('Creating roles...');
          // Map department names to IDs
          const deptMap = {};
          deptResults.forEach((result, index) => {
            if (result.success) {
              deptMap[departments[index].name] = result.data.id;
            }
          });

          console.log('Department mapping:', deptMap);

          const rolePromises = roles.map(role =>
            fetch('http://localhost:3000/api/roles', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                organization_id: orgId,
                department_id: deptMap[role.departmentName],
                name: role.name,
                priority: role.priority,
                permissions: {},  // Default empty JSONB object
                can_accept_requests: false,  // Default false (not used but required by schema)
                can_post_news: role.can_post_news,
                can_post_event: role.can_post_event,
                can_assign_tasks: role.can_assign_tasks,
                can_receive_tasks: role.can_receive_tasks,
                can_view_statistics: role.can_view_statistics,
                can_hire: role.can_hire,
                can_reassign_tasks: role.can_reassign_tasks,
                can_modify_department_look: false,  // Default false (not used but required by schema)
                created_at: new Date().toISOString()
              })
            }).then(res => res.json())
          );

          const roleResults = await Promise.all(rolePromises);
          console.log('Role results:', roleResults);

          // Check if any role creation failed
          const failedRoles = roleResults.filter(result => !result.success);
          if (failedRoles.length > 0) {
            console.error('Some roles failed to create:', failedRoles);
            showNotification('Some roles failed to create. Please check the data.', 'error');
            return;
          }
        }
      }

      // Show success popup with unique code
      console.log('Registration successful! Showing popup...');
      setGeneratedCode(uniqueCode);
      setRegisteredOrgId(orgId);
      setShowSuccessPopup(true);

    } catch (error) {
      console.error('Registration error:', error);
      showNotification('An error occurred during registration. Please try again.', 'error');
    }
  };

  const handleSuccessOk = () => {
    // Store authentication data in localStorage
    const authData = {
      userId: registeredOrgId,
      userType: 'ceo',
    };
    
    localStorage.setItem('authData', JSON.stringify(authData));
    localStorage.setItem('isAuthenticated', 'true');
    
    // Navigate to home page
    navigate('/');
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      if (currentStep === 1) {
        handleNextFromOrg(e);
      }
    }
  };

  return (
    <div className="org-register-container">
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
            <h2>Organization Created Successfully!</h2>
            <p>Your organization has been created. Please save your unique code:</p>
            <div className="unique-code-display">{generatedCode}</div>
            <p className="unique-code-note">
              This code is important for your employees to join your organization. 
              Please write it down or save it securely.
            </p>
            <button className="popup-ok-button" onClick={handleSuccessOk}>
              OK
            </button>
          </div>
        </div>
      )}

      <div className="org-register-content">
        <h1 className="org-register-title">
          {currentStep === 1 && 'Organization Account Registration'}
          {currentStep === 2 && 'Department Manager'}
          {currentStep === 3 && 'Role Manager'}
        </h1>

        {/* Step 1: Organization Info */}
        {currentStep === 1 && (
          <div className="register-box">
            <form onSubmit={handleNextFromOrg}>
              {/* Unique ID */}
              <div className="form-group">
                <label htmlFor="uniqueId">
                  Unique Organization ID <span className="required">*</span>
                  <span className="field-hint">Format: ORG-LETTERS-ALPHANUMERIC</span>
                </label>
                <div className={`input-wrapper ${errors.uniqueId ? 'error' : ''} ${wiggleFields.uniqueId ? 'wiggle' : ''}`}>
                  <input
                    type="text"
                    id="uniqueId"
                    value={uniqueId}
                    onChange={(e) => handleFieldChange('uniqueId', e.target.value.toUpperCase())}
                    onKeyPress={handleKeyPress}
                    placeholder="e.g., ORG-TECH-A7K4M"
                    style={{ textTransform: 'uppercase' }}
                  />
                  {errors.uniqueId && (
                    <img src={warningIcon} alt="Error" className="warning-icon" />
                  )}
                </div>
              </div>

              {/* Organization Name */}
              <div className="form-group">
                <label htmlFor="orgName">
                  Organization Name <span className="required">*</span>
                </label>
                <div className={`input-wrapper ${errors.orgName ? 'error' : ''} ${wiggleFields.orgName ? 'wiggle' : ''}`}>
                  <input
                    type="text"
                    id="orgName"
                    value={orgName}
                    onChange={(e) => handleFieldChange('orgName', e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Enter organization name"
                  />
                  {errors.orgName && (
                    <img src={warningIcon} alt="Error" className="warning-icon" />
                  )}
                </div>
              </div>

              {/* Logo URL */}
              <div className="form-group">
                <label htmlFor="logoUrl">
                  Logo URL <span className="optional">(Optional)</span>
                </label>
                <div className={`input-wrapper ${errors.logoUrl ? 'error' : ''} ${wiggleFields.logoUrl ? 'wiggle' : ''}`}>
                  <input
                    type="url"
                    id="logoUrl"
                    value={logoUrl}
                    onChange={(e) => handleFieldChange('logoUrl', e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="https://example.com/logo.png"
                  />
                  {errors.logoUrl && (
                    <img src={warningIcon} alt="Error" className="warning-icon" />
                  )}
                </div>
              </div>

              {/* Industry */}
              <div className="form-group">
                <label htmlFor="industry">
                  Industry <span className="required">*</span>
                </label>
                <div className={`input-wrapper ${errors.industry ? 'error' : ''} ${wiggleFields.industry ? 'wiggle' : ''}`}>
                  <input
                    type="text"
                    id="industry"
                    value={industry}
                    onChange={(e) => handleFieldChange('industry', e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="e.g., Technology, Healthcare, Finance"
                  />
                  {errors.industry && (
                    <img src={warningIcon} alt="Error" className="warning-icon" />
                  )}
                </div>
              </div>

              {/* Description */}
              <div className="form-group">
                <label htmlFor="description">
                  Description <span className="required">*</span>
                </label>
                <div className={`input-wrapper ${errors.description ? 'error' : ''} ${wiggleFields.description ? 'wiggle' : ''}`}>
                  <textarea
                    id="description"
                    value={description}
                    onChange={(e) => handleFieldChange('description', e.target.value)}
                    placeholder="Brief description of your organization"
                    rows="3"
                  />
                  {errors.description && (
                    <img src={warningIcon} alt="Error" className="warning-icon" />
                  )}
                </div>
              </div>

              {/* Founding Date */}
              <div className="form-group">
                <label htmlFor="foundingDate">
                  Founding Date <span className="required">*</span>
                </label>
                <div className={`input-wrapper ${errors.foundingDate ? 'error' : ''} ${wiggleFields.foundingDate ? 'wiggle' : ''}`}>
                  <input
                    type="date"
                    id="foundingDate"
                    value={foundingDate}
                    onChange={(e) => handleFieldChange('foundingDate', e.target.value)}
                    onKeyPress={handleKeyPress}
                    max={new Date().toISOString().split('T')[0]}
                  />
                  {errors.foundingDate && (
                    <img src={warningIcon} alt="Error" className="warning-icon" />
                  )}
                </div>
              </div>

              {/* Website */}
              <div className="form-group">
                <label htmlFor="website">
                  Website <span className="required">*</span>
                </label>
                <div className={`input-wrapper ${errors.website ? 'error' : ''} ${wiggleFields.website ? 'wiggle' : ''}`}>
                  <input
                    type="url"
                    id="website"
                    value={website}
                    onChange={(e) => handleFieldChange('website', e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="https://www.example.com"
                  />
                  {errors.website && (
                    <img src={warningIcon} alt="Error" className="warning-icon" />
                  )}
                </div>
              </div>

              {/* LinkedIn */}
              <div className="form-group">
                <label htmlFor="linkedin">
                  LinkedIn <span className="optional">(Optional)</span>
                </label>
                <div className={`input-wrapper ${errors.linkedin ? 'error' : ''} ${wiggleFields.linkedin ? 'wiggle' : ''}`}>
                  <input
                    type="url"
                    id="linkedin"
                    value={linkedin}
                    onChange={(e) => handleFieldChange('linkedin', e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="https://linkedin.com/company/..."
                  />
                  {errors.linkedin && (
                    <img src={warningIcon} alt="Error" className="warning-icon" />
                  )}
                </div>
              </div>

              {/* Twitter */}
              <div className="form-group">
                <label htmlFor="twitter">
                  Twitter <span className="optional">(Optional)</span>
                </label>
                <div className={`input-wrapper ${errors.twitter ? 'error' : ''} ${wiggleFields.twitter ? 'wiggle' : ''}`}>
                  <input
                    type="url"
                    id="twitter"
                    value={twitter}
                    onChange={(e) => handleFieldChange('twitter', e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="https://twitter.com/..."
                  />
                  {errors.twitter && (
                    <img src={warningIcon} alt="Error" className="warning-icon" />
                  )}
                </div>
              </div>

              {/* Contact Email */}
              <div className="form-group">
                <label htmlFor="contactEmail">
                  Contact Email <span className="required">*</span>
                </label>
                <div className={`input-wrapper ${errors.contactEmail ? 'error' : ''} ${wiggleFields.contactEmail ? 'wiggle' : ''}`}>
                  <input
                    type="email"
                    id="contactEmail"
                    value={contactEmail}
                    onChange={(e) => handleFieldChange('contactEmail', e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="contact@example.com"
                  />
                  {errors.contactEmail && (
                    <img src={warningIcon} alt="Error" className="warning-icon" />
                  )}
                </div>
              </div>

              {/* CEO Email */}
              <div className="form-group">
                <label htmlFor="ceoEmail">
                  CEO Email <span className="required">*</span>
                </label>
                <div className={`input-wrapper ${errors.ceoEmail ? 'error' : ''} ${wiggleFields.ceoEmail ? 'wiggle' : ''}`}>
                  <input
                    type="email"
                    id="ceoEmail"
                    value={ceoEmail}
                    onChange={(e) => handleFieldChange('ceoEmail', e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="ceo@example.com"
                  />
                  {errors.ceoEmail && (
                    <img src={warningIcon} alt="Error" className="warning-icon" />
                  )}
                </div>
              </div>

              {/* CEO Password */}
              <div className="form-group">
                <label htmlFor="ceoPassword">
                  Password <span className="required">*</span>
                </label>
                <div className={`input-wrapper ${errors.ceoPassword ? 'error' : ''} ${wiggleFields.ceoPassword ? 'wiggle' : ''}`}>
                  <input
                    type="password"
                    id="ceoPassword"
                    value={ceoPassword}
                    onChange={(e) => handleFieldChange('ceoPassword', e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Create a password (min 6 characters)"
                  />
                  {errors.ceoPassword && (
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

              {/* Next Button */}
              <button type="submit" className="register-button">
                Next
              </button>
            </form>
          </div>
        )}

        {/* Step 2: Departments */}
        {currentStep === 2 && (
          <div className="register-box">
            <p className="step-subtitle">Add departments to your organization (optional)</p>
            
            {/* List of departments */}
            {departments.length > 0 && (
              <div className="items-list">
                {departments.map((dept, index) => (
                  <div key={index} className="list-item">
                    <div className="item-info">
                      <strong>{dept.name}</strong>
                      {dept.description && <span className="item-desc">{dept.description}</span>}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Add department form */}
            {showDeptForm && (
              <div className="add-form">
                <div className="form-group">
                  <label>Department Name *</label>
                  <input
                    type="text"
                    value={currentDept.name}
                    onChange={(e) => setCurrentDept({ ...currentDept, name: e.target.value })}
                    placeholder="e.g., Engineering, Marketing"
                  />
                </div>
                <div className="form-group">
                  <label>Description</label>
                  <textarea
                    value={currentDept.description}
                    onChange={(e) => setCurrentDept({ ...currentDept, description: e.target.value })}
                    placeholder="Brief description"
                    rows="2"
                  />
                </div>
                <div className="form-buttons">
                  <button className="done-button" onClick={handleAddDepartment}>Done</button>
                  <button className="cancel-button" onClick={() => setShowDeptForm(false)}>Cancel</button>
                </div>
              </div>
            )}

            {/* Add button */}
            {!showDeptForm && (
              <button className="add-button" onClick={() => setShowDeptForm(true)}>
                + Add Department
              </button>
            )}

            {/* Single Next button */}
            <button className="register-button" onClick={handleNextFromDepartments}>
              {departments.length > 0 ? 'Next' : 'Register Organization'}
            </button>
          </div>
        )}

        {/* Step 3: Roles */}
        {currentStep === 3 && (
          <div className="register-box">
            <p className="step-subtitle">Define roles for your departments</p>
            
            {/* List of roles */}
            {roles.length > 0 && (
              <div className="items-list">
                {roles.map((role, index) => (
                  <div key={index} className="list-item">
                    <div className="item-info">
                      <strong>{role.name}</strong>
                      <span className="item-desc">
                        {role.departmentName} • Priority: {role.priority}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Add role form */}
            {showRoleForm && (
              <div className="add-form">
                <div className="form-group">
                  <label>Department *</label>
                  <select
                    value={currentRole.departmentIndex}
                    onChange={(e) => setCurrentRole({ ...currentRole, departmentIndex: parseInt(e.target.value) })}
                  >
                    {departments.map((dept, index) => (
                      <option key={index} value={index}>{dept.name}</option>
                    ))}
                  </select>
                </div>
                <div className="form-group">
                  <label>Role Name *</label>
                  <input
                    type="text"
                    value={currentRole.name}
                    onChange={(e) => setCurrentRole({ ...currentRole, name: e.target.value })}
                    placeholder="e.g., Manager, Developer"
                  />
                </div>
                <div className="form-group">
                  <label>Priority *</label>
                  <input
                    type="number"
                    min="1"
                    value={currentRole.priority}
                    onChange={(e) => setCurrentRole({ ...currentRole, priority: parseInt(e.target.value) })}
                  />
                </div>
                <div className="form-group">
                  <label>Permissions</label>
                  <div className="permissions-grid">
                    <label className="permission-item">
                      <input
                        type="checkbox"
                        checked={currentRole.can_post_news}
                        onChange={(e) => setCurrentRole({ ...currentRole, can_post_news: e.target.checked })}
                      />
                      <span>Can Post News</span>
                    </label>
                    <label className="permission-item">
                      <input
                        type="checkbox"
                        checked={currentRole.can_post_event}
                        onChange={(e) => setCurrentRole({ ...currentRole, can_post_event: e.target.checked })}
                      />
                      <span>Can Post Event</span>
                    </label>
                    <label className="permission-item">
                      <input
                        type="checkbox"
                        checked={currentRole.can_assign_tasks}
                        onChange={(e) => setCurrentRole({ ...currentRole, can_assign_tasks: e.target.checked })}
                      />
                      <span>Can Assign Tasks</span>
                    </label>
                    <label className="permission-item">
                      <input
                        type="checkbox"
                        checked={currentRole.can_receive_tasks}
                        onChange={(e) => setCurrentRole({ ...currentRole, can_receive_tasks: e.target.checked })}
                      />
                      <span>Can Receive Tasks</span>
                    </label>
                    <label className="permission-item">
                      <input
                        type="checkbox"
                        checked={currentRole.can_view_statistics}
                        onChange={(e) => setCurrentRole({ ...currentRole, can_view_statistics: e.target.checked })}
                      />
                      <span>Can View Statistics</span>
                    </label>
                    <label className="permission-item">
                      <input
                        type="checkbox"
                        checked={currentRole.can_hire}
                        onChange={(e) => setCurrentRole({ ...currentRole, can_hire: e.target.checked })}
                      />
                      <span>Can Hire</span>
                    </label>
                    <label className="permission-item">
                      <input
                        type="checkbox"
                        checked={currentRole.can_reassign_tasks}
                        onChange={(e) => setCurrentRole({ ...currentRole, can_reassign_tasks: e.target.checked })}
                      />
                      <span>Can Reassign Tasks</span>
                    </label>
                  </div>
                </div>
                <div className="form-buttons">
                  <button className="done-button" onClick={handleAddRole}>Done</button>
                  <button className="cancel-button" onClick={() => setShowRoleForm(false)}>Cancel</button>
                </div>
              </div>
            )}

            {/* Add button */}
            {!showRoleForm && (
              <button className="add-button" onClick={() => setShowRoleForm(true)}>
                + Add Role
              </button>
            )}

            {/* Register button */}
            <button className="register-button" onClick={handleFinalRegister}>
              Register Organization
            </button>
          </div>
        )}
      </div>
    </div>
  );
}