import React, { useState } from 'react';
import '../../styles/SettingsPage.css';

export default function AccountSection({ userData, isCEO, organizationData, onError, onUpdate }) {
  const [showChangeDialog, setShowChangeDialog] = useState(false);
  const [currentField, setCurrentField] = useState('');
  const [newValue, setNewValue] = useState('');
  const [password, setPassword] = useState('');

  const handleChangeClick = (field, currentValue) => {
    setCurrentField(field);
    setNewValue('');
    setPassword('');
    setShowChangeDialog(true);
  };

  const handleConfirmChange = async () => {
    if (!password.trim()) {
      onError('Password is required to make changes', 'error');
      return;
    }

    if (!newValue.trim()) {
      onError('New value cannot be empty', 'error');
      return;
    }

    try {
      if (isCEO) {
        // Verify CEO password
        const verifyResponse = await fetch('http://localhost:3000/api/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            email: organizationData.ceo_email,
            password: password,
            uniqueCode: organizationData.unique_code
          })
        });

        const verifyResult = await verifyResponse.json();

        if (!verifyResult.success) {
          onError('Incorrect password', 'error');
          return;
        }

        // Update organization field
        let updateData = { updated_at: new Date().toISOString() };
        
        if (currentField === 'email') {
          updateData.ceo_email = newValue;
        } else if (currentField === 'password') {
          updateData.ceo_password_hash = newValue;
        }

        const updateResponse = await fetch(`http://localhost:3000/api/organizations/${organizationData.id}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(updateData)
        });

        const updateResult = await updateResponse.json();

        if (updateResult.success) {
          onError(`${currentField.charAt(0).toUpperCase() + currentField.slice(1)} updated successfully!`, 'success');
          setShowChangeDialog(false);
          onUpdate(); // Refresh data
        } else {
          onError('Failed to update field', 'error');
        }

      } else {
        // Verify employee password
        const verifyResponse = await fetch('http://localhost:3000/api/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            email: userData.email,
            password: password
          })
        });

        const verifyResult = await verifyResponse.json();

        if (!verifyResult.success) {
          onError('Incorrect password', 'error');
          return;
        }

        // Update user field
        let updateData = { updated_at: new Date().toISOString() };
        
        if (currentField === 'first_name') {
          if (!/^[a-zA-Z]/.test(newValue.trim())) {
            onError('First name must start with a letter', 'error');
            return;
          }
          updateData.first_name = newValue;
        } else if (currentField === 'last_name') {
          if (!/^[a-zA-Z]/.test(newValue.trim())) {
            onError('Last name must start with a letter', 'error');
            return;
          }
          updateData.last_name = newValue;
        } else if (currentField === 'email') {
          const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
          if (!emailRegex.test(newValue)) {
            onError('Invalid email format', 'error');
            return;
          }
          updateData.email = newValue;
        } else if (currentField === 'password') {
          updateData.password_hash = newValue;
        }

        const updateResponse = await fetch(`http://localhost:3000/api/users/${userData.id}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(updateData)
        });

        const updateResult = await updateResponse.json();

        if (updateResult.success) {
          onError(`${currentField.replace('_', ' ').charAt(0).toUpperCase() + currentField.replace('_', ' ').slice(1)} updated successfully!`, 'success');
          setShowChangeDialog(false);
          onUpdate(); // Refresh data
        } else {
          onError(updateResult.message || 'Failed to update field', 'error');
        }
      }

    } catch (error) {
      console.error('Error updating field:', error);
      onError('Failed to update field', 'error');
    }
  };

  const getFieldLabel = (field) => {
    const labels = {
      'first_name': 'First Name',
      'last_name': 'Last Name',
      'email': 'Email',
      'password': 'Password'
    };
    return labels[field] || field;
  };

  return (
    <div className="account-section">
      <div className="section-header">
        <h2>Account Settings</h2>
        <p className="section-subtitle">Manage your personal account information</p>
      </div>

      <div className="account-fields-container">
        {isCEO ? (
          // CEO Fields
          <>
            <div className="account-field">
              <div className="field-info">
                <label>Email</label>
                <span className="field-value">{organizationData.ceo_email}</span>
              </div>
              <button 
                className="change-link"
                onClick={() => handleChangeClick('email', organizationData.ceo_email)}
              >
                Change
              </button>
            </div>

            <div className="account-field">
              <div className="field-info">
                <label>Password</label>
              </div>
              <button 
                className="change-link"
                onClick={() => handleChangeClick('password', '')}
              >
                Change
              </button>
            </div>
          </>
        ) : (
          // Employee/Manager Fields
          <>
            <div className="account-field">
              <div className="field-info">
                <label>First Name</label>
                <span className="field-value">{userData.first_name}</span>
              </div>
              <button 
                className="change-link"
                onClick={() => handleChangeClick('first_name', userData.first_name)}
              >
                Change
              </button>
            </div>

            <div className="account-field">
              <div className="field-info">
                <label>Last Name</label>
                <span className="field-value">{userData.last_name}</span>
              </div>
              <button 
                className="change-link"
                onClick={() => handleChangeClick('last_name', userData.last_name)}
              >
                Change
              </button>
            </div>

            <div className="account-field">
              <div className="field-info">
                <label>Email</label>
                <span className="field-value">{userData.email}</span>
              </div>
              <button 
                className="change-link"
                onClick={() => handleChangeClick('email', userData.email)}
              >
                Change
              </button>
            </div>

            <div className="account-field">
              <div className="field-info">
                <label>Password</label>
              </div>
              <button 
                className="change-link"
                onClick={() => handleChangeClick('password', '')}
              >
                Change
              </button>
            </div>
          </>
        )}
      </div>

      {/* Change Dialog */}
      {showChangeDialog && (
        <div className="change-dialog-overlay" onClick={() => setShowChangeDialog(false)}>
          <div className="change-dialog" onClick={e => e.stopPropagation()}>
            <h2>Change {getFieldLabel(currentField)}</h2>
            
            <div className="dialog-form">
              <div className="form-group">
                <label>New {getFieldLabel(currentField)}</label>
                <input 
                  type={currentField === 'password' ? 'password' : 'text'}
                  value={newValue}
                  onChange={(e) => setNewValue(e.target.value)}
                  placeholder={`Enter new ${getFieldLabel(currentField).toLowerCase()}`}
                  autoFocus
                />
              </div>

              <div className="form-group">
                <label>Current Password (to confirm)</label>
                <input 
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter your current password"
                />
              </div>
            </div>

            <div className="dialog-buttons">
              <button 
                className="dialog-cancel-button"
                onClick={() => setShowChangeDialog(false)}
              >
                Cancel
              </button>
              <button 
                className="dialog-confirm-button"
                onClick={handleConfirmChange}
              >
                Confirm Change
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}