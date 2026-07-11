import React from 'react'
// Utility function to generate user profile icon
// Works with WorkHub backend API (port 3000)
// FIXED: No process.env runtime access

// API Base URL - hardcoded for reliability
// Change this if your backend is on a different host/port
const API_BASE_URL = 'http://localhost:3000/api';

/**
 * Fetches user data from backend and generates a profile icon element
 * @param {number} userId - The ID of the user
 * @returns {Promise<Object>} Object containing user data and icon element
 */
export const getUserProfileIcon = async (userId) => {
  try {
    const response = await fetch(`${API_BASE_URL}/users/${userId}`);
    const result = await response.json();
    
    if (!result.success) {
      throw new Error('Failed to fetch user data');
    }
    
    const userData = result.data;
    const firstLetter = userData.first_name.charAt(0).toUpperCase();
    const backgroundColor = userData.avatar_color || '#667eea';
    const fullName = `${userData.first_name} ${userData.last_name}`;
    
    return {
      userData,
      iconElement: (
        <div 
          className="profile-icon" 
          style={{ 
            backgroundColor,
            width: '40px',
            height: '40px',
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontWeight: 'bold',
            color: '#fff',
            fontSize: '18px',
            flexShrink: 0
          }}
        >
          {firstLetter}
        </div>
      ),
      firstLetter,
      backgroundColor,
      fullName,
      firstName: userData.first_name,
      lastName: userData.last_name,
      email: userData.email,
      departmentName: userData.department_name,
      roleName: userData.role_name,
      avatarColor: userData.avatar_color
    };
  } catch (error) {
    console.error('Error fetching user profile:', error);
    // Return default values in case of error
    return {
      userData: null,
      iconElement: (
        <div 
          className="profile-icon" 
          style={{ 
            backgroundColor: '#667eea',
            width: '40px',
            height: '40px',
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontWeight: 'bold',
            color: '#fff',
            fontSize: '18px',
            flexShrink: 0
          }}
        >
          ?
        </div>
      ),
      firstLetter: '?',
      backgroundColor: '#667eea',
      fullName: 'Unknown User',
      firstName: 'Unknown',
      lastName: 'User',
      email: '',
      departmentName: '',
      roleName: '',
      avatarColor: '#667eea'
    };
  }
};

/**
 * Creates a static profile icon without API call (when user data is already available)
 * @param {string} firstName - User's first name
 * @param {string} avatarColor - User's avatar color (hex color code)
 * @param {number} size - Size of the icon in pixels (default 40)
 * @returns {JSX.Element} Profile icon element
 */
export const createProfileIcon = (firstName, avatarColor, size = 40) => {
  const firstLetter = firstName ? firstName.charAt(0).toUpperCase() : '?';
  const backgroundColor = avatarColor || '#667eea';
  
  return (
    <div 
      className="profile-icon" 
      style={{ 
        backgroundColor,
        width: `${size}px`,
        height: `${size}px`,
        borderRadius: '50%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontWeight: 'bold',
        color: '#fff',
        fontSize: `${size * 0.45}px`,
        flexShrink: 0
      }}
    >
      {firstLetter}
    </div>
  );
};

/**
 * Creates profile icon HTML string (for non-React contexts)
 * @param {string} firstName - User's first name
 * @param {string} avatarColor - User's avatar color
 * @param {number} size - Size in pixels
 * @returns {string} HTML string
 */
export const createProfileIconHTML = (firstName, avatarColor, size = 40) => {
  const firstLetter = firstName ? firstName.charAt(0).toUpperCase() : '?';
  const backgroundColor = avatarColor || '#667eea';
  const fontSize = size * 0.45;
  
  return `
    <div 
      class="profile-icon" 
      style="
        background-color: ${backgroundColor};
        width: ${size}px;
        height: ${size}px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        color: #fff;
        font-size: ${fontSize}px;
        flex-shrink: 0;
      "
    >
      ${firstLetter}
    </div>
  `;
};

/**
 * Generates a random avatar color from a predefined palette
 * @returns {string} Hex color code
 */
export const generateRandomAvatarColor = () => {
  const colors = [
    '#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b',
    '#fa709a', '#fee140', '#30cfd0', '#a8edea', '#ff6b6b',
    '#4ecdc4', '#feca57', '#48dbfb', '#ff9ff3', '#54a0ff',
    '#5f27cd', '#00d2d3', '#1dd1a1', '#ee5a6f'
  ];
  
  return colors[Math.floor(Math.random() * colors.length)];
};

/**
 * Gets initials from full name
 * @param {string} firstName - First name
 * @param {string} lastName - Last name (optional)
 * @returns {string} Initials (1 or 2 letters)
 */
export const getInitials = (firstName, lastName = '') => {
  const first = firstName ? firstName.charAt(0).toUpperCase() : '';
  const last = lastName ? lastName.charAt(0).toUpperCase() : '';
  return last ? `${first}${last}` : first || '?';
};

/**
 * Creates a profile icon with initials (first + last name)
 * @param {string} firstName - User's first name
 * @param {string} lastName - User's last name
 * @param {string} avatarColor - User's avatar color
 * @param {number} size - Size in pixels
 * @returns {JSX.Element} Profile icon with initials
 */
export const createProfileIconWithInitials = (firstName, lastName, avatarColor, size = 40) => {
  const initials = getInitials(firstName, lastName);
  const backgroundColor = avatarColor || '#667eea';
  
  return (
    <div 
      className="profile-icon" 
      style={{ 
        backgroundColor,
        width: `${size}px`,
        height: `${size}px`,
        borderRadius: '50%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontWeight: 'bold',
        color: '#fff',
        fontSize: `${size * 0.4}px`,
        flexShrink: 0,
        letterSpacing: initials.length > 1 ? '1px' : '0'
      }}
    >
      {initials}
    </div>
  );
};

/**
 * Validates if a color is a valid hex color
 * @param {string} color - Color string to validate
 * @returns {boolean} True if valid hex color
 */
export const isValidHexColor = (color) => {
  return /^#[0-9A-F]{6}$/i.test(color);
};

/**
 * Ensures the color is valid, returns default if not
 * @param {string} color - Color to validate
 * @param {string} defaultColor - Default color if invalid
 * @returns {string} Valid hex color
 */
export const ensureValidColor = (color, defaultColor = '#667eea') => {
  return isValidHexColor(color) ? color : defaultColor;
};

// Export all functions
export default {
  getUserProfileIcon,
  createProfileIcon,
  createProfileIconHTML,
  createProfileIconWithInitials,
  generateRandomAvatarColor,
  getInitials,
  isValidHexColor,
  ensureValidColor
};