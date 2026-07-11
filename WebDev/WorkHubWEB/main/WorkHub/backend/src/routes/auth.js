import express from 'express';
import { query } from '../config/database.js';
import { generateToken, verifyToken } from '../middleware/auth_middleware.js';

const router = express.Router();

// Login endpoint
router.post('/login', async (req, res) => {
  try {
    const { email, password, uniqueCode } = req.body;

    console.log('Login attempt:', { email, hasPassword: !!password, hasUniqueCode: !!uniqueCode });

    // Check if it's a CEO login (with unique code)
    if (uniqueCode) {
      const orgResult = await query(
        'SELECT * FROM organizations WHERE ceo_email = $1 AND is_active = TRUE',
        [email]
      );

      if (orgResult.rows.length === 0) {
        return res.status(404).json({
          success: false,
          message: 'CEO account not found'
        });
      }

      const org = orgResult.rows[0];

      // Verify password
      if (password !== org.ceo_password_hash) {
        return res.status(401).json({
          success: false,
          message: 'Invalid password'
        });
      }

      // Verify unique code
      if (uniqueCode !== org.unique_code) {
        return res.status(401).json({
          success: false,
          message: 'Invalid organization code'
        });
      }

      // Generate JWT token
      const token = generateToken(org.id, 'ceo', org.id);

      return res.json({
        success: true,
        message: 'CEO login successful',
        token,
        user: {
          id: org.id,
          name: org.name,
          email: org.ceo_email,
          userType: 'ceo',
          organizationId: org.id,
          organizationName: org.name
        }
      });
    }

    // Employee login
    const userResult = await query(
      'SELECT u.*, o.name as organization_name, d.name as department_name, r.name as role_name ' +
      'FROM users u ' +
      'LEFT JOIN organizations o ON u.organization_id = o.id ' +
      'LEFT JOIN departments d ON u.department_id = d.id ' +
      'LEFT JOIN roles r ON u.role_id = r.id ' +
      'WHERE u.email = $1 AND u.is_active = TRUE',
      [email]
    );

    if (userResult.rows.length === 0) {
      return res.status(404).json({
        success: false,
        message: 'Account not found'
      });
    }

    const user = userResult.rows[0];

    // Check if approved
    if (!user.is_approved) {
      return res.status(403).json({
        success: false,
        message: 'Your account is pending approval. Please wait for your administrator to approve your request.'
      });
    }

    // Verify password
    if (password !== user.password_hash) {
      return res.status(401).json({
        success: false,
        message: 'Invalid password'
      });
    }

    // Generate JWT token
    const token = generateToken(user.id, 'employee', user.organization_id);

    return res.json({
      success: true,
      message: 'Login successful',
      token,
      user: {
        id: user.id,
        firstName: user.first_name,
        lastName: user.last_name,
        email: user.email,
        userType: 'employee',
        organizationId: user.organization_id,
        organizationName: user.organization_name,
        departmentId: user.department_id,
        departmentName: user.department_name,
        roleId: user.role_id,
        roleName: user.role_name
      }
    });

  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({
      success: false,
      message: 'An error occurred during login'
    });
  }
});

// Check if email exists
router.post('/check-email', async (req, res) => {
  try {
    const { email } = req.body;

    // Check in organizations (CEO)
    const orgResult = await query(
      'SELECT id FROM organizations WHERE ceo_email = $1',
      [email]
    );

    if (orgResult.rows.length > 0) {
      return res.json({
        success: true,
        exists: true,
        userType: 'ceo'
      });
    }

    // Check in users (Employee)
    const userResult = await query(
      'SELECT id FROM users WHERE email = $1',
      [email]
    );

    if (userResult.rows.length > 0) {
      return res.json({
        success: true,
        exists: true,
        userType: 'employee'
      });
    }

    return res.json({
      success: true,
      exists: false
    });

  } catch (error) {
    console.error('Check email error:', error);
    res.status(500).json({
      success: false,
      message: 'An error occurred while checking email'
    });
  }
});

// Verify token and get current user
router.get('/me', verifyToken, async (req, res) => {
  try {
    const { userId, userType, organizationId } = req.user;

    if (userType === 'ceo') {
      const orgResult = await query(
        'SELECT id, name, ceo_email, industry, description FROM organizations WHERE id = $1',
        [userId]
      );

      if (orgResult.rows.length === 0) {
        return res.status(404).json({
          success: false,
          message: 'Organization not found'
        });
      }

      const org = orgResult.rows[0];

      return res.json({
        success: true,
        user: {
          id: org.id,
          name: org.name,
          email: org.ceo_email,
          userType: 'ceo',
          organizationId: org.id,
          organizationName: org.name
        }
      });
    }

    // Employee
    const userResult = await query(
      'SELECT u.*, o.name as organization_name, d.name as department_name, r.name as role_name ' +
      'FROM users u ' +
      'LEFT JOIN organizations o ON u.organization_id = o.id ' +
      'LEFT JOIN departments d ON u.department_id = d.id ' +
      'LEFT JOIN roles r ON u.role_id = r.id ' +
      'WHERE u.id = $1',
      [userId]
    );

    if (userResult.rows.length === 0) {
      return res.status(404).json({
        success: false,
        message: 'User not found'
      });
    }

    const user = userResult.rows[0];

    return res.json({
      success: true,
      user: {
        id: user.id,
        firstName: user.first_name,
        lastName: user.last_name,
        email: user.email,
        userType: 'employee',
        organizationId: user.organization_id,
        organizationName: user.organization_name,
        departmentId: user.department_id,
        departmentName: user.department_name,
        roleId: user.role_id,
        roleName: user.role_name
      }
    });

  } catch (error) {
    console.error('Get current user error:', error);
    res.status(500).json({
      success: false,
      message: 'An error occurred while fetching user data'
    });
  }
});

// Logout (client-side will remove token, but we can log it)
router.post('/logout', verifyToken, (req, res) => {
  // In a production app, you might want to blacklist the token
  res.json({
    success: true,
    message: 'Logged out successfully'
  });
});

export default router;