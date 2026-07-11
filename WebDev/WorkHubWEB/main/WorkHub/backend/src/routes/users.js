import express from 'express';
import { query } from '../config/database.js';

const router = express.Router();

// Get all users for an organization
router.get('/organization/:orgId', async (req, res) => {
  try {
    const { orgId } = req.params;
    const result = await query(
      `SELECT u.*, d.name as department_name, r.name as role_name
       FROM users u
       LEFT JOIN departments d ON u.department_id = d.id
       LEFT JOIN roles r ON u.role_id = r.id
       WHERE u.organization_id = $1 AND u.is_approved = TRUE
       ORDER BY u.first_name`,
      [orgId]
    );
    res.json({ success: true, data: result.rows });
  } catch (error) {
    res.status(500).json({ success: false, error: 'Failed to fetch users' });
  }
});

// Get user by ID
router.get('/:id', async (req, res) => {
  try {
    const result = await query('SELECT * FROM users WHERE id = $1', [req.params.id]);
    if (result.rows.length === 0) {
      return res.status(404).json({ success: false, error: 'User not found' });
    }
    res.json({ success: true, data: result.rows[0] });
  } catch (error) {
    res.status(500).json({ success: false, error: 'Failed to fetch user' });
  }
});

// Get all users
router.get('/', async (req, res) => {
  try {
    const result = await query('SELECT * FROM users');
    if (result.rows.length === 0) {
      return res.status(404).json({ success: false, error: 'User not found' });
    }
    res.json({ success: true, data: result.rows });
  } catch (error) {
    res.status(500).json({ success: false, error: 'Failed to fetch users' });
  }
});

// Get all users by department ID
router.get('/department/:departmentId', async (req, res) => {
  try {
    const { departmentId } = req.params;
    
    const result = await query(
      `SELECT 
        u.*,
        r.name as role_name,
        r.priority as role_priority,
        d.name as department_name
      FROM users u
      LEFT JOIN roles r ON u.role_id = r.id
      LEFT JOIN departments d ON u.department_id = d.id
      WHERE u.department_id = $1 AND u.is_approved = true
      ORDER BY r.priority ASC, u.first_name ASC`,
      [departmentId]
    );
    
    res.json({
      success: true,
      count: result.rows.length,
      data: result.rows
    });
  } catch (error) {
    console.error('Error fetching users by department:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch users by department'
    });
  }
});

router.get('/organization/:organizationId', async (req, res) => {
  try {
    const { organizationId } = req.params;
    
    const result = await query(
      `SELECT 
        u.*,
        r.name as role_name,
        d.name as department_name
      FROM users u
      LEFT JOIN roles r ON u.role_id = r.id
      LEFT JOIN departments d ON u.department_id = d.id
      WHERE u.organization_id = $1 AND u.is_approved = true
      ORDER BY u.first_name ASC`,
      [organizationId]
    );
    
    res.json({
      success: true,
      count: result.rows.length,
      data: result.rows
    });
  } catch (error) {
    console.error('Error fetching users by organization:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch users by organization'
    });
  }
});

// NEW ROUTE: Get users by department, excluding a specific user (e.g., the manager viewing the list)
router.get('/department/:departmentId/exclude/:userId', async (req, res) => {
  try {
    const { departmentId, userId } = req.params;
    
    // Simple validation to ensure IDs are provided and look numeric
    if (!departmentId || !userId || isNaN(departmentId) || isNaN(userId)) {
      return res.status(400).json({ success: false, error: 'Invalid departmentId or userId provided.' });
    }

    console.log(`Fetching users for Department ID: ${departmentId}, excluding User ID: ${userId}`);

    const result = await query(
      `SELECT u.*, r.name as role_name
       FROM users u
       LEFT JOIN roles r ON u.role_id = r.id
       WHERE u.department_id = $1 
         AND u.id != $2 
         AND u.is_approved = TRUE
       ORDER BY u.first_name`,
      [departmentId, userId]
    );

    res.json({ success: true, data: result.rows });
  } catch (error) {
    // CRITICAL: This is where the error was likely happening before, now we catch and log it!
    console.error('Error fetching users by department and exclusion:', error); 
    res.status(500).json({ success: false, error: 'Failed to fetch department users with exclusion filter.' });
  }
});

// Create new user (for employee registration)
router.post('/', async (req, res) => {
  try {
    const {
      first_name,
      last_name,
      email,
      password_hash,
      avatar_color,
      organization_id,
      department_id,
      role_id,
      is_approved,
      is_active,
      hire_date
    } = req.body;

    // Validation
    if (!first_name || !last_name || !email || !password_hash || !organization_id) {
      return res.status(400).json({
        success: false,
        message: 'first_name, last_name, email, password_hash, and organization_id are required'
      });
    }

    // Validate first name starts with a letter
    if (!/^[a-zA-Z]/.test(first_name.trim())) {
      return res.status(400).json({
        success: false,
        message: 'First name must start with a letter'
      });
    }

    // Validate last name starts with a letter
    if (!/^[a-zA-Z]/.test(last_name.trim())) {
      return res.status(400).json({
        success: false,
        message: 'Last name must start with a letter'
      });
    }

    // Validate email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      return res.status(400).json({
        success: false,
        message: 'Invalid email format'
      });
    }

    // Check if email already exists
    const existingUser = await query(
      'SELECT id FROM users WHERE email = $1',
      [email]
    );

    if (existingUser.rows.length > 0) {
      return res.status(409).json({
        success: false,
        message: 'Email already exists'
      });
    }

    // Insert new user
    const result = await query(
      `INSERT INTO users (
        first_name,
        last_name,
        email,
        password_hash,
        avatar_color,
        organization_id,
        department_id,
        role_id,
        is_approved,
        is_active,
        hire_date
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
      RETURNING id, first_name, last_name, email, avatar_color, organization_id, is_approved, is_active, created_at`,
      [
        first_name.trim(),
        last_name.trim(),
        email.toLowerCase().trim(),
        password_hash,
        avatar_color || '#667eea',
        organization_id,
        department_id || null,
        role_id || null,
        is_approved !== undefined ? is_approved : false,
        is_active !== undefined ? is_active : true,
        hire_date || null
      ]
    );

    res.status(201).json({
      success: true,
      message: 'User created successfully',
      data: result.rows[0]
    });

  } catch (error) {
    console.error('Error creating user:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to create user',
      error: error.message
    });
  }
});

// Update user
router.put('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    const {
      first_name,
      last_name,
      email,
      department_id,
      role_id,
      is_approved,
      is_active,
      password_hash
    } = req.body;

    // Build dynamic update query
    const updates = [];
    const values = [];
    let paramCount = 1;

    if (first_name !== undefined) {
      if (!/^[a-zA-Z]/.test(first_name.trim())) {
        return res.status(400).json({
          success: false,
          message: 'First name must start with a letter'
        });
      }
      updates.push(`first_name = $${paramCount++}`);
      values.push(first_name.trim());
    }

    if (last_name !== undefined) {
      if (!/^[a-zA-Z]/.test(last_name.trim())) {
        return res.status(400).json({
          success: false,
          message: 'Last name must start with a letter'
        });
      }
      updates.push(`last_name = $${paramCount++}`);
      values.push(last_name.trim());
    }

    if (email !== undefined) {
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      if (!emailRegex.test(email)) {
        return res.status(400).json({
          success: false,
          message: 'Invalid email format'
        });
      }
      updates.push(`email = $${paramCount++}`);
      values.push(email.toLowerCase().trim());
    }

    if (department_id !== undefined) {
      updates.push(`department_id = $${paramCount++}`);
      values.push(department_id);
    }

    if (role_id !== undefined) {
      updates.push(`role_id = $${paramCount++}`);
      values.push(role_id);
    }

    if (is_approved !== undefined) {
      updates.push(`is_approved = $${paramCount++}`);
      values.push(is_approved);
    }

    if (is_active !== undefined) {
      updates.push(`is_active = $${paramCount++}`);
      values.push(is_active);
    }

    if (password_hash !== undefined) {
      updates.push(`password_hash = $${paramCount++}`);
      values.push(password_hash);
    }

    if (updates.length === 0) {
      return res.status(400).json({
        success: false,
        message: 'No fields to update'
      });
    }

    values.push(id);
    const updateQuery = `
      UPDATE users 
      SET ${updates.join(', ')}
      WHERE id = $${paramCount}
      RETURNING *
    `;

    const result = await query(updateQuery, values);

    if (result.rows.length === 0) {
      return res.status(404).json({
        success: false,
        message: 'User not found'
      });
    }

    res.json({
      success: true,
      message: 'User updated successfully',
      data: result.rows[0]
    });

  } catch (error) {
    console.error('Error updating user:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to update user',
      error: error.message
    });
  }
});

// Delete user
router.delete('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    
    const result = await query(
      'DELETE FROM users WHERE id = $1 RETURNING *',
      [id]
    );

    if (result.rows.length === 0) {
      return res.status(404).json({
        success: false,
        message: 'User not found'
      });
    }

    res.json({
      success: true,
      message: 'User deleted successfully',
      data: result.rows[0]
    });

  } catch (error) {
    console.error('Error deleting user:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to delete user',
      error: error.message
    });
  }
});

export default router;