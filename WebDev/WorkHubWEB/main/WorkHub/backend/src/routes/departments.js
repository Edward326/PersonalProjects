import express from 'express';
import { query } from '../config/database.js';

const router = express.Router();

// Get all departments for an organization
router.get('/organization/:orgId', async (req, res) => {
  try {
    const { orgId } = req.params;
    
    const result = await query(
      `SELECT 
        d.*,
        u.first_name as manager_first_name,
        u.last_name as manager_last_name,
        (SELECT COUNT(*) FROM users WHERE department_id = d.id AND is_approved = TRUE) as employee_count
      FROM departments d
      LEFT JOIN users u ON d.manager_id = u.id
      WHERE d.organization_id = $1
      ORDER BY d.name`,
      [orgId]
    );
    
    res.json({
      success: true,
      count: result.rows.length,
      data: result.rows
    });
  } catch (error) {
    console.error('Error fetching departments:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch departments'
    });
  }
});

// Get department by ID with full details
router.get('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    
    const result = await query(
      `SELECT 
        d.*,
        o.name as organization_name,
        u.first_name as manager_first_name,
        u.last_name as manager_last_name
      FROM departments d
      JOIN organizations o ON d.organization_id = o.id
      LEFT JOIN users u ON d.manager_id = u.id
      WHERE d.id = $1`,
      [id]
    );
    
    if (result.rows.length === 0) {
      return res.status(404).json({
        success: false,
        error: 'Department not found'
      });
    }
    
    res.json({
      success: true,
      data: result.rows[0]
    });
  } catch (error) {
    console.error('Error fetching department:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch department'
    });
  }
});

// Get department by ID with full details
router.get('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    
    const result = await query(
      `SELECT 
        d.*,
        o.name as organization_name,
        u.first_name as manager_first_name,
        u.last_name as manager_last_name
      FROM departments d
      JOIN organizations o ON d.organization_id = o.id
      LEFT JOIN users u ON d.manager_id = u.id
      WHERE d.id = $1`,
      [id]
    );
    
    if (result.rows.length === 0) {
      return res.status(404).json({
        success: false,
        error: 'Department not found'
      });
    }
    
    res.json({
      success: true,
      data: result.rows[0]
    });
  } catch (error) {
    console.error('Error fetching department:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch department'
    });
  }
});

// Get all departments
router.get('/', async (req, res) => {
  try {
    // Check if organizationId query param is provided
    const { organizationId } = req.query;
    
    let queryStr = 'SELECT * FROM departments';
    let params = [];
    
    if (organizationId) {
      queryStr += ' WHERE organization_id = $1';
      params.push(organizationId);
    }
    
    queryStr += ' ORDER BY name';
    
    const result = await query(queryStr, params);
    
    res.json({ 
      success: true, 
      count: result.rows.length,
      data: result.rows 
    });
  } catch (error) {
    console.error('Error fetching departments:', error);
    res.status(500).json({ 
      success: false, 
      error: 'Failed to fetch departments' 
    });
  }
});

// Get department employees
router.get('/:id/employees', async (req, res) => {
  try {
    const { id } = req.params;
    
    const result = await query(
      `SELECT 
        u.id,
        u.first_name,
        u.last_name,
        u.email,
        u.avatar_color,
        u.hire_date,
        r.name as role_name,
        r.priority as role_priority
      FROM users u
      LEFT JOIN roles r ON u.role_id = r.id
      WHERE u.department_id = $1 AND u.is_approved = TRUE
      ORDER BY r.priority, u.first_name`,
      [id]
    );
    
    res.json({
      success: true,
      count: result.rows.length,
      data: result.rows
    });
  } catch (error) {
    console.error('Error fetching department employees:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch department employees'
    });
  }
});

// Get department attendance statistics
router.get('/:id/attendance', async (req, res) => {
  try {
    const { id } = req.params;
    
    const result = await query(
      `SELECT * FROM department_attendance_stats
       WHERE department_id = $1
       ORDER BY month DESC
       LIMIT 12`,
      [id]
    );
    
    res.json({
      success: true,
      data: result.rows
    });
  } catch (error) {
    console.error('Error fetching department attendance:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch attendance statistics'
    });
  }
});

// Create department
router.post('/', async (req, res) => {
  try {
    const {
      organization_id,
      name,
      description,
      has_header = true,
      header_background_color = '#2ecc71',
      header_display_type = 'both',
      header_position = 'center',
      logo_size = 'medium',
      sidebar_position = 'left',
      layout_color_hover = '#3498db',
      layout_color_clicked = '#2980b9',
      layout_color_selected = '#667eea'
    } = req.body;
    
    if (!organization_id || !name) {
      return res.status(400).json({
        success: false,
        error: 'Missing required fields: organization_id and name'
      });
    }
    
    const result = await query(
      `INSERT INTO departments (
        organization_id, 
        name, 
        description, 
        has_header, 
        header_background_color, 
        header_display_type, 
        header_position, 
        logo_size,
        sidebar_position,
        layout_color_hover,
        layout_color_clicked,
        layout_color_selected
      )
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
       RETURNING *`,
      [
        organization_id, 
        name, 
        description || null, 
        has_header, 
        header_background_color, 
        header_display_type, 
        header_position, 
        logo_size,
        sidebar_position,
        layout_color_hover,
        layout_color_clicked,
        layout_color_selected
      ]
    );
    
    res.status(201).json({
      success: true,
      message: 'Department created successfully',
      data: result.rows[0]
    });
  } catch (error) {
    console.error('Error creating department:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to create department',
      message: error.message
    });
  }
});

// Update department
router.put('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    const {
      name,
      description,
      manager_id,
      has_header,
      header_background_color,
      header_display_type,
      header_position,
      logo_size,
      sidebar_position,
      layout_color_hover,
      layout_color_clicked,
      layout_color_selected
    } = req.body;

    // Build dynamic update query
    const updates = [];
    const values = [];
    let paramCount = 1;

    if (name !== undefined) {
      updates.push(`name = $${paramCount++}`);
      values.push(name.trim());
    }

    if (description !== undefined) {
      updates.push(`description = $${paramCount++}`);
      values.push(description);
    }

    if (manager_id !== undefined) {
      updates.push(`manager_id = $${paramCount++}`);
      values.push(manager_id);
    }

    if (has_header !== undefined) {
      updates.push(`has_header = $${paramCount++}`);
      values.push(has_header);
    }

    if (header_background_color !== undefined) {
      updates.push(`header_background_color = $${paramCount++}`);
      values.push(header_background_color);
    }

    if (header_display_type !== undefined) {
      updates.push(`header_display_type = $${paramCount++}`);
      values.push(header_display_type);
    }

    if (header_position !== undefined) {
      updates.push(`header_position = $${paramCount++}`);
      values.push(header_position);
    }

    if (logo_size !== undefined) {
      updates.push(`logo_size = $${paramCount++}`);
      values.push(logo_size);
    }

    if (sidebar_position !== undefined) {
      updates.push(`sidebar_position = $${paramCount++}`);
      values.push(sidebar_position);
    }

    if (layout_color_hover !== undefined) {
      updates.push(`layout_color_hover = $${paramCount++}`);
      values.push(layout_color_hover);
    }

    if (layout_color_clicked !== undefined) {
      updates.push(`layout_color_clicked = $${paramCount++}`);
      values.push(layout_color_clicked);
    }

    if (layout_color_selected !== undefined) {
      updates.push(`layout_color_selected = $${paramCount++}`);
      values.push(layout_color_selected);
    }

    if (updates.length === 0) {
      return res.status(400).json({
        success: false,
        message: 'No fields to update'
      });
    }

    // Always update the updated_at timestamp
    updates.push(`updated_at = CURRENT_TIMESTAMP`);

    values.push(id);
    const updateQuery = `
      UPDATE departments 
      SET ${updates.join(', ')}
      WHERE id = $${paramCount}
      RETURNING *
    `;

    const result = await query(updateQuery, values);

    if (result.rows.length === 0) {
      return res.status(404).json({
        success: false,
        message: 'Department not found'
      });
    }

    res.json({
      success: true,
      message: 'Department updated successfully',
      data: result.rows[0]
    });

  } catch (error) {
    console.error('Error updating department:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to update department',
      error: error.message
    });
  }
});

// Delete department
router.delete('/:id', async (req, res) => {
  try {
    const { id } = req.params;

    // Check if department exists
    const checkResult = await query(
      'SELECT id FROM departments WHERE id = $1',
      [id]
    );

    if (checkResult.rows.length === 0) {
      return res.status(404).json({
        success: false,
        message: 'Department not found'
      });
    }

    // Delete department (CASCADE will handle related records if set in DB)
    const result = await query(
      'DELETE FROM departments WHERE id = $1 RETURNING *',
      [id]
    );

    res.json({
      success: true,
      message: 'Department deleted successfully',
      data: result.rows[0]
    });

  } catch (error) {
    console.error('Error deleting department:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to delete department',
      error: error.message
    });
  }
});

export default router;