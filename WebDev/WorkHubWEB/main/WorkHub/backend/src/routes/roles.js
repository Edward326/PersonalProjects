import express from 'express';
import { query } from '../config/database.js';

const router = express.Router();

// Get all roles
router.get('/', async (req, res) => {
  try {
    // Check if departmentId query param is provided
    const { departmentId, organizationId } = req.query;
    
    let queryStr = 'SELECT r.*, d.name as department_name FROM roles r LEFT JOIN departments d ON r.department_id = d.id';
    let params = [];
    let conditions = [];
    
    if (departmentId) {
      conditions.push(`r.department_id = $${params.length + 1}`);
      params.push(departmentId);
    }
    
    if (organizationId) {
      conditions.push(`r.organization_id = $${params.length + 1}`);
      params.push(organizationId);
    }
    
    if (conditions.length > 0) {
      queryStr += ' WHERE ' + conditions.join(' AND ');
    }
    
    queryStr += ' ORDER BY r.priority';
    
    const result = await query(queryStr, params);
    
    res.json({ 
      success: true, 
      count: result.rows.length,
      data: result.rows 
    });
  } catch (error) {
    console.error('Error fetching roles:', error);
    res.status(500).json({ 
      success: false, 
      error: 'Failed to fetch roles' 
    });
  }
});

// Get roles for a department
router.get('/department/:deptId', async (req, res) => {
  try {
    const result = await query(
      'SELECT * FROM roles WHERE department_id = $1 ORDER BY priority',
      [req.params.deptId]
    );
    res.json({ success: true, data: result.rows });
  } catch (error) {
    res.status(500).json({ success: false, error: 'Failed to fetch roles' });
  }
});

// Get all roles for an organization and for all deprtamens of it
router.get('/organization/:orgId', async (req, res) => {
  try {
    const result = await query(
      `SELECT r.*, d.name as department_name
       FROM roles r
       JOIN departments d ON r.department_id = d.id
       WHERE r.organization_id = $1
       ORDER BY d.name, r.priority`,
      [req.params.orgId]
    );
    res.json({ success: true, data: result.rows });
  } catch (error) {
    res.status(500).json({ success: false, error: 'Failed to fetch roles' });
  }
});

// Get an role properties by id
router.get('/:roleId', async (req, res) => {
  try {
    const result = await query(
      'SELECT * FROM roles WHERE id = $1',
      [req.params.roleId]
    );
    res.json({ success: true, data: result.rows[0] });
  } catch (error) {
    res.status(500).json({ success: false, error: 'Failed to fetch roles' });
  }
});

// Create new role
router.post('/', async (req, res) => {
  try {
    console.log('Creating role with data:', req.body);

    const {
      organization_id,
      department_id,
      name,
      priority,
      permissions = {},
      can_accept_requests = false,
      can_post_news = false,
      can_post_event = false,
      can_assign_tasks = false,
      can_receive_tasks = false,
      can_view_statistics = false,
      can_hire = false,
      can_reassign_tasks = false,
      can_modify_department_look = false,
      created_at
    } = req.body;

    // Validation
    if (!organization_id || !department_id || !name || !priority) {
      return res.status(400).json({
        success: false,
        message: 'Required fields: organization_id, department_id, name, priority'
      });
    }

    // Validate priority is a positive number
    if (priority < 1) {
      return res.status(400).json({
        success: false,
        message: 'Priority must be at least 1'
      });
    }

    const result = await query(
      `INSERT INTO roles (
        organization_id,
        department_id,
        name,
        priority,
        permissions,
        can_accept_requests,
        can_post_news,
        can_post_event,
        can_assign_tasks,
        can_receive_tasks,
        can_view_statistics,
        can_hire,
        can_reassign_tasks,
        can_modify_department_look,
        created_at
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
      RETURNING *`,
      [
        organization_id,
        department_id,
        name.trim(),
        priority,
        JSON.stringify(permissions),
        can_accept_requests,
        can_post_news,
        can_post_event,
        can_assign_tasks,
        can_receive_tasks,
        can_view_statistics,
        can_hire,
        can_reassign_tasks,
        can_modify_department_look,
        created_at || new Date().toISOString()
      ]
    );

    console.log('Role created successfully:', result.rows[0]);

    res.status(201).json({
      success: true,
      message: 'Role created successfully',
      data: result.rows[0]
    });

  } catch (error) {
    console.error('Error creating role:', error);
    console.error('Error stack:', error.stack);
    res.status(500).json({
      success: false,
      message: 'Failed to create role',
      error: error.message
    });
  }
});

// Update role
router.put('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    const {
      name,
      priority,
      permissions,
      can_accept_requests,
      can_post_news,
      can_post_event,
      can_assign_tasks,
      can_receive_tasks,
      can_view_statistics,
      can_hire,
      can_reassign_tasks,
      can_modify_department_look
    } = req.body;

    // Build dynamic update query
    const updates = [];
    const values = [];
    let paramCount = 1;

    if (name !== undefined) {
      updates.push(`name = $${paramCount++}`);
      values.push(name.trim());
    }

    if (priority !== undefined) {
      if (priority < 1) {
        return res.status(400).json({
          success: false,
          message: 'Priority must be at least 1'
        });
      }
      updates.push(`priority = $${paramCount++}`);
      values.push(priority);
    }

    if (permissions !== undefined) {
      updates.push(`permissions = $${paramCount++}`);
      values.push(JSON.stringify(permissions));
    }

    if (can_accept_requests !== undefined) {
      updates.push(`can_accept_requests = $${paramCount++}`);
      values.push(can_accept_requests);
    }

    if (can_post_news !== undefined) {
      updates.push(`can_post_news = $${paramCount++}`);
      values.push(can_post_news);
    }

    if (can_post_event !== undefined) {
      updates.push(`can_post_event = $${paramCount++}`);
      values.push(can_post_event);
    }

    if (can_assign_tasks !== undefined) {
      updates.push(`can_assign_tasks = $${paramCount++}`);
      values.push(can_assign_tasks);
    }

    if (can_receive_tasks !== undefined) {
      updates.push(`can_receive_tasks = $${paramCount++}`);
      values.push(can_receive_tasks);
    }

    if (can_view_statistics !== undefined) {
      updates.push(`can_view_statistics = $${paramCount++}`);
      values.push(can_view_statistics);
    }

    if (can_hire !== undefined) {
      updates.push(`can_hire = $${paramCount++}`);
      values.push(can_hire);
    }

    if (can_reassign_tasks !== undefined) {
      updates.push(`can_reassign_tasks = $${paramCount++}`);
      values.push(can_reassign_tasks);
    }

    if (can_modify_department_look !== undefined) {
      updates.push(`can_modify_department_look = $${paramCount++}`);
      values.push(can_modify_department_look);
    }

    if (updates.length === 0) {
      return res.status(400).json({
        success: false,
        message: 'No fields to update'
      });
    }

    values.push(id);
    const updateQuery = `
      UPDATE roles 
      SET ${updates.join(', ')}
      WHERE id = $${paramCount}
      RETURNING *
    `;

    const result = await query(updateQuery, values);

    if (result.rows.length === 0) {
      return res.status(404).json({
        success: false,
        message: 'Role not found'
      });
    }

    res.json({
      success: true,
      message: 'Role updated successfully',
      data: result.rows[0]
    });

  } catch (error) {
    console.error('Error updating role:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to update role',
      error: error.message
    });
  }
});

// Delete role
router.delete('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    
    const result = await query(
      'DELETE FROM roles WHERE id = $1 RETURNING *',
      [id]
    );

    if (result.rows.length === 0) {
      return res.status(404).json({
        success: false,
        message: 'Role not found'
      });
    }

    res.json({
      success: true,
      message: 'Role deleted successfully',
      data: result.rows[0]
    });

  } catch (error) {
    console.error('Error deleting role:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to delete role',
      error: error.message
    });
  }
});

export default router;