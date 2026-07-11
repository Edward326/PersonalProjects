import express from 'express';
import { query } from '../config/database.js';

const router = express.Router();

// Get assignments for a department
router.get('/department/:deptId', async (req, res) => {
  try {
    const result = await query(
      `SELECT a.*, u.first_name, u.last_name
       FROM assignments a
       JOIN users u ON a.created_by = u.id
       WHERE a.department_id = $1
       ORDER BY a.created_at DESC`,
      [req.params.deptId]
    );
    res.json({ success: true, data: result.rows });
  } catch (error) {
    res.status(500).json({ success: false, error: 'Failed to fetch assignments' });
  }
});

// Get assignments for a user
router.get('/user/:userId', async (req, res) => {
  try {
    const result = await query(
      `SELECT a.*, au.assigned_at
       FROM assignments a
       JOIN assignment_users au ON a.id = au.assignment_id
       WHERE au.user_id = $1
       ORDER BY a.deadline`,
      [req.params.userId]
    );
    res.json({ success: true, data: result.rows });
  } catch (error) {
    res.status(500).json({ success: false, error: 'Failed to fetch user assignments' });
  }
});

// Create assignment
router.post('/', async (req, res) => {
  try {
    const { organization_id, department_id, created_by, title, description, category, deadline, priority } = req.body;
    const result = await query(
      `INSERT INTO assignments (organization_id, department_id, created_by, title, description, category, deadline, priority)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8) RETURNING *`,
      [organization_id, department_id, created_by, title, description, category, deadline, priority]
    );
    res.status(201).json({ success: true, data: result.rows[0] });
  } catch (error) {
    res.status(500).json({ success: false, error: 'Failed to create assignment' });
  }
});

// Add these routes to your existing assignments.js file

// Get single assignment by ID
router.get('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    
    const result = await query(
      `SELECT 
        a.*,
        u.first_name as creator_first_name,
        u.last_name as creator_last_name,
        d.name as department_name
      FROM assignments a
      JOIN users u ON a.created_by = u.id
      JOIN departments d ON a.department_id = d.id
      WHERE a.id = $1`,
      [id]
    );
    
    if (result.rows.length === 0) {
      return res.status(404).json({
        success: false,
        error: 'Assignment not found'
      });
    }
    
    res.json({
      success: true,
      data: result.rows[0]
    });
  } catch (error) {
    console.error('Error fetching assignment:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch assignment'
    });
  }
});

// Update assignment
router.put('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    const { title, description, deadline, priority, status, category } = req.body;
    
    // Build dynamic update query
    const updates = [];
    const values = [];
    let paramCount = 1;
    
    if (title !== undefined) {
      updates.push(`title = $${paramCount++}`);
      values.push(title);
    }
    
    if (description !== undefined) {
      updates.push(`description = $${paramCount++}`);
      values.push(description);
    }
    
    if (deadline !== undefined) {
      updates.push(`deadline = $${paramCount++}`);
      values.push(deadline);
    }
    
    if (priority !== undefined) {
      updates.push(`priority = $${paramCount++}`);
      values.push(priority);
    }
    
    if (status !== undefined) {
      updates.push(`status = $${paramCount++}`);
      values.push(status);
    }
    
    if (category !== undefined) {
      updates.push(`category = $${paramCount++}`);
      values.push(category);
    }
    
    if (updates.length === 0) {
      return res.status(400).json({
        success: false,
        error: 'No fields to update'
      });
    }
    
    updates.push(`updated_at = CURRENT_TIMESTAMP`);
    values.push(id);
    
    const updateQuery = `
      UPDATE assignments 
      SET ${updates.join(', ')}
      WHERE id = $${paramCount}
      RETURNING *
    `;
    
    const result = await query(updateQuery, values);
    
    if (result.rows.length === 0) {
      return res.status(404).json({
        success: false,
        error: 'Assignment not found'
      });
    }
    
    res.json({
      success: true,
      message: 'Assignment updated successfully',
      data: result.rows[0]
    });
  } catch (error) {
    console.error('Error updating assignment:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to update assignment'
    });
  }
});

// Delete assignment
router.delete('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    
    const result = await query(
      'DELETE FROM assignments WHERE id = $1 RETURNING *',
      [id]
    );
    
    if (result.rows.length === 0) {
      return res.status(404).json({
        success: false,
        error: 'Assignment not found'
      });
    }
    
    res.json({
      success: true,
      message: 'Assignment deleted successfully',
      data: result.rows[0]
    });
  } catch (error) {
    console.error('Error deleting assignment:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to delete assignment'
    });
  }
});

export default router;