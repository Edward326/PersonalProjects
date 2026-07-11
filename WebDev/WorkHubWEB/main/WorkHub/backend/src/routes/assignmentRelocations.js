import express from 'express';
import { query } from '../config/database.js';

const router = express.Router();

// Get assignment relocations by department with status filter
router.get('/department/:deptId', async (req, res) => {
  try {
    const { deptId } = req.params;
    const { status } = req.query;
    
    let queryStr = `
      SELECT 
        ar.*,
        a.title as assignment_title,
        fu.first_name as from_user_first_name,
        fu.last_name as from_user_last_name,
        CONCAT(fu.first_name, ' ', fu.last_name) as from_user_name,
        tu.first_name as to_user_first_name,
        tu.last_name as to_user_last_name,
        CONCAT(tu.first_name, ' ', tu.last_name) as to_user_name
      FROM assignment_relocations ar
      JOIN assignments a ON ar.assignment_id = a.id
      JOIN users fu ON ar.from_user_id = fu.id
      LEFT JOIN users tu ON ar.to_user_id = tu.id
      WHERE ar.department_id = $1
    `;
    
    const params = [deptId];
    
    if (status) {
      queryStr += ' AND ar.status = $2';
      params.push(status);
    }
    
    queryStr += ' ORDER BY ar.requested_at DESC';
    
    const result = await query(queryStr, params);
    
    res.json({
      success: true,
      count: result.rows.length,
      data: result.rows
    });
  } catch (error) {
    console.error('Error fetching assignment relocations:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch assignment relocations'
    });
  }
});

// Get single assignment relocation by ID
router.get('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    
    const result = await query(
      'SELECT * FROM assignment_relocations WHERE id = $1',
      [id]
    );
    
    if (result.rows.length === 0) {
      return res.status(404).json({
        success: false,
        error: 'Assignment relocation not found'
      });
    }
    
    res.json({
      success: true,
      data: result.rows[0]
    });
  } catch (error) {
    console.error('Error fetching assignment relocation:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch assignment relocation'
    });
  }
});

router.get('/user/:userId', async (req, res) => {
  try {
    const { userId } = req.params;
    
    const result = await query(
      `SELECT * FROM assignment_relocations 
       WHERE from_user_id = $1 OR to_user_id = $1
       ORDER BY requested_at DESC`,
      [userId]
    );
    
    res.json({
      success: true,
      count: result.rows.length,
      data: result.rows
    });
  } catch (error) {
    console.error('Error fetching assignment relocations:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch assignment relocations'
    });
  }
});

// Update assignment relocation (for approval/rejection)
router.put('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    const { status, reviewed_at, reviewed_by } = req.body;
    
    // Check if relocation exists
    const checkResult = await query(
      'SELECT id FROM assignment_relocations WHERE id = $1',
      [id]
    );
    
    if (checkResult.rows.length === 0) {
      return res.status(404).json({
        success: false,
        error: 'Assignment relocation not found'
      });
    }
    
    const result = await query(
      `UPDATE assignment_relocations 
       SET status = $1, reviewed_at = $2, reviewed_by = $3
       WHERE id = $4
       RETURNING *`,
      [status, reviewed_at, reviewed_by, id]
    );
    
    res.json({
      success: true,
      message: 'Assignment relocation updated successfully',
      data: result.rows[0]
    });
  } catch (error) {
    console.error('Error updating assignment relocation:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to update assignment relocation'
    });
  }
});

// Create assignment relocation
router.post('/', async (req, res) => {
  try {
    const {
      assignment_id,
      department_id,
      from_user_id,
      to_user_id,
      reason
    } = req.body;
    
    if (!assignment_id || !department_id || !from_user_id || !reason) {
      return res.status(400).json({
        success: false,
        error: 'Missing required fields'
      });
    }
    
    const result = await query(
      `INSERT INTO assignment_relocations (
        assignment_id,
        department_id,
        from_user_id,
        to_user_id,
        reason,
        status,
        requested_at
      ) VALUES ($1, $2, $3, $4, $5, $6, $7)
      RETURNING *`,
      [assignment_id, department_id, from_user_id, to_user_id || null, reason, 'pending', new Date()]
    );
    
    res.status(201).json({
      success: true,
      message: 'Assignment relocation created successfully',
      data: result.rows[0]
    });
  } catch (error) {
    console.error('Error creating assignment relocation:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to create assignment relocation'
    });
  }
});

// Delete assignment relocation
router.delete('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    
    const result = await query(
      'DELETE FROM assignment_relocations WHERE id = $1 RETURNING *',
      [id]
    );
    
    if (result.rows.length === 0) {
      return res.status(404).json({
        success: false,
        error: 'Assignment relocation not found'
      });
    }
    
    res.json({
      success: true,
      message: 'Assignment relocation deleted successfully',
      data: result.rows[0]
    });
  } catch (error) {
    console.error('Error deleting assignment relocation:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to delete assignment relocation'
    });
  }
});

export default router;