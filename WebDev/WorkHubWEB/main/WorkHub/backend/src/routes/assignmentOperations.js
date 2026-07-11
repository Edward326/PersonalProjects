import express from 'express';
import { query } from '../config/database.js';

const router = express.Router();

// Get assignments with employee count for department
router.get('/department/:deptId/with-counts', async (req, res) => {
  try {
    const { deptId } = req.params;
    
    const result = await query(
      `SELECT 
        a.*,
        u.first_name as creator_first_name,
        u.last_name as creator_last_name,
        (SELECT COUNT(*) FROM assignment_users WHERE assignment_id = a.id) as employee_count
      FROM assignments a
      JOIN users u ON a.created_by = u.id
      WHERE a.department_id = $1
      ORDER BY a.created_at DESC`,
      [deptId]
    );
    
    // Filter assignments with at least one employee assigned
    const assignmentsWithEmployees = result.rows.filter(row => row.employee_count > 0);
    
    res.json({
      success: true,
      count: assignmentsWithEmployees.length,
      data: assignmentsWithEmployees
    });
  } catch (error) {
    console.error('Error fetching assignments with counts:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch assignments'
    });
  }
});

// Get users not assigned to specific assignment in department
router.get('/department/:deptId/unassigned-users/:assignmentId', async (req, res) => {
  try {
    const { deptId, assignmentId } = req.params;
    
    const result = await query(
      `SELECT u.id, u.first_name, u.last_name, u.email
       FROM users u
       WHERE u.department_id = $1 
       AND u.is_approved = TRUE
       AND u.id NOT IN (
         SELECT user_id FROM assignment_users WHERE assignment_id = $2
       )
       AND u.id NOT IN (
         SELECT user_id FROM assignment_submissions WHERE assignment_id = $2
       )
       ORDER BY u.first_name`,
      [deptId, assignmentId]
    );
    
    res.json({
      success: true,
      count: result.rows.length,
      data: result.rows
    });
  } catch (error) {
    console.error('Error fetching unassigned users:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch unassigned users'
    });
  }
});

// Delete assignment cascade
router.delete('/:id/cascade', async (req, res) => {
  try {
    const { id } = req.params;
    
    // Start transaction
    await query('BEGIN');
    
    try {
      // Delete assignment_users
      await query('DELETE FROM assignment_users WHERE assignment_id = $1', [id]);
      
      // Delete assignment_submissions
      await query('DELETE FROM assignment_submissions WHERE assignment_id = $1', [id]);
      
      // Delete assignment_relocations
      await query('DELETE FROM assignment_relocations WHERE assignment_id = $1', [id]);
      
      // Delete assignment
      const result = await query('DELETE FROM assignments WHERE id = $1 RETURNING *', [id]);
      
      if (result.rows.length === 0) {
        await query('ROLLBACK');
        return res.status(404).json({
          success: false,
          error: 'Assignment not found'
        });
      }
      
      await query('COMMIT');
      
      res.json({
        success: true,
        message: 'Assignment deleted successfully',
        data: result.rows[0]
      });
    } catch (error) {
      await query('ROLLBACK');
      throw error;
    }
  } catch (error) {
    console.error('Error deleting assignment:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to delete assignment'
    });
  }
});

// Batch remove users from assignment
router.post('/:assignmentId/remove-users', async (req, res) => {
  try {
    const { assignmentId } = req.params;
    const { userIds } = req.body;
    
    if (!userIds || userIds.length === 0) {
      return res.status(400).json({
        success: false,
        error: 'No user IDs provided'
      });
    }
    
    const placeholders = userIds.map((_, i) => `$${i + 2}`).join(',');
    await query(
      `DELETE FROM assignment_users 
       WHERE assignment_id = $1 AND user_id IN (${placeholders})`,
      [assignmentId, ...userIds]
    );
    
    res.json({
      success: true,
      message: 'Users removed from assignment'
    });
  } catch (error) {
    console.error('Error removing users:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to remove users'
    });
  }
});

// Batch add users to assignment
router.post('/:assignmentId/add-users', async (req, res) => {
  try {
    const { assignmentId } = req.params;
    const { userIds } = req.body;
    
    if (!userIds || userIds.length === 0) {
      return res.status(400).json({
        success: false,
        error: 'No user IDs provided'
      });
    }
    
    const values = userIds.map((userId, i) => 
      `($1, $${i + 2}, NOW())`
    ).join(',');
    
    await query(
      `INSERT INTO assignment_users (assignment_id, user_id, assigned_at)
       VALUES ${values}`,
      [assignmentId, ...userIds]
    );
    
    res.json({
      success: true,
      message: 'Users added to assignment'
    });
  } catch (error) {
    console.error('Error adding users:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to add users'
    });
  }
});

export default router;