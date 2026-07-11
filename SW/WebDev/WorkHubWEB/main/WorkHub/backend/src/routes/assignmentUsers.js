import express from 'express';
import { query } from '../config/database.js';

const router = express.Router();

// Get all assignment users
router.get('/', async (req, res) => {
  try {
    const result = await query(
      `SELECT 
        au.*,
        a.title as assignment_title,
        u.first_name,
        u.last_name,
        u.email
      FROM assignment_users au
      JOIN assignments a ON au.assignment_id = a.id
      JOIN users u ON au.user_id = u.id
      ORDER BY au.assigned_at DESC`
    );
    
    res.json({
      success: true,
      count: result.rows.length,
      data: result.rows
    });
  } catch (error) {
    console.error('Error fetching assignment users:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch assignment users'
    });
  }
});

// Get assignment users by assignment ID
router.get('/assignment/:assignmentId', async (req, res) => {
  try {
    const { assignmentId } = req.params;
    
    const result = await query(
      `SELECT 
        au.*,
        u.first_name,
        u.last_name,
        u.email,
        u.avatar_color
      FROM assignment_users au
      JOIN users u ON au.user_id = u.id
      WHERE au.assignment_id = $1
      ORDER BY au.assigned_at`,
      [assignmentId]
    );
    
    res.json({
      success: true,
      count: result.rows.length,
      data: result.rows
    });
  } catch (error) {
    console.error('Error fetching assignment users:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch assignment users'
    });
  }
});

// Get assignments by user ID
router.get('/user/:userId', async (req, res) => {
  try {
    const { userId } = req.params;
    
    const result = await query(
      `SELECT 
        au.*,
        a.title,
        a.description,
        a.deadline,
        a.priority,
        a.status,
        a.category
      FROM assignment_users au
      JOIN assignments a ON au.assignment_id = a.id
      WHERE au.user_id = $1
      ORDER BY a.deadline`,
      [userId]
    );
    
    res.json({
      success: true,
      count: result.rows.length,
      data: result.rows
    });
  } catch (error) {
    console.error('Error fetching user assignments:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch user assignments'
    });
  }
});

// Create assignment user (assign user to assignment)
router.post('/', async (req, res) => {
  try {
    const { assignment_id, user_id, assigned_at } = req.body;
    
    if (!assignment_id || !user_id) {
      return res.status(400).json({
        success: false,
        error: 'assignment_id and user_id are required'
      });
    }
    
    // Check if assignment exists
    const assignmentCheck = await query(
      'SELECT id FROM assignments WHERE id = $1',
      [assignment_id]
    );
    
    if (assignmentCheck.rows.length === 0) {
      return res.status(404).json({
        success: false,
        error: 'Assignment not found'
      });
    }
    
    // Check if user exists
    const userCheck = await query(
      'SELECT id FROM users WHERE id = $1',
      [user_id]
    );
    
    if (userCheck.rows.length === 0) {
      return res.status(404).json({
        success: false,
        error: 'User not found'
      });
    }
    
    // Check if assignment is already assigned to this user
    const existingCheck = await query(
      'SELECT id FROM assignment_users WHERE assignment_id = $1 AND user_id = $2',
      [assignment_id, user_id]
    );
    
    if (existingCheck.rows.length > 0) {
      return res.status(409).json({
        success: false,
        error: 'User is already assigned to this assignment'
      });
    }
    
    const result = await query(
      `INSERT INTO assignment_users (assignment_id, user_id, assigned_at)
       VALUES ($1, $2, $3)
       RETURNING *`,
      [assignment_id, user_id, assigned_at || new Date()]
    );
    
    res.status(201).json({
      success: true,
      message: 'User assigned to assignment successfully',
      data: result.rows[0]
    });
  } catch (error) {
    console.error('Error creating assignment user:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to assign user to assignment'
    });
  }
});

// Delete assignment user by assignment ID and user ID
router.delete('/assignment/:assignmentId/user/:userId', async (req, res) => {
  try {
    const { assignmentId, userId } = req.params;
    
    const result = await query(
      'DELETE FROM assignment_users WHERE assignment_id = $1 AND user_id = $2 RETURNING *',
      [assignmentId, userId]
    );
    
    if (result.rows.length === 0) {
      return res.status(404).json({
        success: false,
        error: 'Assignment user relationship not found'
      });
    }
    
    res.json({
      success: true,
      message: 'User removed from assignment successfully',
      data: result.rows[0]
    });
  } catch (error) {
    console.error('Error deleting assignment user:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to remove user from assignment'
    });
  }
});

// Delete assignment user by ID
router.delete('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    
    const result = await query(
      'DELETE FROM assignment_users WHERE id = $1 RETURNING *',
      [id]
    );
    
    if (result.rows.length === 0) {
      return res.status(404).json({
        success: false,
        error: 'Assignment user not found'
      });
    }
    
    res.json({
      success: true,
      message: 'Assignment user deleted successfully',
      data: result.rows[0]
    });
  } catch (error) {
    console.error('Error deleting assignment user:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to delete assignment user'
    });
  }
});

// *** NEW: Delete all assignment_users for a specific user ***
router.delete('/user/:userId/all', async (req, res) => {
  try {
    const { userId } = req.params;
    
    const result = await query(
      'DELETE FROM assignment_users WHERE user_id = $1 RETURNING *',
      [userId]
    );
    
    res.json({
      success: true,
      message: `Deleted ${result.rows.length} assignment(s) for user`,
      count: result.rows.length,
      data: result.rows
    });
  } catch (error) {
    console.error('Error deleting user assignments:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to delete user assignments'
    });
  }
});

// Get single assignment by ID (helper route for the reassignment section)
router.get('/assignments/:id', async (req, res) => {
  try {
    const { id } = req.params;
    
    const result = await query(
      'SELECT * FROM assignments WHERE id = $1',
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

export default router;