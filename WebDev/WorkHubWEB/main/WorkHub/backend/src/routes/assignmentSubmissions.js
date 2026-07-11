import express from 'express';
import { query } from '../config/database.js';

const router = express.Router();

// Get all assignment submissions by user ID
router.get('/user/:userId', async (req, res) => {
  try {
    const { userId } = req.params;
    
    const result = await query(
      `SELECT * FROM assignment_submissions 
       WHERE user_id = $1 
       ORDER BY submitted_at DESC`,
      [userId]
    );
    
    res.json({
      success: true,
      count: result.rows.length,
      data: result.rows
    });
  } catch (error) {
    console.error('Error fetching assignment submissions:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch assignment submissions'
    });
  }
});

// Get single assignment submission
router.get('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    
    const result = await query(
      'SELECT * FROM assignment_submissions WHERE id = $1',
      [id]
    );
    
    if (result.rows.length === 0) {
      return res.status(404).json({
        success: false,
        error: 'Assignment submission not found'
      });
    }
    
    res.json({
      success: true,
      data: result.rows[0]
    });
  } catch (error) {
    console.error('Error fetching assignment submission:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch assignment submission'
    });
  }
});

// Create assignment submission
router.post('/', async (req, res) => {
  try {
    const { assignment_id, user_id, notes } = req.body;
    
    if (!assignment_id || !user_id) {
      return res.status(400).json({
        success: false,
        error: 'assignment_id and user_id are required'
      });
    }
    
    const result = await query(
      `INSERT INTO assignment_submissions (assignment_id, user_id, notes, submitted_at)
       VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
       RETURNING *`,
      [assignment_id, user_id, notes || null]
    );
    
    res.status(201).json({
      success: true,
      message: 'Assignment submitted successfully',
      data: result.rows[0]
    });
  } catch (error) {
    console.error('Error creating assignment submission:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to create assignment submission'
    });
  }
});

// *** NEW: Delete assignment submission ***
router.delete('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    
    const result = await query(
      'DELETE FROM assignment_submissions WHERE id = $1 RETURNING *',
      [id]
    );
    
    if (result.rows.length === 0) {
      return res.status(404).json({
        success: false,
        error: 'Assignment submission not found'
      });
    }
    
    res.json({
      success: true,
      message: 'Assignment submission deleted successfully',
      data: result.rows[0]
    });
  } catch (error) {
    console.error('Error deleting assignment submission:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to delete assignment submission'
    });
  }
});

export default router;