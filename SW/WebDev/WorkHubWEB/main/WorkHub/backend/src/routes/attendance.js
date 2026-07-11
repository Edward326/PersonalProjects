import express from 'express';
import { query } from '../config/database.js';

const router = express.Router();

router.get('/user/:userId', async (req, res) => {
  try {
    const { userId } = req.params;
    const { isComplete } = req.query;
    
    let queryStr = 'SELECT * FROM attendance WHERE user_id = $1';
    const params = [userId];
    
    if (isComplete !== undefined) {
      queryStr += ' AND is_complete = $2';
      params.push(isComplete === 'true');
    }
    
    queryStr += ' ORDER BY date DESC';
    
    const result = await query(queryStr, params);
    
    res.json({
      success: true,
      count: result.rows.length,
      data: result.rows
    });
  } catch (error) {
    console.error('Error fetching attendance:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch attendance'
    });
  }
});

// Get all attendance records
router.get('/', async (req, res) => {
  try {
    const result = await query('SELECT * FROM attendance ORDER BY date DESC');
    
    res.json({
      success: true,
      count: result.rows.length,
      data: result.rows
    });
  } catch (error) {
    console.error('Error fetching attendance:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch attendance'
    });
  }
});

// Clock in
router.post('/clock-in', async (req, res) => {
  try {
    const { user_id, organization_id, department_id } = req.body;
    const today = new Date().toISOString().split('T')[0];
    
    // Check if already clocked in today
    const existing = await query(
      'SELECT * FROM attendance WHERE user_id = $1 AND date = $2',
      [user_id, today]
    );
    
    if (existing.rows.length > 0) {
      return res.status(400).json({ success: false, error: 'Already clocked in today' });
    }
    
    const result = await query(
      `INSERT INTO attendance (user_id, organization_id, department_id, date, check_in_time)
       VALUES ($1, $2, $3, $4, NOW()) RETURNING *`,
      [user_id, organization_id, department_id, today]
    );
    
    res.json({ success: true, message: 'Clocked in successfully', data: result.rows[0] });
  } catch (error) {
    res.status(500).json({ success: false, error: 'Failed to clock in' });
  }
});

// Clock out
router.post('/clock-out', async (req, res) => {
  try {
    const { user_id } = req.body;
    const today = new Date().toISOString().split('T')[0];
    
    const result = await query(
      `UPDATE attendance 
       SET check_out_time = NOW(), is_complete = TRUE
       WHERE user_id = $1 AND date = $2 AND check_out_time IS NULL
       RETURNING *`,
      [user_id, today]
    );
    
    if (result.rows.length === 0) {
      return res.status(400).json({ success: false, error: 'No active clock-in found' });
    }
    
    res.json({ success: true, message: 'Clocked out successfully', data: result.rows[0] });
  } catch (error) {
    res.status(500).json({ success: false, error: 'Failed to clock out' });
  }
});

// *** NEW: Delete all attendance records for a user ***
router.delete('/user/:userId/all', async (req, res) => {
  try {
    const { userId } = req.params;
    
    const result = await query(
      'DELETE FROM attendance WHERE user_id = $1 RETURNING *',
      [userId]
    );
    
    res.json({
      success: true,
      message: `Deleted ${result.rows.length} attendance record(s) for user`,
      count: result.rows.length,
      data: result.rows
    });
  } catch (error) {
    console.error('Error deleting user attendance:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to delete user attendance'
    });
  }
});

export default router;