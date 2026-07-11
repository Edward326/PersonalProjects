import express from 'express';
import { query } from '../config/database.js';

const router = express.Router();

// Get department attendance stats by department ID
router.get('/department/:departmentId', async (req, res) => {
  try {
    const { departmentId } = req.params;
    
    const result = await query(
      `SELECT * FROM department_attendance_stats 
       WHERE department_id = $1 
       ORDER BY month DESC`,
      [departmentId]
    );
    
    res.json({
      success: true,
      count: result.rows.length,
      data: result.rows
    });
  } catch (error) {
    console.error('Error fetching department attendance stats:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch department attendance stats'
    });
  }
});

// Get all department attendance stats
router.get('/', async (req, res) => {
  try {
    const result = await query(
      'SELECT * FROM department_attendance_stats ORDER BY month DESC'
    );
    
    res.json({
      success: true,
      count: result.rows.length,
      data: result.rows
    });
  } catch (error) {
    console.error('Error fetching department attendance stats:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch department attendance stats'
    });
  }
});

export default router;