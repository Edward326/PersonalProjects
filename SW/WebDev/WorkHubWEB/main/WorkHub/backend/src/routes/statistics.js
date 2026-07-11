import express from 'express';
import { query } from '../config/database.js';

const router = express.Router();

// Get user statistics
router.get('/user/:userId', async (req, res) => {
  try {
    const result = await query(
      'SELECT * FROM user_statistics WHERE user_id = $1',
      [req.params.userId]
    );
    
    if (result.rows.length === 0) {
      return res.status(404).json({ success: false, error: 'Statistics not found' });
    }
    
    res.json({ success: true, data: result.rows[0] });
  } catch (error) {
    res.status(500).json({ success: false, error: 'Failed to fetch statistics' });
  }
});

// Get organization statistics
router.get('/organization/:orgId', async (req, res) => {
  try {
    const [employees, departments, assignments, news] = await Promise.all([
      query('SELECT COUNT(*) as count FROM users WHERE organization_id = $1 AND is_approved = TRUE', [req.params.orgId]),
      query('SELECT COUNT(*) as count FROM departments WHERE organization_id = $1', [req.params.orgId]),
      query('SELECT COUNT(*) as count FROM assignments WHERE organization_id = $1', [req.params.orgId]),
      query('SELECT COUNT(*) as count FROM news_posts WHERE organization_id = $1', [req.params.orgId])
    ]);
    
    res.json({
      success: true,
      data: {
        total_employees: parseInt(employees.rows[0].count),
        total_departments: parseInt(departments.rows[0].count),
        total_assignments: parseInt(assignments.rows[0].count),
        total_news: parseInt(news.rows[0].count)
      }
    });
  } catch (error) {
    res.status(500).json({ success: false, error: 'Failed to fetch statistics' });
  }
});

// Get new hires tracking
router.get('/hires/:orgId', async (req, res) => {
  try {
    const result = await query(
      `SELECT * FROM new_hires_tracking 
       WHERE organization_id = $1 
       ORDER BY year DESC, month DESC 
       LIMIT 12`,
      [req.params.orgId]
    );
    
    res.json({ success: true, data: result.rows });
  } catch (error) {
    res.status(500).json({ success: false, error: 'Failed to fetch hire statistics' });
  }
});

// *** NEW: Delete user statistics ***
router.delete('/user/:userId', async (req, res) => {
  try {
    const { userId } = req.params;
    
    const result = await query(
      'DELETE FROM user_statistics WHERE user_id = $1 RETURNING *',
      [userId]
    );
    
    res.json({
      success: true,
      message: 'User statistics deleted successfully',
      count: result.rows.length,
      data: result.rows
    });
  } catch (error) {
    console.error('Error deleting user statistics:', error);
    res.status(500).json({ 
      success: false, 
      error: 'Failed to delete user statistics' 
    });
  }
});

export default router;