import express from 'express';
import { query } from '../config/database.js';

const router = express.Router();

// Get events by department
router.get('/department/:deptId', async (req, res) => {
  try {
    const { deptId } = req.params;
    const { limit = 50, offset = 0 } = req.query;
    
    const result = await query(
      `SELECT 
        e.*,
        u.first_name,
        u.last_name,
        u.avatar_color
      FROM events e
      JOIN users u ON e.created_by = u.id
      WHERE e.department_id = $1 AND e.is_personal = FALSE
      ORDER BY e.event_date ASC
      LIMIT $2 OFFSET $3`,
      [deptId, limit, offset]
    );
    
    res.json({
      success: true,
      count: result.rows.length,
      data: result.rows
    });
  } catch (error) {
    console.error('Error fetching department events:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch department events'
    });
  }
});

// Get events by organization (all departments) - for CEO
router.get('/organization/:orgId', async (req, res) => {
  try {
    const { orgId } = req.params;
    const { limit = 50, offset = 0 } = req.query;
    
    const result = await query(
      `SELECT 
        e.*,
        u.first_name,
        u.last_name,
        u.avatar_color,
        d.name as department_name
      FROM events e
      JOIN users u ON e.created_by = u.id
      JOIN departments d ON e.department_id = d.id
      WHERE e.organization_id = $1 AND e.is_personal = FALSE
      ORDER BY e.event_date ASC
      LIMIT $2 OFFSET $3`,
      [orgId, limit, offset]
    );
    
    res.json({
      success: true,
      count: result.rows.length,
      data: result.rows
    });
  } catch (error) {
    console.error('Error fetching organization events:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch organization events'
    });
  }
});

// *** NEW: Get events by creator ***
router.get('/creator/:creatorId', async (req, res) => {
  try {
    const { creatorId } = req.params;
    
    const result = await query(
      `SELECT 
        e.*,
        u.first_name,
        u.last_name,
        u.avatar_color,
        d.name as department_name,
        o.name as organization_name
      FROM events e
      JOIN users u ON e.created_by = u.id
      JOIN departments d ON e.department_id = d.id
      JOIN organizations o ON e.organization_id = o.id
      WHERE e.created_by = $1
      ORDER BY e.event_date ASC`,
      [creatorId]
    );
    
    res.json({
      success: true,
      count: result.rows.length,
      data: result.rows
    });
  } catch (error) {
    console.error('Error fetching creator events:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch creator events'
    });
  }
});

// Get single event by ID
router.get('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    
    const result = await query(
      `SELECT 
        e.*,
        u.first_name,
        u.last_name,
        u.avatar_color,
        d.name as department_name
      FROM events e
      JOIN users u ON e.created_by = u.id
      JOIN departments d ON e.department_id = d.id
      WHERE e.id = $1`,
      [id]
    );
    
    if (result.rows.length === 0) {
      return res.status(404).json({
        success: false,
        error: 'Event not found'
      });
    }
    
    res.json({
      success: true,
      data: result.rows[0]
    });
  } catch (error) {
    console.error('Error fetching event:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch event'
    });
  }
});

router.get('/', async (req, res) => {
  try {
    const result = await query(
      `SELECT *
      FROM events e
      ORDER BY e.event_date ASC`
    );
    res.json({
      success: true,
      count: result.rows.length,
      data: result.rows
    });
  } catch (error) {
    console.error('Error fetching all events:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch all events'
    });
  }
});

// Create event
router.post('/', async (req, res) => {
  try {
    const { 
      title, 
      description, 
      event_date, 
      location, 
      is_personal = false,
      created_by,
      department_id,
      organization_id
    } = req.body;
    
    if (!title || !description || !event_date || !location || !created_by || !department_id || !organization_id) {
      return res.status(400).json({
        success: false,
        error: 'Missing required fields'
      });
    }
    
    const result = await query(
      `INSERT INTO events (
        title, 
        description, 
        event_date, 
        location, 
        is_personal, 
        created_by, 
        department_id, 
        organization_id
      )
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
       RETURNING *`,
      [title, description, event_date, location, is_personal, created_by, department_id, organization_id]
    );
    
    res.status(201).json({
      success: true,
      message: 'Event created successfully',
      data: result.rows[0]
    });
  } catch (error) {
    console.error('Error creating event:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to create event'
    });
  }
});

// Update event
router.put('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    const { title, description, event_date, location } = req.body;
    
    // Check if event exists
    const checkResult = await query(
      'SELECT id FROM events WHERE id = $1',
      [id]
    );
    
    if (checkResult.rows.length === 0) {
      return res.status(404).json({
        success: false,
        error: 'Event not found'
      });
    }
    
    const result = await query(
      `UPDATE events 
       SET title = $1, description = $2, event_date = $3, location = $4, updated_at = CURRENT_TIMESTAMP
       WHERE id = $5
       RETURNING *`,
      [title, description, event_date, location, id]
    );
    
    res.json({
      success: true,
      message: 'Event updated successfully',
      data: result.rows[0]
    });
  } catch (error) {
    console.error('Error updating event:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to update event'
    });
  }
});

// Delete event by ID
router.delete('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    
    // Check if event exists
    const checkResult = await query(
      'SELECT id FROM events WHERE id = $1',
      [id]
    );
    
    if (checkResult.rows.length === 0) {
      return res.status(404).json({
        success: false,
        error: 'Event not found'
      });
    }
    
    // Delete the event
    await query('DELETE FROM events WHERE id = $1', [id]);
    
    res.json({
      success: true,
      message: 'Event deleted successfully'
    });
  } catch (error) {
    console.error('Error deleting event:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to delete event'
    });
  }
});

export default router;