// backend/routes/joinRequests.js
import express from 'express';
import pool from '../config/database.js';

const router = express.Router();

// Get all join requests for an organization
router.get('/', async (req, res) => {
  try {
    const { organizationId } = req.query;

    let query = `
      SELECT 
        jr.user_id,jr.organization_id,jr.requested_at,
        u.first_name,
        u.last_name,
        u.email,
        o.name AS organization_name
      FROM join_requests jr
      JOIN users u ON jr.user_id = u.id
      JOIN organizations o ON jr.organization_id = o.id
      LEFT JOIN departments d ON jr.requested_department_id = d.id
      LEFT JOIN roles r ON jr.requested_role_id = r.id
    `;

    const params = [];
    if (organizationId) {
      query += ' WHERE jr.organization_id = $1';
      params.push(organizationId);
    }

    query += ' ORDER BY jr.requested_at DESC';

    const result = await pool.query(query, params);

    res.json({
      success: true,
      data: result.rows,
      count: result.rows.length
    });
  } catch (error) {
    console.error('Error fetching join requests:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch join requests',
      error: error.message
    });
  }
});

// Get a single join request
router.get('/:id', async (req, res) => {
  try {
    const { id } = req.params;

    const result = await pool.query(
      `SELECT 
        jr.*,
        u.first_name,
        u.last_name,
        u.email,
        o.name AS organization_name
      FROM join_requests jr
      JOIN users u ON jr.user_id = u.id
      JOIN organizations o ON jr.organization_id = o.id
      LEFT JOIN departments d ON jr.requested_department_id = d.id
      LEFT JOIN roles r ON jr.requested_role_id = r.id
      WHERE jr.id = $1`,
      [id]
    );

    if (result.rows.length === 0) {
      return res.status(404).json({
        success: false,
        message: 'Join request not found'
      });
    }

    res.json({
      success: true,
      data: result.rows[0]
    });
  } catch (error) {
    console.error('Error fetching join request:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch join request',
      error: error.message
    });
  }
});

// Create a new join request
router.post('/', async (req, res) => {
  try {
    const {
      user_id,
      organization_id,
      requested_department_id,
      requested_role_id,
      status,
      requested_at
    } = req.body;

    // Validation
    if (!user_id || !organization_id) {
      return res.status(400).json({
        success: false,
        message: 'user_id and organization_id are required'
      });
    }

    const result = await pool.query(
      `INSERT INTO join_requests (
        user_id,
        organization_id,
        requested_department_id,
        requested_role_id,
        status,
        requested_at
      ) VALUES ($1, $2, $3, $4, $5, $6)
      RETURNING *`,
      [
        user_id,
        organization_id,
        requested_department_id || null,
        requested_role_id || null,
        status || 'pending',
        requested_at || new Date()
      ]
    );

    res.status(201).json({
      success: true,
      message: 'Join request created successfully',
      data: result.rows[0]
    });
  } catch (error) {
    console.error('Error creating join request:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to create join request',
      error: error.message
    });
  }
});

// Update join request status (approve/reject)
router.patch('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    const { status, reviewed_by } = req.body;

    if (!status || !['approved', 'rejected'].includes(status)) {
      return res.status(400).json({
        success: false,
        message: 'Valid status (approved/rejected) is required'
      });
    }

    const client = await pool.connect();

    try {
      await client.query('BEGIN');

      // Update join request
      const updateResult = await client.query(
        `UPDATE join_requests 
        SET status = $1, reviewed_at = NOW(), reviewed_by = $2
        WHERE id = $3
        RETURNING *`,
        [status, reviewed_by || null, id]
      );

      if (updateResult.rows.length === 0) {
        await client.query('ROLLBACK');
        return res.status(404).json({
          success: false,
          message: 'Join request not found'
        });
      }

      const joinRequest = updateResult.rows[0];

      // If approved, update user
      if (status === 'approved') {
        await client.query(
          `UPDATE users 
          SET 
            is_approved = TRUE,
            department_id = $1,
            role_id = $2,
            hire_date = NOW()
          WHERE id = $3`,
          [
            joinRequest.requested_department_id,
            joinRequest.requested_role_id,
            joinRequest.user_id
          ]
        );

        // Update organization employee count
        await client.query(
          `UPDATE organizations 
          SET employees_count = employees_count + 1
          WHERE id = $1`,
          [joinRequest.organization_id]
        );
      }

      await client.query('COMMIT');

      res.json({
        success: true,
        message: `Join request ${status} successfully`,
        data: joinRequest
      });
    } catch (error) {
      await client.query('ROLLBACK');
      throw error;
    } finally {
      client.release();
    }
  } catch (error) {
    console.error('Error updating join request:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to update join request',
      error: error.message
    });
  }
});

// Delete a join request
router.delete('/:id', async (req, res) => {
  try {
    const { id } = req.params;

    const result = await pool.query(
      'DELETE FROM join_requests WHERE id = $1 RETURNING *',
      [id]
    );

    if (result.rows.length === 0) {
      return res.status(404).json({
        success: false,
        message: 'Join request not found'
      });
    }

    res.json({
      success: true,
      message: 'Join request deleted successfully',
      data: result.rows[0]
    });
  } catch (error) {
    console.error('Error deleting join request:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to delete join request',
      error: error.message
    });
  }
});

// Delete a join request by user_id and organization_id
router.delete('/user/:userId/organization/:orgId', async (req, res) => {
  try {
    const { userId, orgId } = req.params;

    const result = await pool.query(
      'DELETE FROM join_requests WHERE user_id = $1 AND organization_id = $2 RETURNING *',
      [userId, orgId]
    );

    if (result.rows.length === 0) {
      return res.status(404).json({
        success: false,
        message: 'Join request not found'
      });
    }

    res.json({
      success: true,
      message: 'Join request deleted successfully',
      data: result.rows[0]
    });
  } catch (error) {
    console.error('Error deleting join request:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to delete join request',
      error: error.message
    });
  }
});

export default router;