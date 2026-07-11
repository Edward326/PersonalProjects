import express from 'express';
import { query } from '../config/database.js';

const router = express.Router();

// Get all organizations
router.get('/', async (req, res) => {
  try {
    const result = await query(
      'SELECT * FROM organizations WHERE is_active = TRUE ORDER BY name'
    );
    
    res.json({
      success: true,
      count: result.rows.length,
      data: result.rows
    });
  } catch (error) {
    console.error('Error fetching organizations:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch organizations'
    });
  }
});

// Get organization by ID
router.get('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    
    const result = await query(
      'SELECT * FROM organizations WHERE id = $1 AND is_active = TRUE',
      [id]
    );
    
    if (result.rows.length === 0) {
      return res.status(404).json({
        success: false,
        error: 'Organization not found'
      });
    }
    
    res.json({
      success: true,
      data: result.rows[0]
    });
  } catch (error) {
    console.error('Error fetching organization:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch organization'
    });
  }
});

// Create new organization - SIMPLIFIED VERSION
router.post('/', async (req, res) => {
  try {
    console.log('Received organization data:', req.body);

    const {
      unique_id,
      unique_code,
      name,
      logo_url,
      industry,
      description,
      founding_date,
      website,
      linkedin_url,
      twitter_url,
      contact_email,
      ceo_email,
      ceo_password_hash,
      employees_count,
      departments_count,
      is_active,
      created_at,
      updated_at
    } = req.body;

    // Validation
    if (!unique_id || !unique_code || !name || !industry || !description || !website || !contact_email || !ceo_email || !ceo_password_hash) {
      console.error('Validation failed - missing fields');
      return res.status(400).json({
        success: false,
        message: 'Required fields missing',
        received: { unique_id, unique_code, name, industry, description, website, contact_email, ceo_email, has_password: !!ceo_password_hash }
      });
    }

    // Validate unique_id format: ORG-[A-Z]+-[A-Z0-9]+
    const uniqueIdRegex = /^ORG-[A-Z]+-[A-Z0-9]+$/;
    if (!uniqueIdRegex.test(unique_id)) {
      console.error('Invalid unique_id format:', unique_id);
      return res.status(400).json({
        success: false,
        message: 'Invalid unique_id format. Must be ORG-LETTERS-ALPHANUMERIC'
      });
    }

    // Check if unique_id already exists
    console.log('Checking if unique_id exists...');
    const existingId = await query(
      'SELECT id FROM organizations WHERE unique_id = $1',
      [unique_id]
    );

    if (existingId.rows.length > 0) {
      console.error('Unique ID already exists');
      return res.status(409).json({
        success: false,
        message: 'This unique ID already exists'
      });
    }

    // Check if ceo_email already exists
    console.log('Checking if CEO email exists...');
    const existingEmail = await query(
      'SELECT id FROM organizations WHERE ceo_email = $1',
      [ceo_email]
    );

    if (existingEmail.rows.length > 0) {
      console.error('CEO email already exists');
      return res.status(409).json({
        success: false,
        message: 'This CEO email is already registered'
      });
    }

    // Try to insert with basic fields first
    console.log('Attempting to insert organization...');
    
    try {
      // First, try with ALL fields
      const result = await query(
        `INSERT INTO organizations (
          unique_id,
          unique_code,
          name,
          logo_url,
          industry,
          description,
          founding_date,
          website,
          linkedin_url,
          twitter_url,
          contact_email,
          ceo_email,
          ceo_password_hash,
          employees_count,
          departments_count,
          is_active,
          created_at,
          updated_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
        RETURNING *`,
        [
          unique_id.trim(),
          unique_code.trim(),
          name.trim(),
          logo_url || null,
          industry.trim(),
          description.trim(),
          founding_date || new Date().toISOString().split('T')[0],
          website.trim(),
          linkedin_url || null,
          twitter_url || null,
          contact_email.trim(),
          ceo_email.trim(),
          ceo_password_hash,
          employees_count || 0,
          departments_count || 0,
          is_active !== undefined ? is_active : true,
          created_at || new Date().toISOString(),
          updated_at || new Date().toISOString()
        ]
      );

      console.log('Organization created successfully:', result.rows[0].id);

      res.status(201).json({
        success: true,
        message: 'Organization created successfully',
        data: result.rows[0]
      });

    } catch (insertError) {
      // If that fails, try with minimum required fields
      console.error('Full insert failed, trying minimal insert:', insertError.message);
      
      const minimalResult = await query(
        `INSERT INTO organizations (
          name,
          industry,
          description,
          website,
          contact_email,
          ceo_email,
          ceo_password_hash,
          is_active
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        RETURNING *`,
        [
          name.trim(),
          industry.trim(),
          description.trim(),
          website.trim(),
          contact_email.trim(),
          ceo_email.trim(),
          ceo_password_hash,
          true
        ]
      );

      console.log('Organization created with minimal fields:', minimalResult.rows[0].id);

      res.status(201).json({
        success: true,
        message: 'Organization created successfully (some fields omitted due to database schema)',
        data: minimalResult.rows[0]
      });
    }

  } catch (error) {
    console.error('Error creating organization:', error);
    console.error('Error stack:', error.stack);
    res.status(500).json({
      success: false,
      message: 'Failed to create organization',
      error: error.message,
      details: error.toString()
    });
  }
});

// Update organization
router.put('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    const {
      name,
      logo_url,
      industry,
      description,
      founding_date,
      website,
      linkedin_url,
      twitter_url,
      contact_email,
      ceo_email,
      ceo_password_hash,
      employees_count,
      departments_count,
      is_active,
    } = req.body;

    // Build dynamic update query
    const updates = [];
    const values = [];
    let paramCount = 1;

    if (name !== undefined) {
      updates.push(`name = $${paramCount++}`);
      values.push(name.trim());
    }

    if (logo_url !== undefined) {
      updates.push(`logo_url = $${paramCount++}`);
      values.push(logo_url);
    }

    if (industry !== undefined) {
      updates.push(`industry = $${paramCount++}`);
      values.push(industry.trim());
    }

    if (description !== undefined) {
      updates.push(`description = $${paramCount++}`);
      values.push(description.trim());
    }

    if (founding_date !== undefined) {
      updates.push(`founding_date = $${paramCount++}`);
      values.push(founding_date);
    }

    if (website !== undefined) {
      updates.push(`website = $${paramCount++}`);
      values.push(website.trim());
    }

    if (linkedin_url !== undefined) {
      updates.push(`linkedin_url = $${paramCount++}`);
      values.push(linkedin_url);
    }

    if (twitter_url !== undefined) {
      updates.push(`twitter_url = $${paramCount++}`);
      values.push(twitter_url);
    }

    if (contact_email !== undefined) {
      // Validate email format
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      if (!emailRegex.test(contact_email)) {
        return res.status(400).json({
          success: false,
          message: 'Invalid contact email format'
        });
      }
      updates.push(`contact_email = $${paramCount++}`);
      values.push(contact_email.toLowerCase().trim());
    }

    if (ceo_email !== undefined) {
      // Validate email format
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      if (!emailRegex.test(ceo_email)) {
        return res.status(400).json({
          success: false,
          message: 'Invalid CEO email format'
        });
      }

      // Check if new email already exists in another organization
      const existingEmail = await query(
        'SELECT id FROM organizations WHERE ceo_email = $1 AND id != $2',
        [ceo_email.toLowerCase().trim(), id]
      );

      if (existingEmail.rows.length > 0) {
        return res.status(409).json({
          success: false,
          message: 'This email is already registered to another organization'
        });
      }

      updates.push(`ceo_email = $${paramCount++}`);
      values.push(ceo_email.toLowerCase().trim());
    }

    if (ceo_password_hash !== undefined) {
      updates.push(`ceo_password_hash = $${paramCount++}`);
      values.push(ceo_password_hash);
    }

    if (employees_count !== undefined) {
      updates.push(`employees_count = $${paramCount++}`);
      values.push(employees_count);
    }

    if (departments_count !== undefined) {
      updates.push(`departments_count = $${paramCount++}`);
      values.push(departments_count);
    }

    if (is_active !== undefined) {
      updates.push(`is_active = $${paramCount++}`);
      values.push(is_active);
    }

    if (updates.length === 0) {
      return res.status(400).json({
        success: false,
        message: 'No fields to update'
      });
    }

    // Always update the updated_at timestamp
    updates.push(`updated_at = CURRENT_TIMESTAMP`);

    values.push(id);
    const updateQuery = `
      UPDATE organizations 
      SET ${updates.join(', ')}
      WHERE id = $${paramCount}
      RETURNING *
    `;

    const result = await query(updateQuery, values);

    if (result.rows.length === 0) {
      return res.status(404).json({
        success: false,
        message: 'Organization not found'
      });
    }

    console.log('Organization updated successfully:', result.rows[0].id);

    res.json({
      success: true,
      message: 'Organization updated successfully',
      data: result.rows[0]
    });

  } catch (error) {
    console.error('Error updating organization:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to update organization',
      error: error.message
    });
  }
});

// Get organization statistics
router.get('/:id/statistics', async (req, res) => {
  try {
    const { id } = req.params;
    
    // Get various statistics for the organization
    const [orgData, deptCount, empCount, newsCount, assignmentCount] = await Promise.all([
      query('SELECT * FROM organizations WHERE id = $1', [id]),
      query('SELECT COUNT(*) as count FROM departments WHERE organization_id = $1', [id]),
      query('SELECT COUNT(*) as count FROM users WHERE organization_id = $1 AND is_approved = TRUE', [id]),
      query('SELECT COUNT(*) as count FROM news_posts WHERE organization_id = $1', [id]),
      query('SELECT COUNT(*) as count FROM assignments WHERE organization_id = $1', [id])
    ]);
    
    if (orgData.rows.length === 0) {
      return res.status(404).json({
        success: false,
        error: 'Organization not found'
      });
    }
    
    res.json({
      success: true,
      data: {
        organization: orgData.rows[0],
        statistics: {
          departments: parseInt(deptCount.rows[0].count),
          employees: parseInt(empCount.rows[0].count),
          news_posts: parseInt(newsCount.rows[0].count),
          assignments: parseInt(assignmentCount.rows[0].count)
        }
      }
    });
  } catch (error) {
    console.error('Error fetching organization statistics:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch organization statistics'
    });
  }
});

export default router;