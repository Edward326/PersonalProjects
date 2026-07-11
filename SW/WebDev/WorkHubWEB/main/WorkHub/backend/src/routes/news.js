import express from 'express';
import { query } from '../config/database.js';

const router = express.Router();

// Get all public news (for WelcomePage)
router.get('/', async (req, res) => {
  try {
    const { limit = 10, offset = 0 } = req.query;
    
    const result = await query(
      `SELECT 
        n.id,
        n.title,
        n.content,
        n.image_url,
        n.views_count,
        n.created_at,
        n.updated_at,
        u.first_name,
        u.last_name,
        u.avatar_color,
        o.id as company_id,
        o.name as company_name,
        o.industry as company_industry
      FROM news_posts n
      JOIN users u ON n.author_id = u.id
      JOIN organizations o ON n.organization_id = o.id
      WHERE n.is_public = TRUE
      ORDER BY n.created_at DESC
      LIMIT $1 OFFSET $2`,
      [limit, offset]
    );
    
    // Format the response
    const formattedNews = result.rows.map(row => ({
      id: row.id,
      title: row.title,
      content: row.content,
      image: row.image_url,
      date: row.created_at,
      views: row.views_count,
      author: {
        name: `${row.first_name} ${row.last_name}`,
        profileIcon: row.first_name.charAt(0),
        avatarColor: row.avatar_color,
        company: row.company_name,
        companyId: row.company_id
      },
      company: {
        id: row.company_id,
        name: row.company_name,
        industry: row.company_industry
      }
    }));
    
    res.json({
      success: true,
      count: formattedNews.length,
      data: formattedNews
    });
  } catch (error) {
    console.error('Error fetching public news:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch news'
    });
  }
});

// Get news by organization
router.get('/organization/:orgId', async (req, res) => {
  try {
    const { orgId } = req.params;
    const { limit = 10, offset = 0 } = req.query;
    
    const result = await query(
      `SELECT 
        n.*,
        u.first_name,
        u.last_name,
        u.avatar_color
      FROM news_posts n
      JOIN users u ON n.author_id = u.id
      WHERE n.organization_id = $1
      ORDER BY n.created_at DESC
      LIMIT $2 OFFSET $3`,
      [orgId, limit, offset]
    );
    
    res.json({
      success: true,
      count: result.rows.length,
      data: result.rows
    });
  } catch (error) {
    console.error('Error fetching organization news:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch organization news'
    });
  }
});

// *** NEW: Get news by author ***
router.get('/author/:authorId', async (req, res) => {
  try {
    const { authorId } = req.params;
    
    const result = await query(
      `SELECT 
        n.*,
        u.first_name,
        u.last_name,
        u.avatar_color,
        o.name as organization_name
      FROM news_posts n
      JOIN users u ON n.author_id = u.id
      JOIN organizations o ON n.organization_id = o.id
      WHERE n.author_id = $1
      ORDER BY n.created_at DESC`,
      [authorId]
    );
    
    res.json({
      success: true,
      count: result.rows.length,
      data: result.rows
    });
  } catch (error) {
    console.error('Error fetching author news:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch author news'
    });
  }
});

// Get single news post by ID
router.get('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    
    const result = await query(
      `SELECT 
        n.*,
        u.first_name,
        u.last_name,
        u.avatar_color,
        o.name as company_name,
        o.industry as company_industry
      FROM news_posts n
      JOIN users u ON n.author_id = u.id
      JOIN organizations o ON n.organization_id = o.id
      WHERE n.id = $1`,
      [id]
    );
    
    if (result.rows.length === 0) {
      return res.status(404).json({
        success: false,
        error: 'News post not found'
      });
    }
    
    // Increment view count
    await query(
      'UPDATE news_posts SET views_count = views_count + 1 WHERE id = $1',
      [id]
    );
    
    res.json({
      success: true,
      data: result.rows[0]
    });
  } catch (error) {
    console.error('Error fetching news post:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch news post'
    });
  }
});

// Create news post
router.post('/', async (req, res) => {
  try {
    const { organization_id, author_id, title, content, image_url, is_public = true } = req.body;
    
    if (!organization_id || !author_id || !title || !content) {
      return res.status(400).json({
        success: false,
        error: 'Missing required fields'
      });
    }
    
    const result = await query(
      `INSERT INTO news_posts (organization_id, author_id, title, content, image_url, is_public)
       VALUES ($1, $2, $3, $4, $5, $6)
       RETURNING *`,
      [organization_id, author_id, title, content, image_url, is_public]
    );
    
    res.status(201).json({
      success: true,
      message: 'News post created successfully',
      data: result.rows[0]
    });
  } catch (error) {
    console.error('Error creating news post:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to create news post'
    });
  }
});

// Delete news post by ID
router.delete('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    
    // Check if news exists
    const checkResult = await query(
      'SELECT id FROM news_posts WHERE id = $1',
      [id]
    );
    
    if (checkResult.rows.length === 0) {
      return res.status(404).json({
        success: false,
        error: 'News post not found'
      });
    }
    
    // Delete the news post
    await query('DELETE FROM news_posts WHERE id = $1', [id]);
    
    res.json({
      success: true,
      message: 'News post deleted successfully'
    });
  } catch (error) {
    console.error('Error deleting news post:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to delete news post'
    });
  }
});

export default router;