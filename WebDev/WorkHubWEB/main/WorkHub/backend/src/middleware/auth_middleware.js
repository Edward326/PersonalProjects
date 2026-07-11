import jwt from 'jsonwebtoken';

// Secret key for JWT (in production, use environment variable)
const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key-change-in-production';

// Generate JWT token
export const generateToken = (userId, userType, organizationId = null) => {
  return jwt.sign(
    { 
      userId, 
      userType,  // 'ceo', 'employee', 'admin'
      organizationId 
    },
    JWT_SECRET,
    { expiresIn: '7d' }  // Token expires in 7 days
  );
};

// Verify JWT token middleware
export const verifyToken = (req, res, next) => {
  try {
    const token = req.headers.authorization?.split(' ')[1]; // Bearer TOKEN
    
    if (!token) {
      return res.status(401).json({
        success: false,
        message: 'Access denied. No token provided.'
      });
    }

    const decoded = jwt.verify(token, JWT_SECRET);
    req.user = decoded;  // Attach user info to request
    next();
  } catch (error) {
    return res.status(401).json({
      success: false,
      message: 'Invalid or expired token.'
    });
  }
};

// Check if user is CEO
export const isCEO = (req, res, next) => {
  if (req.user.userType !== 'ceo') {
    return res.status(403).json({
      success: false,
      message: 'Access denied. CEO privileges required.'
    });
  }
  next();
};

// Check if user belongs to organization
export const belongsToOrganization = (organizationId) => {
  return (req, res, next) => {
    if (req.user.organizationId !== organizationId) {
      return res.status(403).json({
        success: false,
        message: 'Access denied. You do not belong to this organization.'
      });
    }
    next();
  };
};

export default {
  generateToken,
  verifyToken,
  isCEO,
  belongsToOrganization
};