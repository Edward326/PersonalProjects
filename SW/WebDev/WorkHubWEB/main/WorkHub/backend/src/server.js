import express from 'express';
import cors from 'cors';
import morgan from 'morgan';
import dotenv from 'dotenv';

// Import routes
import organizationRoutes from './routes/organizations.js';
import newsRoutes from './routes/news.js';
import departmentRoutes from './routes/departments.js';
import userRoutes from './routes/users.js';
import roleRoutes from './routes/roles.js';
import assignmentRoutes from './routes/assignments.js';
import eventRoutes from './routes/events.js';
import attendanceRoutes from './routes/attendance.js';
import statisticsRoutes from './routes/statistics.js';
import authRoutes from './routes/auth.js';
import joinRequestRoutes from './routes/joinRequests.js';
import assignmentRelocationRoutes from './routes/assignmentRelocations.js';
import assignmentUsersRoutes from './routes/assignmentUsers.js';
import assignmentSubmissionsRoutes from './routes/assignmentSubmissions.js';
import departmentAttendanceStatsRoutes from './routes/departmentAttendanceStats.js';
import assignmentOperationsRouter from './routes/assignmentOperations.js';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// ============================================
// MIDDLEWARE
// ============================================

// CORS configuration
app.use(cors({
  origin: process.env.FRONTEND_URL || 'http://localhost:5173',
  credentials: true
}));

// Body parser
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Logging
if (process.env.NODE_ENV === 'development') {
  app.use(morgan('dev'));
}

// ============================================
// ROUTES
// ============================================

// Health check
app.get('/api/health', (req, res) => {
  res.json({
    status: 'OK',
    message: 'WorkHub API is running',
    timestamp: new Date().toISOString()
  });
});

// API Routes
app.use('/api/organizations', organizationRoutes);
app.use('/api/news', newsRoutes);
app.use('/api/departments', departmentRoutes);
app.use('/api/users', userRoutes);
app.use('/api/roles', roleRoutes);
app.use('/api/assignments', assignmentRoutes);
app.use('/api/events', eventRoutes);
app.use('/api/attendance', attendanceRoutes);
app.use('/api/statistics', statisticsRoutes);
app.use('/api/auth', authRoutes);
app.use('/api/join-requests', joinRequestRoutes);
app.use('/api/assignment-relocations', assignmentRelocationRoutes);
app.use('/api/assignment-users', assignmentUsersRoutes);
app.use('/api/assignment-submissions', assignmentSubmissionsRoutes);
app.use('/api/attendance', attendanceRoutes);
app.use('/api/department-attendance-stats', departmentAttendanceStatsRoutes);
app.use('/api/assignment-operations', assignmentOperationsRouter);

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    error: 'Not Found',
    message: `Route ${req.originalUrl} not found`
  });
});

// Global error handler
app.use((err, req, res, next) => {
  console.error('Error:', err);
  res.status(err.status || 500).json({
    error: err.message || 'Internal Server Error',
    ...(process.env.NODE_ENV === 'development' && { stack: err.stack })
  });
});

// ============================================
// START SERVER
// ============================================

app.listen(PORT, () => {
  console.log(`\n🚀 WorkHub Backend Server`);
  console.log(`📡 Server running on http://localhost:${PORT}`);
  console.log(`🌍 Environment: ${process.env.NODE_ENV}`);
  console.log(`📝 API Documentation: http://localhost:${PORT}/api/health\n`);
});