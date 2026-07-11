-- ============================================
-- WorkHub Database - Create Tables
-- ============================================





-- Drop database if exists and create new
DROP DATABASE IF EXISTS workhub;
CREATE DATABASE workhub;

-- Connect to database
\c workhub;

-- ============================================
-- 1. ORGANIZATIONS TABLE
-- ============================================
CREATE TABLE organizations (
  id SERIAL PRIMARY KEY,
  unique_id VARCHAR(100) NOT NULL UNIQUE,
  name VARCHAR(255) NOT NULL UNIQUE,
  logo_url VARCHAR(500),
  industry VARCHAR(100) NOT NULL,
  description TEXT,
  founding_date DATE,
  website VARCHAR(255),
  linkedin_url VARCHAR(255),
  twitter_url VARCHAR(255),
  contact_email VARCHAR(255) NOT NULL,
  unique_code VARCHAR(50) NOT NULL UNIQUE,
  ceo_email VARCHAR(255),
  ceo_password_hash VARCHAR(255),
  employees_count INTEGER DEFAULT 0,
  departments_count INTEGER DEFAULT 0,
  is_active BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- 2. DEPARTMENTS TABLE
-- ============================================
CREATE TABLE departments (
  id SERIAL PRIMARY KEY,
  organization_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
  name VARCHAR(255) NOT NULL,
  description TEXT,
  manager_id INTEGER,
  
  -- Department Look Configuration
  has_header BOOLEAN DEFAULT TRUE,
  header_background_color VARCHAR(7) DEFAULT '#667eea',
  header_display_type VARCHAR(20) DEFAULT 'both',
  header_position VARCHAR(20) DEFAULT 'center',
  logo_size INTEGER DEFAULT 50,
  sidebar_position VARCHAR(20) DEFAULT 'left',
  layout_color_hover VARCHAR(7) DEFAULT '#3498db',
  layout_color_clicked VARCHAR(7) DEFAULT '#2980b9',
  layout_color_selected VARCHAR(7) DEFAULT '#667eea',
  
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- 3. ROLES TABLE
-- ============================================
CREATE TABLE roles (
  id SERIAL PRIMARY KEY,
  organization_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
  department_id INTEGER NOT NULL REFERENCES departments(id) ON DELETE CASCADE,
  name VARCHAR(100) NOT NULL,
  priority INTEGER NOT NULL DEFAULT 0,
  permissions JSONB DEFAULT '{}',
  
  -- Permission Flags
  can_accept_requests BOOLEAN DEFAULT FALSE,
  can_post_news BOOLEAN DEFAULT FALSE,
  can_post_event BOOLEAN DEFAULT FALSE,
  can_assign_tasks BOOLEAN DEFAULT FALSE,
  can_receive_tasks BOOLEAN DEFAULT FALSE,
  can_view_statistics BOOLEAN DEFAULT FALSE,
  can_hire BOOLEAN DEFAULT FALSE,
  can_reassign_tasks BOOLEAN DEFAULT FALSE,
  can_modify_department_look BOOLEAN DEFAULT FALSE,
  
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- 4. USERS TABLE
-- ============================================
CREATE TABLE users (
  id SERIAL PRIMARY KEY,
  first_name VARCHAR(255) NOT NULL,
  last_name VARCHAR(255) NOT NULL,
  email VARCHAR(255) NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL,
  avatar_color VARCHAR(7),
  organization_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
  department_id INTEGER REFERENCES departments(id) ON DELETE SET NULL,
  role_id INTEGER REFERENCES roles(id) ON DELETE SET NULL,
  is_approved BOOLEAN DEFAULT FALSE,
  is_active BOOLEAN DEFAULT TRUE,
  hire_date DATE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add foreign key for department manager after users table exists
ALTER TABLE departments 
  ADD CONSTRAINT fk_department_manager 
  FOREIGN KEY (manager_id) REFERENCES users(id) ON DELETE SET NULL;

-- ============================================
-- 5. JOIN REQUESTS TABLE
-- ============================================
CREATE TABLE join_requests (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  organization_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
  requested_department_id INTEGER REFERENCES departments(id) ON DELETE SET NULL,
  requested_role_id INTEGER REFERENCES roles(id) ON DELETE SET NULL,
  status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected')),
  requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  reviewed_at TIMESTAMP,
  reviewed_by INTEGER REFERENCES users(id) ON DELETE SET NULL
);

-- ============================================
-- 6. ASSIGNMENTS TABLE
-- ============================================
CREATE TABLE assignments (
  id SERIAL PRIMARY KEY,
  organization_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
  department_id INTEGER NOT NULL REFERENCES departments(id) ON DELETE CASCADE,
  created_by INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  title VARCHAR(500) NOT NULL,
  description TEXT,
  category VARCHAR(100),
  deadline TIMESTAMP,
  priority VARCHAR(20) DEFAULT 'medium' CHECK (priority IN ('low', 'medium', 'high')),
  status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'in_progress', 'completed')),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- 7. ASSIGNMENT USERS (Many-to-Many)
-- ============================================
CREATE TABLE assignment_users (
  id SERIAL PRIMARY KEY,
  assignment_id INTEGER NOT NULL REFERENCES assignments(id) ON DELETE CASCADE,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(assignment_id, user_id)
);

-- ============================================
-- 8. ASSIGNMENT SUBMISSIONS TABLE
-- ============================================
CREATE TABLE assignment_submissions (
  id SERIAL PRIMARY KEY,
  assignment_id INTEGER NOT NULL REFERENCES assignments(id) ON DELETE CASCADE,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  notes TEXT,
  submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- 9. ASSIGNMENT RELOCATIONS TABLE
-- ============================================
CREATE TABLE assignment_relocations (
  id SERIAL PRIMARY KEY,
  assignment_id INTEGER NOT NULL REFERENCES assignments(id) ON DELETE CASCADE,
  department_id INTEGER NOT NULL REFERENCES departments(id) ON DELETE CASCADE,
  from_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  to_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
  reason TEXT NOT NULL,
  status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected')),
  requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  reviewed_at TIMESTAMP,
  reviewed_by INTEGER REFERENCES users(id) ON DELETE SET NULL
);

-- ============================================
-- 10. NEWS POSTS TABLE
-- ============================================
CREATE TABLE news_posts (
  id SERIAL PRIMARY KEY,
  organization_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
  author_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  title VARCHAR(500) NOT NULL,
  content TEXT NOT NULL,
  image_url VARCHAR(500),
  is_public BOOLEAN DEFAULT TRUE,
  views_count INTEGER DEFAULT 0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- 11. EVENTS TABLE
-- ============================================
CREATE TABLE events (
  id SERIAL PRIMARY KEY,
  organization_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
  created_by INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  department_id INTEGER REFERENCES departments(id) ON DELETE CASCADE,
  title VARCHAR(255) NOT NULL,
  description TEXT,
  event_date TIMESTAMP NOT NULL,
  location VARCHAR(255),
  is_personal BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- 12. ATTENDANCE TABLE
-- ============================================
CREATE TABLE attendance (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  organization_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
  department_id INTEGER NOT NULL REFERENCES departments(id) ON DELETE CASCADE,
  date DATE NOT NULL,
  check_in_time TIMESTAMP NOT NULL,
  check_out_time TIMESTAMP,
  is_complete BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(user_id, date)
);

-- ============================================
-- 13. DEPARTMENT ATTENDANCE STATS TABLE
-- ============================================
CREATE TABLE department_attendance_stats (
  id SERIAL PRIMARY KEY,
  department_id INTEGER NOT NULL REFERENCES departments(id) ON DELETE CASCADE,
  month DATE NOT NULL,
  total_employees INTEGER NOT NULL,
  total_attendances INTEGER DEFAULT 0,
  attendance_percentage DECIMAL(5,2),
  last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(department_id, month)
);

-- ============================================
-- 14. NEW HIRES TRACKING TABLE
-- ============================================
CREATE TABLE new_hires_tracking (
  id SERIAL PRIMARY KEY,
  organization_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
  hires_this_month INTEGER DEFAULT 0,
  hires_this_year INTEGER DEFAULT 0,
  month DATE NOT NULL,
  year INTEGER NOT NULL,
  last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- 15. USER STATISTICS TABLE
-- ============================================
CREATE TABLE user_statistics (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
  total_assignments INTEGER DEFAULT 0,
  completed_assignments INTEGER DEFAULT 0,
  pending_assignments INTEGER DEFAULT 0,
  total_work_hours DECIMAL(10,2) DEFAULT 0,
  last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- CREATE INDEXES FOR BETTER PERFORMANCE
-- ============================================

-- Organizations
CREATE INDEX idx_organizations_industry ON organizations(industry);
CREATE INDEX idx_organizations_unique_code ON organizations(unique_code);

-- Users
CREATE INDEX idx_users_organization ON users(organization_id);
CREATE INDEX idx_users_department ON users(department_id);
CREATE INDEX idx_users_email ON users(email);

-- Departments
CREATE INDEX idx_departments_organization ON departments(organization_id);

-- Roles
CREATE INDEX idx_roles_department ON roles(department_id);
CREATE INDEX idx_roles_priority ON roles(priority);

-- Assignments
CREATE INDEX idx_assignments_organization ON assignments(organization_id);
CREATE INDEX idx_assignments_department ON assignments(department_id);
CREATE INDEX idx_assignments_status ON assignments(status);
CREATE INDEX idx_assignments_deadline ON assignments(deadline);

-- News Posts
CREATE INDEX idx_news_posts_organization ON news_posts(organization_id);
CREATE INDEX idx_news_posts_public ON news_posts(is_public);
CREATE INDEX idx_news_posts_created ON news_posts(created_at DESC);

-- Attendance
CREATE INDEX idx_attendance_user ON attendance(user_id);
CREATE INDEX idx_attendance_date ON attendance(date);
CREATE INDEX idx_attendance_department ON attendance(department_id);

-- ============================================
-- SUCCESS MESSAGE
-- ============================================
SELECT 'WorkHub database tables created successfully!' AS message;