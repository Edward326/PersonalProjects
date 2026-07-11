-- ============================================
-- WorkHub Database - Complete Data Insert
-- Contains realistic data for all tables
-- ============================================




-- ============================================
-- 1: INSERT DATA
-- ============================================
\c workhub;

-- ============================================
-- STEP 1: CLEAN EXISTING DATA
-- ============================================

-- Delete all data (cascade will handle related tables)
TRUNCATE TABLE 
  assignment_relocations,
  assignment_submissions,
  assignment_users,
  assignments,
  join_requests,
  events,
  news_posts,
  attendance,
  department_attendance_stats,
  new_hires_tracking,
  user_statistics,
  users,
  roles,
  departments,
  organizations
CASCADE;

-- Reset all sequences
ALTER SEQUENCE organizations_id_seq RESTART WITH 1;
ALTER SEQUENCE departments_id_seq RESTART WITH 1;
ALTER SEQUENCE roles_id_seq RESTART WITH 1;
ALTER SEQUENCE users_id_seq RESTART WITH 1;
ALTER SEQUENCE assignments_id_seq RESTART WITH 1;
ALTER SEQUENCE assignment_users_id_seq RESTART WITH 1;
ALTER SEQUENCE assignment_submissions_id_seq RESTART WITH 1;
ALTER SEQUENCE assignment_relocations_id_seq RESTART WITH 1;
ALTER SEQUENCE news_posts_id_seq RESTART WITH 1;
ALTER SEQUENCE events_id_seq RESTART WITH 1;
ALTER SEQUENCE attendance_id_seq RESTART WITH 1;
ALTER SEQUENCE department_attendance_stats_id_seq RESTART WITH 1;
ALTER SEQUENCE new_hires_tracking_id_seq RESTART WITH 1;
ALTER SEQUENCE user_statistics_id_seq RESTART WITH 1;
ALTER SEQUENCE join_requests_id_seq RESTART WITH 1;

-- ============================================
-- STEP 2: INSERT ORGANIZATIONS (10 Companies)
-- ============================================

INSERT INTO organizations (unique_id, name, logo_url, industry, description, founding_date, website, linkedin_url, twitter_url, contact_email, unique_code, ceo_email, ceo_password_hash, employees_count, departments_count, is_active) VALUES

-- 1. Technology - AI Company
('ORG-TECH-A7K4M', 'NeuralTech AI', 'https://logo.clearbit.com/openai.com', 'Technology', 'NeuralTech AI is a pioneering artificial intelligence company specializing in machine learning solutions, natural language processing, and computer vision technologies. We help businesses transform their operations through intelligent automation and predictive analytics.', '2018-03-15', 'https://neuraltech.ai', 'https://linkedin.com/company/neuraltech', 'https://twitter.com/neuraltech', 'contact@neuraltech.com', 'TECH-A7K4M', 'ceo.neuraltech@gmail.com', 'neural123', 0, 0, TRUE),

-- 2. Finance - Investment Firm
('ORG-FIN-B3X9Q', 'Quantum Capital Partners', 'https://logo.clearbit.com/goldmansachs.com', 'Finance', 'Quantum Capital Partners is a premier investment management firm providing sophisticated financial solutions to institutional clients and high-net-worth individuals. With over $50 billion in assets under management, we specialize in alternative investments and private equity.', '2012-07-22', 'https://quantumcapital.com', 'https://linkedin.com/company/quantum-capital', NULL, 'info@quantumcapital.com', 'FIN-B3X9Q', 'ceo.quantum@gmail.com', 'quantum123', 0, 0, TRUE),

-- 3. Healthcare - Digital Health Platform
('ORG-HCR-C5P2W', 'MediConnect Health', 'https://logo.clearbit.com/unitedhealthgroup.com', 'Healthcare', 'MediConnect Health is revolutionizing healthcare delivery through our comprehensive digital health platform. We connect patients, providers, and payers in a seamless ecosystem that improves health outcomes while reducing costs.', '2016-01-10', 'https://mediconnect.health', 'https://linkedin.com/company/mediconnect', 'https://twitter.com/mediconnect', 'support@mediconnect.com', 'HCR-C5P2W', 'ceo.mediconnect@gmail.com', 'medi123', 0, 0, TRUE),

-- 4. E-commerce - Retail Platform
('ORG-RET-D8N6T', 'ShopFlow Commerce', 'https://logo.clearbit.com/shopify.com', 'Retail', 'ShopFlow Commerce empowers retailers with an all-in-one e-commerce platform that scales from startups to enterprises. Our cloud-based solution includes inventory management, multi-channel selling, and payment processing.', '2015-05-20', 'https://shopflow.com', NULL, 'https://twitter.com/shopflow', 'hello@shopflow.com', 'RET-D8N6T', 'ceo.shopflow@gmail.com', 'shop123', 0, 0, TRUE),

-- 5. SaaS - Cloud Infrastructure
('ORG-TECH-E4R7L', 'CloudSphere Technologies', 'https://logo.clearbit.com/aws.amazon.com', 'Technology', 'CloudSphere Technologies provides enterprise-grade cloud infrastructure and DevOps solutions that enable businesses to build, deploy, and scale applications with confidence. Our platform offers automated deployment pipelines and container orchestration.', '2017-11-08', 'https://cloudsphere.io', 'https://linkedin.com/company/cloudsphere', 'https://twitter.com/cloudsphere', 'team@cloudsphere.com', 'TECH-E4R7L', 'ceo.cloudsphere@gmail.com', 'cloud123', 0, 0, TRUE),

-- 6. Sustainability - Green Technology
('ORG-SUS-F9M3K', 'EcoVision Solutions', 'https://logo.clearbit.com/tesla.com', 'Sustainability', 'EcoVision Solutions is dedicated to creating a sustainable future through innovative green technology. We specialize in renewable energy systems, carbon footprint tracking, and sustainable supply chain management.', '2014-02-14', 'https://ecovision.green', 'https://linkedin.com/company/ecovision', NULL, 'contact@ecovision.com', 'SUS-F9M3K', 'ceo.ecovision@gmail.com', 'eco123', 0, 0, TRUE),

-- 7. Marketing - Digital Agency
('ORG-MKT-G1H8V', 'BrandBoost Digital', NULL, 'Marketing', 'BrandBoost Digital is a full-service digital marketing agency that helps brands connect with their audiences through data-driven strategies and creative excellence. Our services include SEO, social media marketing, and content creation.', '2013-04-20', 'https://brandboost.digital', NULL, 'https://twitter.com/brandboost', 'hello@brandboost.com', 'MKT-G1H8V', 'ceo.brandboost@gmail.com', 'brand123', 0, 0, TRUE),

-- 8. Logistics - Supply Chain Management
('ORG-LOG-H6Y2P', 'GlobalFlow Logistics', 'https://logo.clearbit.com/fedex.com', 'Logistics', 'GlobalFlow Logistics is a technology-driven logistics company providing end-to-end supply chain solutions for global businesses. We operate a network spanning 150 countries with real-time tracking and predictive analytics.', '2011-09-10', 'https://globalflow.logistics', 'https://linkedin.com/company/globalflow', 'https://twitter.com/globalflow', 'support@globalflow.com', 'LOG-H6Y2P', 'ceo.globalflow@gmail.com', 'global123', 0, 0, TRUE),

-- 9. Cybersecurity - Security Solutions
('ORG-SEC-J3Z5N', 'ShieldGuard Security', 'https://logo.clearbit.com/crowdstrike.com', 'Security', 'ShieldGuard Security is a leading cybersecurity firm protecting organizations from advanced cyber threats. Our comprehensive security platform includes threat detection, incident response, and penetration testing.', '2010-11-15', 'https://shieldguard.security', 'https://linkedin.com/company/shieldguard', 'https://twitter.com/shieldguard', 'info@shieldguard.com', 'SEC-J3Z5N', 'ceo.shieldguard@gmail.com', 'shield123', 0, 0, TRUE),

-- 10. Education - EdTech Platform
('ORG-EDU-K7W4R', 'LearnSphere EdTech', 'https://logo.clearbit.com/coursera.org', 'Education', 'LearnSphere EdTech is transforming education through technology-enabled learning experiences. Our platform provides interactive courses, virtual classrooms, and AI-powered personalized learning paths for students worldwide.', '2019-08-25', 'https://learnsphere.edu', 'https://linkedin.com/company/learnsphere', NULL, 'hello@learnsphere.com', 'EDU-K7W4R', 'ceo.learnsphere@gmail.com', 'learn123', 0, 0, TRUE);


-- ============================================
-- STEP 3: INSERT DEPARTMENTS
-- ============================================

-- NeuralTech AI Departments (Org 1)
INSERT INTO departments (organization_id, name, description, has_header, header_background_color, header_display_type, header_position, sidebar_position, layout_color_hover, layout_color_clicked, layout_color_selected) VALUES
(1, 'Machine Learning Research', 'Core ML research and algorithm development', TRUE, '#667eea', 'both', 'center', 'left', '#3498db', '#2980b9', '#667eea'),
(1, 'Engineering', 'Platform development and infrastructure', TRUE, '#764ba2', 'both', 'center', 'left', '#3498db', '#2980b9', '#764ba2'),
(1, 'Product Management', 'Product strategy and customer success', TRUE, '#27ae60', 'both', 'center', 'left', '#229954', '#1e8449', '#27ae60'),
(1, 'Sales & Business Development', 'Client acquisition and partnerships', TRUE, '#e74c3c', 'both', 'center', 'left', '#c0392b', '#a93226', '#e74c3c'),
(1, 'Human Resources', 'Talent and organizational development', TRUE, '#f39c12', 'both', 'center', 'left', '#e67e22', '#d68910', '#f39c12');

-- Quantum Capital Partners Departments (Org 2)
INSERT INTO departments (organization_id, name, description, has_header, header_background_color) VALUES
(2, 'Investment Banking', 'M&A advisory and capital markets', TRUE, '#3498db'),
(2, 'Private Equity', 'Private investment and portfolio management', TRUE, '#2ecc71'),
(2, 'Quantitative Trading', 'Algorithmic trading and risk management', TRUE, '#9b59b6'),
(2, 'Operations & Compliance', 'Back office and regulatory compliance', TRUE, '#95a5a6');

-- MediConnect Health Departments (Org 3)
INSERT INTO departments (organization_id, name, description, has_header, header_background_color) VALUES
(3, 'Clinical Operations', 'Healthcare delivery and patient care', TRUE, '#27ae60'),
(3, 'Technology & Product', 'Platform development and innovation', TRUE, '#3498db'),
(3, 'Medical Affairs', 'Clinical research and medical oversight', TRUE, '#9b59b6'),
(3, 'Customer Success', 'Patient support and provider relations', TRUE, '#e67e22'),
(3, 'Data Science', 'Healthcare analytics and AI', TRUE, '#34495e');

-- ShopFlow Commerce Departments (Org 4)
INSERT INTO departments (organization_id, name, description) VALUES
(4, 'Engineering', 'Platform and infrastructure development'),
(4, 'Product Design', 'User experience and interface design'),
(4, 'Customer Success', 'Merchant support and onboarding'),
(4, 'Sales & Partnerships', 'Business development and strategic alliances');

-- CloudSphere Technologies Departments (Org 5)
INSERT INTO departments (organization_id, name, description) VALUES
(5, 'Cloud Engineering', 'Infrastructure and platform services'),
(5, 'DevOps & Automation', 'CI/CD and deployment automation'),
(5, 'Security & Compliance', 'Information security and governance'),
(5, 'Solutions Architecture', 'Enterprise solutions and consulting'),
(5, 'Sales & Marketing', 'Go-to-market and demand generation');

-- EcoVision Solutions Departments (Org 6)
INSERT INTO departments (organization_id, name, description) VALUES
(6, 'Renewable Energy', 'Solar and wind energy solutions'),
(6, 'Sustainability Consulting', 'Carbon reduction and ESG advisory'),
(6, 'Technology & Engineering', 'Clean tech R&D and implementation'),
(6, 'Operations', 'Project delivery and client success');

-- BrandBoost Digital Departments (Org 7)
INSERT INTO departments (organization_id, name, description) VALUES
(7, 'Strategy & Planning', 'Marketing strategy and campaign planning'),
(7, 'Creative & Content', 'Design, copy, and content production'),
(7, 'Digital Marketing', 'SEO, SEM, and social media management'),
(7, 'Analytics & Insights', 'Performance tracking and optimization'),
(7, 'Client Services', 'Account management and client relations');

-- GlobalFlow Logistics Departments (Org 8)
INSERT INTO departments (organization_id, name, description) VALUES
(8, 'Operations', 'Warehouse and transportation management'),
(8, 'Technology', 'Logistics platform and automation'),
(8, 'Supply Chain Solutions', 'Client services and optimization'),
(8, 'Regional Management', 'Geographic operations oversight');

-- ShieldGuard Security Departments (Org 9)
INSERT INTO departments (organization_id, name, description) VALUES
(9, 'Security Operations Center', 'Threat monitoring and incident response'),
(9, 'Penetration Testing', 'Ethical hacking and vulnerability assessment'),
(9, 'Security Consulting', 'Risk assessment and strategy advisory'),
(9, 'Research & Development', 'Security tools and threat intelligence');

-- LearnSphere EdTech Departments (Org 10)
INSERT INTO departments (organization_id, name, description) VALUES
(10, 'Content Development', 'Course creation and curriculum design'),
(10, 'Platform Engineering', 'Learning management system development'),
(10, 'Student Success', 'Learner support and engagement'),
(10, 'Partnerships', 'University and corporate partnerships');


-- ============================================
-- STEP 4: INSERT ROLES
-- ============================================
-- 1-17
-- NeuralTech AI - Machine Learning Research (Dept 1)
INSERT INTO roles (organization_id, department_id, name, priority, can_accept_requests, can_post_news, can_assign_tasks, can_receive_tasks, can_view_statistics, can_hire, can_reassign_tasks) VALUES
(1, 1, 'Research Director', 0, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE),
(1, 1, 'Senior ML Researcher', 1, FALSE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE),
(1, 1, 'ML Researcher', 2, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE),
(1, 1, 'Research Engineer', 3, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE);

-- NeuralTech AI - Engineering (Dept 2)
INSERT INTO roles (organization_id, department_id, name, priority, can_assign_tasks, can_receive_tasks, can_view_statistics) VALUES
(1, 2, 'VP of Engineering', 0, TRUE, TRUE, TRUE),
(1, 2, 'Engineering Manager', 1, TRUE, TRUE, TRUE),
(1, 2, 'Senior Engineer', 2, FALSE, TRUE, FALSE),
(1, 2, 'Software Engineer', 3, FALSE, TRUE, FALSE);

-- NeuralTech AI - Product Management (Dept 3)
INSERT INTO roles (organization_id, department_id, name, priority, can_post_news, can_assign_tasks, can_receive_tasks) VALUES
(1, 3, 'Head of Product', 0, TRUE, TRUE, TRUE),
(1, 3, 'Product Manager', 1, TRUE, TRUE, TRUE),
(1, 3, 'Product Analyst', 2, FALSE, FALSE, TRUE);

-- NeuralTech AI - Sales & BD (Dept 4)
INSERT INTO roles (organization_id, department_id, name, priority, can_post_news, can_receive_tasks) VALUES
(1, 4, 'VP Sales', 0, TRUE, TRUE),
(1, 4, 'Enterprise Account Executive', 1, FALSE, TRUE),
(1, 4, 'Business Development Manager', 2, FALSE, TRUE);

-- NeuralTech AI - HR (Dept 5)
INSERT INTO roles (organization_id, department_id, name, priority, can_accept_requests, can_hire, can_view_statistics) VALUES
(1, 5, 'Chief People Officer', 0, TRUE, TRUE, TRUE),
(1, 5, 'HR Manager', 1, TRUE, TRUE, FALSE),
(1, 5, 'HR Specialist', 2, FALSE, FALSE, FALSE);

-- 18-30
-- Quantum Capital - Investment Banking (Dept 6)
INSERT INTO roles (organization_id, department_id, name, priority, can_assign_tasks, can_receive_tasks, can_view_statistics) VALUES
(2, 6, 'Managing Director', 0, TRUE, TRUE, TRUE),
(2, 6, 'Vice President', 1, TRUE, TRUE, FALSE),
(2, 6, 'Associate', 2, FALSE, TRUE, FALSE),
(2, 6, 'Analyst', 3, FALSE, TRUE, FALSE);

-- Quantum Capital - Private Equity (Dept 7)
INSERT INTO roles (organization_id, department_id, name, priority, can_assign_tasks, can_receive_tasks) VALUES
(2, 7, 'Partner', 0, TRUE, TRUE),
(2, 7, 'Principal', 1, TRUE, TRUE),
(2, 7, 'Associate', 2, FALSE, TRUE);

-- Quantum Capital - Quant Trading (Dept 8)
INSERT INTO roles (organization_id, department_id, name, priority, can_receive_tasks) VALUES
(2, 8, 'Quantitative Researcher', 0, TRUE),
(2, 8, 'Quantitative Developer', 1, TRUE),
(2, 8, 'Trading Analyst', 2, TRUE);

-- Quantum Capital - Operations (Dept 9)
INSERT INTO roles (organization_id, department_id, name, priority, can_accept_requests) VALUES
(2, 9, 'Operations Manager', 0, TRUE),
(2, 9, 'Compliance Officer', 1, FALSE),
(2, 9, 'Operations Analyst', 2, FALSE);

-- 31-48
-- MediConnect - Clinical Operations (Dept 10)
INSERT INTO roles (organization_id, department_id, name, priority, can_post_news, can_receive_tasks) VALUES
(3, 10, 'Chief Medical Officer', 0, TRUE, TRUE),
(3, 10, 'Clinical Director', 1, TRUE, TRUE),
(3, 10, 'Physician', 2, FALSE, TRUE),
(3, 10, 'Nurse Practitioner', 3, FALSE, TRUE);

-- MediConnect - Technology & Product (Dept 11)
INSERT INTO roles (organization_id, department_id, name, priority, can_assign_tasks, can_receive_tasks) VALUES
(3, 11, 'CTO', 0, TRUE, TRUE),
(3, 11, 'Engineering Manager', 1, TRUE, TRUE),
(3, 11, 'Senior Engineer', 2, FALSE, TRUE),
(3, 11, 'Product Manager', 1, TRUE, TRUE);

-- MediConnect - Medical Affairs (Dept 12)
INSERT INTO roles (organization_id, department_id, name, priority, can_post_news) VALUES
(3, 12, 'Medical Director', 0, TRUE),
(3, 12, 'Clinical Research Manager', 1, FALSE),
(3, 12, 'Medical Science Liaison', 2, FALSE);

-- MediConnect - Customer Success (Dept 13)
INSERT INTO roles (organization_id, department_id, name, priority) VALUES
(3, 13, 'Head of Customer Success', 0),
(3, 13, 'Customer Success Manager', 1),
(3, 13, 'Support Specialist', 2);

-- MediConnect - Data Science (Dept 14)
INSERT INTO roles (organization_id, department_id, name, priority, can_receive_tasks) VALUES
(3, 14, 'Head of Data Science', 0, TRUE),
(3, 14, 'Data Scientist', 1, TRUE),
(3, 14, 'Data Analyst', 2, TRUE);

-- 49-61
-- ShopFlow - Engineering (Dept 15)
INSERT INTO roles (organization_id, department_id, name, priority, can_assign_tasks, can_receive_tasks) VALUES
(4, 15, 'Head of Engineering', 0, TRUE, TRUE),
(4, 15, 'Tech Lead', 1, TRUE, TRUE),
(4, 15, 'Senior Developer', 2, FALSE, TRUE),
(4, 15, 'Developer', 3, FALSE, TRUE);

-- ShopFlow - Product Design (Dept 16)
INSERT INTO roles (organization_id, department_id, name, priority, can_receive_tasks) VALUES
(4, 16, 'Design Director', 0, TRUE),
(4, 16, 'Senior Designer', 1, TRUE),
(4, 16, 'UX Designer', 2, TRUE);

-- ShopFlow - Customer Success (Dept 17)
INSERT INTO roles (organization_id, department_id, name, priority) VALUES
(4, 17, 'CS Director', 0),
(4, 17, 'Account Manager', 1),
(4, 17, 'Support Engineer', 2);

-- ShopFlow - Sales & Partnerships (Dept 18)
INSERT INTO roles (organization_id, department_id, name, priority, can_post_news) VALUES
(4, 18, 'VP Partnerships', 0, TRUE),
(4, 18, 'Partnership Manager', 1, FALSE),
(4, 18, 'Sales Executive', 2, FALSE);

-- 62-77
-- CloudSphere - Cloud Engineering (Dept 19)
INSERT INTO roles (organization_id, department_id, name, priority, can_assign_tasks, can_receive_tasks) VALUES
(5, 19, 'VP Engineering', 0, TRUE, TRUE),
(5, 19, 'Principal Engineer', 1, TRUE, TRUE),
(5, 19, 'Senior Cloud Engineer', 2, FALSE, TRUE),
(5, 19, 'Cloud Engineer', 3, FALSE, TRUE);

-- CloudSphere - DevOps (Dept 20)
INSERT INTO roles (organization_id, department_id, name, priority, can_receive_tasks) VALUES
(5, 20, 'DevOps Lead', 0, TRUE),
(5, 20, 'DevOps Engineer', 1, TRUE),
(5, 20, 'Automation Engineer', 2, TRUE);

-- CloudSphere - Security (Dept 21)
INSERT INTO roles (organization_id, department_id, name, priority, can_receive_tasks, can_view_statistics) VALUES
(5, 21, 'Security Director', 0, TRUE, TRUE),
(5, 21, 'Security Engineer', 1, TRUE, FALSE),
(5, 21, 'Security Analyst', 2, TRUE, FALSE);

-- CloudSphere - Solutions Architecture (Dept 22)
INSERT INTO roles (organization_id, department_id, name, priority) VALUES
(5, 22, 'Chief Architect', 0),
(5, 22, 'Solutions Architect', 1),
(5, 22, 'Technical Consultant', 2);

-- CloudSphere - Sales & Marketing (Dept 23)
INSERT INTO roles (organization_id, department_id, name, priority, can_post_news) VALUES
(5, 23, 'CMO', 0, TRUE),
(5, 23, 'Enterprise Sales Director', 1, TRUE),
(5, 23, 'Marketing Manager', 2, FALSE);

-- 78-89
-- EcoVision - Renewable Energy (Dept 24)
INSERT INTO roles (organization_id, department_id, name, priority, can_assign_tasks, can_receive_tasks) VALUES
(6, 24, 'Director of Energy Solutions', 0, TRUE, TRUE),
(6, 24, 'Energy Engineer', 1, FALSE, TRUE),
(6, 24, 'Project Manager', 2, TRUE, TRUE);

-- EcoVision - Sustainability Consulting (Dept 25)
INSERT INTO roles (organization_id, department_id, name, priority) VALUES
(6, 25, 'Head of Consulting', 0),
(6, 25, 'Sustainability Consultant', 1),
(6, 25, 'ESG Analyst', 2);

-- EcoVision - Technology & Engineering (Dept 26)
INSERT INTO roles (organization_id, department_id, name, priority, can_receive_tasks) VALUES
(6, 26, 'CTO', 0, TRUE),
(6, 26, 'R&D Engineer', 1, TRUE),
(6, 26, 'Systems Engineer', 2, TRUE);

-- EcoVision - Operations (Dept 27)
INSERT INTO roles (organization_id, department_id, name, priority) VALUES
(6, 27, 'Operations Director', 0),
(6, 27, 'Project Coordinator', 1),
(6, 27, 'Operations Specialist', 2);

-- 90-105
-- BrandBoost - Strategy & Planning (Dept 28)
INSERT INTO roles (organization_id, department_id, name, priority, can_post_news, can_assign_tasks) VALUES
(7, 28, 'Strategy Director', 0, TRUE, TRUE),
(7, 28, 'Senior Strategist', 1, FALSE, TRUE),
(7, 28, 'Marketing Strategist', 2, FALSE, FALSE);

-- BrandBoost - Creative & Content (Dept 29)
INSERT INTO roles (organization_id, department_id, name, priority, can_receive_tasks) VALUES
(7, 29, 'Creative Director', 0, TRUE),
(7, 29, 'Senior Designer', 1, TRUE),
(7, 29, 'Content Creator', 2, TRUE),
(7, 29, 'Copywriter', 3, TRUE);

-- BrandBoost - Digital Marketing (Dept 30)
INSERT INTO roles (organization_id, department_id, name, priority, can_receive_tasks) VALUES
(7, 30, 'Digital Marketing Director', 0, TRUE),
(7, 30, 'SEO Specialist', 1, TRUE),
(7, 30, 'Social Media Manager', 2, TRUE);

-- BrandBoost - Analytics & Insights (Dept 31)
INSERT INTO roles (organization_id, department_id, name, priority, can_view_statistics) VALUES
(7, 31, 'Analytics Director', 0, TRUE),
(7, 31, 'Data Analyst', 1, FALSE),
(7, 31, 'Performance Analyst', 2, FALSE);

-- BrandBoost - Client Services (Dept 32)
INSERT INTO roles (organization_id, department_id, name, priority) VALUES
(7, 32, 'Client Services Director', 0),
(7, 32, 'Account Director', 1),
(7, 32, 'Account Manager', 2);

-- 106-119
-- GlobalFlow - Operations (Dept 33)
INSERT INTO roles (organization_id, department_id, name, priority, can_assign_tasks, can_receive_tasks) VALUES
(8, 33, 'Operations Director', 0, TRUE, TRUE),
(8, 33, 'Warehouse Manager', 1, TRUE, TRUE),
(8, 33, 'Operations Supervisor', 2, FALSE, TRUE),
(8, 33, 'Logistics Coordinator', 3, FALSE, TRUE);

-- GlobalFlow - Technology (Dept 34)
INSERT INTO roles (organization_id, department_id, name, priority, can_receive_tasks) VALUES
(8, 34, 'Head of Technology', 0, TRUE),
(8, 34, 'Software Engineer', 1, TRUE),
(8, 34, 'Systems Analyst', 2, TRUE);

-- GlobalFlow - Supply Chain Solutions (Dept 35)
INSERT INTO roles (organization_id, department_id, name, priority) VALUES
(8, 35, 'Solutions Director', 0),
(8, 35, 'Supply Chain Consultant', 1),
(8, 35, 'Client Success Manager', 2);

-- GlobalFlow - Regional Management (Dept 36)
INSERT INTO roles (organization_id, department_id, name, priority, can_view_statistics) VALUES
(8, 36, 'Regional Director', 0, TRUE),
(8, 36, 'Area Manager', 1, FALSE),
(8, 36, 'Regional Coordinator', 2, FALSE);

-- 120-133
-- ShieldGuard - SOC (Dept 37)
INSERT INTO roles (organization_id, department_id, name, priority, can_assign_tasks, can_receive_tasks, can_post_news) VALUES
(9, 37, 'SOC Director', 0, TRUE, TRUE, TRUE),
(9, 37, 'Senior Security Analyst', 1, FALSE, TRUE, FALSE),
(9, 37, 'Security Analyst', 2, FALSE, TRUE, FALSE),
(9, 37, 'Incident Responder', 3, FALSE, TRUE, FALSE);

-- ShieldGuard - Penetration Testing (Dept 38)
INSERT INTO roles (organization_id, department_id, name, priority, can_receive_tasks) VALUES
(9, 38, 'Pentesting Lead', 0, TRUE),
(9, 38, 'Senior Pentester', 1, TRUE),
(9, 38, 'Security Tester', 2, TRUE);

-- ShieldGuard - Security Consulting (Dept 39)
INSERT INTO roles (organization_id, department_id, name, priority) VALUES
(9, 39, 'Principal Consultant', 0),
(9, 39, 'Security Consultant', 1),
(9, 39, 'Risk Analyst', 2);

-- ShieldGuard - R&D (Dept 40)
INSERT INTO roles (organization_id, department_id, name, priority, can_receive_tasks) VALUES
(9, 40, 'Research Director', 0, TRUE),
(9, 40, 'Security Researcher', 1, TRUE),
(9, 40, 'Threat Intelligence Analyst', 2, TRUE);

-- 134-146
-- LearnSphere - Content Development (Dept 41)
INSERT INTO roles (organization_id, department_id, name, priority, can_assign_tasks, can_receive_tasks) VALUES
(10, 41, 'Content Director', 0, TRUE, TRUE),
(10, 41, 'Instructional Designer', 1, FALSE, TRUE),
(10, 41, 'Subject Matter Expert', 2, FALSE, TRUE),
(10, 41, 'Content Producer', 3, FALSE, TRUE);

-- LearnSphere - Platform Engineering (Dept 42)
INSERT INTO roles (organization_id, department_id, name, priority, can_receive_tasks) VALUES
(10, 42, 'Head of Engineering', 0, TRUE),
(10, 42, 'Platform Engineer', 1, TRUE),
(10, 42, 'Backend Developer', 2, TRUE);

-- LearnSphere - Student Success (Dept 43)
INSERT INTO roles (organization_id, department_id, name, priority) VALUES
(10, 43, 'Student Success Director', 0),
(10, 43, 'Learning Coach', 1),
(10, 43, 'Support Specialist', 2);

-- LearnSphere - Partnerships (Dept 44)
INSERT INTO roles (organization_id, department_id, name, priority, can_post_news) VALUES
(10, 44, 'VP Partnerships', 0, TRUE),
(10, 44, 'Partnership Manager', 1, FALSE),
(10, 44, 'Business Development Rep', 2, FALSE);

-- ============================================
-- STEP 5: INSERT USERS
-- Many users for NeuralTech AI, minimal for others
-- ============================================

INSERT INTO users (first_name, last_name, email, password_hash, avatar_color, organization_id, department_id, role_id, is_approved, is_active, hire_date) VALUES

-- ============================================
-- NEURALTECH AI USERS (30 users across 5 departments)
-- ============================================

-- ML RESEARCH DEPARTMENT (Dept 1) - 8 users
('Sarah', 'Chen', 'sarah.chen@neuraltech.com', 'sarah123', '#667eea', 1, 1, 1, TRUE, TRUE, '2019-06-10'),  -- Research Director
('Michael', 'Rodriguez', 'michael.r@neuraltech.com', 'michael123', '#764ba2', 1, 1, 2, TRUE, TRUE, '2020-02-15'),  -- Senior ML Researcher
('Priya', 'Patel', 'priya.patel@neuraltech.com', 'priya123', '#f093fb', 1, 1, 2, TRUE, TRUE, '2020-08-20'),  -- Senior ML Researcher
('David', 'Kim', 'david.kim@neuraltech.com', 'david123', '#4facfe', 1, 1, 3, TRUE, TRUE, '2021-03-12'),  -- ML Researcher
('Amanda', 'Thompson', 'amanda.t@neuraltech.com', 'amanda123', '#43e97b', 1, 1, 3, TRUE, TRUE, '2021-09-05'),  -- ML Researcher
('James', 'Lee', 'james.lee@neuraltech.com', 'james123', '#fa709a', 1, 1, 3, TRUE, TRUE, '2022-01-18'),  -- ML Researcher
('Emily', 'Garcia', 'emily.garcia@neuraltech.com', 'emily123', '#fee140', 1, 1, 4, TRUE, TRUE, '2022-07-22'),  -- Research Engineer
('Daniel', 'Nguyen', 'daniel.n@neuraltech.com', 'daniel123', '#30cfd0', 1, 1, 4, TRUE, TRUE, '2023-02-14'),  -- Research Engineer

-- ENGINEERING DEPARTMENT (Dept 2) - 9 users
('Robert', 'Anderson', 'robert.a@neuraltech.com', 'robert123', '#a8edea', 1, 2, 5, TRUE, TRUE, '2018-05-10'),  -- VP of Engineering
('Lisa', 'Martinez', 'lisa.m@neuraltech.com', 'lisa123', '#ff6b6b', 1, 2, 6, TRUE, TRUE, '2019-11-15'),  -- Engineering Manager
('Kevin', 'Brown', 'kevin.brown@neuraltech.com', 'kevin123', '#4ecdc4', 1, 2, 6, TRUE, TRUE, '2020-04-08'),  -- Engineering Manager
('Jennifer', 'Wilson', 'jennifer.w@neuraltech.com', 'jennifer123', '#feca57', 1, 2, 7, TRUE, TRUE, '2020-09-20'),  -- Senior Engineer
('Christopher', 'Davis', 'chris.davis@neuraltech.com', 'chris123', '#48dbfb', 1, 2, 7, TRUE, TRUE, '2021-01-12'),  -- Senior Engineer
('Michelle', 'Taylor', 'michelle.t@neuraltech.com', 'michelle123', '#ff9ff3', 1, 2, 7, TRUE, TRUE, '2021-06-18'),  -- Senior Engineer
('Andrew', 'Moore', 'andrew.moore@neuraltech.com', 'andrew123', '#54a0ff', 1, 2, 8, TRUE, TRUE, '2022-03-05'),  -- Software Engineer
('Jessica', 'White', 'jessica.white@neuraltech.com', 'jessica123', '#5f27cd', 1, 2, 8, TRUE, TRUE, '2022-08-22'),  -- Software Engineer
('Matthew', 'Harris', 'matthew.h@neuraltech.com', 'matthew123', '#00d2d3', 1, 2, 8, TRUE, TRUE, '2023-01-10'),  -- Software Engineer

-- PRODUCT MANAGEMENT DEPARTMENT (Dept 3) - 5 users
('Rachel', 'Thomas', 'rachel.t@neuraltech.com', 'rachel123', '#1dd1a1', 1, 3, 9, TRUE, TRUE, '2019-08-15'),  -- Head of Product
('Brandon', 'Jackson', 'brandon.j@neuraltech.com', 'brandon123', '#ee5a6f', 1, 3, 10, TRUE, TRUE, '2020-12-10'),  -- Product Manager
( 'Stephanie', 'Clark', 'stephanie.c@neuraltech.com', 'stephanie123', '#667eea', 1, 3, 10, TRUE, TRUE, '2021-05-22'),  -- Product Manager
('Ryan', 'Lewis', 'ryan.lewis@neuraltech.com', 'ryan123', '#764ba2', 1, 3, 11, TRUE, TRUE, '2022-09-14'),  -- Product Analyst
('Nicole', 'Walker', 'nicole.w@neuraltech.com', 'nicole123', '#f093fb', 1, 3, 11, TRUE, TRUE, '2023-03-20'),  -- Product Analyst

-- SALES & BUSINESS DEVELOPMENT DEPARTMENT (Dept 4) - 5 users
('Brian', 'Young', 'brian.young@neuraltech.com', 'brian123', '#4facfe', 1, 4, 12, TRUE, TRUE, '2019-10-05'),  -- VP Sales
('Laura', 'King', 'laura.king@neuraltech.com', 'laura123', '#43e97b', 1, 4, 13, TRUE, TRUE, '2020-07-18'),  -- Enterprise Account Executive
('Nathan', 'Scott', 'nathan.scott@neuraltech.com', 'nathan123', '#fa709a', 1, 4, 13, TRUE, TRUE, '2021-02-12'),  -- Enterprise Account Executive
('Emma', 'Green', 'emma.green@neuraltech.com', 'emma123', '#fee140', 1, 4, 14, TRUE, TRUE, '2021-11-08'),  -- Business Development Manager
('Tyler', 'Adams', 'tyler.adams@neuraltech.com', 'tyler123', '#30cfd0', 1, 4, 14, TRUE, TRUE, '2022-06-15'),  -- Business Development Manager

-- HUMAN RESOURCES DEPARTMENT (Dept 5) - 3 users
( 'Sophia', 'Baker', 'sophia.baker@neuraltech.com', 'sophia123', '#a8edea', 1, 5, 15, TRUE, TRUE, '2018-09-10'),  -- Chief People Officer
( 'Jacob', 'Nelson', 'jacob.nelson@neuraltech.com', 'jacob123', '#ff6b6b', 1, 5, 16, TRUE, TRUE, '2020-03-22'),  -- HR Manager
('Olivia', 'Carter', 'olivia.carter@neuraltech.com', 'olivia123', '#4ecdc4', 1, 5, 17, TRUE, TRUE, '2022-01-15'),  -- HR Specialist

-- ============================================
-- OTHER ORGANIZATIONS (1 user each for news posting)
-- ============================================

-- Quantum Capital (Org 2)
('Emily', 'Thompson', 'emily.t@quantumcapital.com', 'emily123', '#feca57', 2, 6, 25, TRUE, TRUE, '2015-03-20'),
('Emily2', 'Thompson', 'emily.t@qua2ntumcapital.com', 'emily123', '#feca57', 2, 6, 25, TRUE, TRUE, '2015-03-20'),
('Emily3', 'Thompson', 'emily.t@qua3ntumcapital.com', 'emily123', '#feca57', 2, 6, 25, TRUE, TRUE, '2015-03-20'),
('Emily4', 'Thompson', 'emily.t@quan4tumcapital.com', 'emily123', '#feca57', 2, 6, 25, TRUE, TRUE, '2015-03-20'),
('Emily5', 'Thompson', 'emily.t@quant5umcapital.com', 'emily123', '#feca57', 2, 6, 25, TRUE, TRUE, '2015-03-20'),
('Emily6', 'Thompson', 'emily.t@qua6ntumcapital.com', 'emily123', '#feca57', 2, 6, 25, TRUE, TRUE, '2015-03-20'),
('Emily7', 'Thompson', 'emily.t@qua7ntumcapital.com', 'emily123', '#feca57', 2, 6, 25, TRUE, TRUE, '2015-03-20'),
('Emily8', 'Thompson', 'emily.t@quan8tumcapital.com', 'emily123', '#feca57', 2, 6, 25, TRUE, TRUE, '2015-03-20'),

-- MediConnect Health (Org 3)
('Jennifer', 'Martinez', 'jennifer.m@mediconnect.com', 'jennifer123', '#48dbfb', 3, 10, 40, TRUE, TRUE, '2017-05-12'),
('Emily9', 'Thompson', 'emily.t@quantum33capital.com', 'emily123', '#feca57', 3, 10, 40, TRUE, TRUE, '2015-03-20'),
('Emily10', 'Thompson', 'emily.t@quant333umcapital.com', 'emily123', '#feca57', 3, 10, 40, TRUE, TRUE, '2015-03-20'),
('Emily11', 'Thompson', 'emily.t@quant3333umcapital.com', 'emily123', '#feca57', 3, 10, 40, TRUE, TRUE, '2015-03-20'),
('Emily12', 'Thompson', 'emily.t@quantum3333capital.com', 'emily123', '#feca57', 3, 10, 40, TRUE, TRUE, '2015-03-20'),


-- ShopFlow Commerce (Org 4)
('Amanda', 'Lee', 'amanda.l@shopflow.com', 'amanda123', '#ff9ff3', 4, 18, 58, TRUE, TRUE, '2016-08-14'),
('Emily13', 'Thompson', 'emily.t@quantum7777capital.com', 'emily123', '#feca57', 4, 18, 58, TRUE, TRUE, '2015-03-20'),
('Emily14', 'Thompson', 'emily.t@quantum789capital.com', 'emily123', '#feca57', 4, 18, 58, TRUE, TRUE, '2015-03-20'),
('Emily15', 'Thompson', 'emily.t@quantumc879apital.com', 'emily123', '#feca57', 4, 18, 58, TRUE, TRUE, '2015-03-20'),
('Emily16', 'Thompson', 'emily.t@quantumc970apital.com', 'emily123', '#feca57', 4, 18, 58, TRUE, TRUE, '2015-03-20'),

-- CloudSphere Technologies (Org 5)
( 'Rachel', 'Kim', 'rachel.k@cloudsphere.io', 'rachel123', '#54a0ff', 5, 19, 40, TRUE, TRUE, '2018-07-22'),
('Emily22', 'Thompson', 'emily.t@quantumc07890apital.com', 'emily123', '#feca57', 5, 19, 40, TRUE, TRUE, '2015-03-20'),
('Emily222', 'Thompson', 'emily.t@quantumc789apital.com', 'emily123', '#feca57', 5, 19, 40, TRUE, TRUE, '2015-03-20'),

-- EcoVision Solutions (Org 6)
( 'Laura', 'Brown', 'laura.b@ecovision.com', 'laura123', '#5f27cd', 6, 24, 51, TRUE, TRUE, '2015-10-30'),
('Emily44', 'Thompson', 'emily.t@quantumca789879pital.com', 'emily123', '#feca57', 6, 24, 51, TRUE, TRUE, '2015-03-20'),
('Emily55', 'Thompson', 'emily.t@quantumca9789789pital.com', 'emily123', '#feca57', 6, 24, 51, TRUE, TRUE, '2015-03-20'),
('Emily67', 'Thompson', 'emily.t@quantumcap9678ital.com', 'emily123', '#feca57', 6, 24, 51, TRUE, TRUE, '2015-03-20'),

-- BrandBoost Digital (Org 7)
('Jessica', 'White', 'jessica.w@brandboost.com', 'jessica123', '#00d2d3', 7, 28, 63, TRUE, TRUE, '2014-12-03'),
('Emily99', 'Thompson', 'emily.t@quantumcap86789ital.com', 'emily123', '#feca57', 7, 28, 63, TRUE, TRUE, '2015-03-20'),

-- GlobalFlow Logistics (Org 8)
( 'Michelle', 'Clark', 'michelle.c@globalflow.com', 'michelle123', '#1dd1a1', 8, 33, 78, TRUE, TRUE, '2013-04-15'),

-- ShieldGuard Security (Org 9)
( 'Nicole', 'Walker', 'nicole.w@shieldguard.com', 'nicole123', '#ee5a6f', 9, 37, 90, TRUE, TRUE, '2012-09-18'),

-- LearnSphere EdTech (Org 10)
( 'Stephanie', 'Allen', 'stephanie.a@learnsphere.edu', 'stephanie123', '#667eea', 10, 41, 102, TRUE, TRUE, '2017-11-05'),
('Emily100', 'Thompson', 'emily.t@quantumc789pital.com', 'emily123', '#feca57', 10, 41, 102, TRUE, TRUE, '2015-03-20');

-- ============================================
-- STEP 6: INSERT NEWS POSTS
-- From update.sql with realistic content
-- ============================================

INSERT INTO news_posts (organization_id, author_id, title, content, image_url, is_public, views_count, created_at) VALUES

-- News 1: NeuralTech AI - WITH IMAGE
(1, 1, 'NeuralTech AI Announces Breakthrough in Natural Language Understanding with GPT-Rival Model',
'NeuralTech AI unveils its revolutionary language model, NeuroLLM-7B, achieving state-of-the-art performance across multiple benchmarks while using 40% fewer parameters than competing models. This breakthrough represents years of research into efficient transformer architectures and novel training methodologies that dramatically reduce computational requirements without sacrificing capability.

The model excels in complex reasoning tasks, multilingual understanding, and context retention across extended conversations. Independent testing shows NeuroLLM-7B outperforms models twice its size in logical reasoning benchmarks and matches leading models in creative writing tasks. Its compact architecture makes deployment feasible on standard hardware, democratizing access to advanced AI capabilities for businesses of all sizes.

Key innovations include our proprietary attention mechanism that selectively focuses on relevant context, reducing memory requirements by 60% compared to standard transformers. We have also developed novel training techniques that improve sample efficiency, enabling the model to learn from significantly less data while achieving superior generalization. These advances position NeuroLLM as an industry leader in efficient, performant language AI.

Enterprise customers can now leverage NeuroLLM through our API or deploy it on-premises for sensitive applications requiring data privacy. We offer specialized fine-tuning services to adapt the model for domain-specific tasks, with existing implementations spanning healthcare diagnosis assistance, legal document analysis, customer service automation, and software code generation.

Our commitment to responsible AI development includes comprehensive safety testing, bias mitigation protocols, and transparent documentation of model limitations. We publish detailed model cards outlining capabilities, training data composition, known biases, and recommended use cases. NeuroLLM represents not just technical achievement but our vision for AI that is powerful, accessible, and deployed responsibly.',
'https://images.unsplash.com/photo-1677442136019-21780ecad995?w=800&h=600&fit=crop',
TRUE, 2567, NOW() - INTERVAL '3 days'),

-- News 2: Quantum Capital - NO IMAGE
(2, 31, 'Quantum Capital Partners Launches $2B Climate Tech Investment Fund',
'Quantum Capital Partners announces the launch of our $2 billion Climate Tech Opportunities Fund, dedicated to investments in companies developing breakthrough technologies to combat climate change. This initiative reflects our conviction that climate technology represents the defining investment opportunity of this decade, combining potential for exceptional returns with measurable environmental impact.

The fund will focus on companies advancing renewable energy, carbon capture, sustainable materials, electric mobility, and climate adaptation technologies. Our investment thesis centers on identifying scalable solutions that can achieve cost parity with conventional alternatives while delivering significant emissions reductions. We target companies at Series B and growth stages with proven technology and clear paths to market leadership.

Our investment team includes domain experts in energy systems, materials science, and environmental engineering who rigorously evaluate technology viability alongside traditional financial metrics. We leverage our extensive network of corporate partners and research institutions to accelerate portfolio company growth through strategic partnerships, customer introductions, and technical collaboration.

Beyond capital, we provide portfolio companies with operational support spanning manufacturing scale-up, supply chain development, regulatory navigation, and go-to-market strategy. Our platform includes dedicated sustainability analysts who track environmental impact metrics and ensure investments deliver on both financial and climate objectives.

Early commitments include investments in a next-generation battery technology company, a carbon capture startup with breakthrough economics, and a sustainable aviation fuel producer. We aim to deploy the fund over 5 years, building a portfolio of 15-20 companies positioned to lead the global transition to sustainable industries.',
'',
TRUE, 1842, NOW() - INTERVAL '5 days'),

-- News 3: MediConnect - WITH IMAGE
(3, 39, 'MediConnect Health Platform Achieves 5 Million Patient Milestone with 98% Satisfaction Rate',
'MediConnect Health celebrates reaching 5 million active patients using our integrated digital health platform, representing exponential growth as healthcare providers and patients embrace telemedicine and connected care solutions. This milestone validates our vision of accessible, convenient, and high-quality healthcare delivered through technology.

Platform features include video consultations with licensed physicians, AI-powered symptom checking, electronic prescription delivery, health record management, and integration with wearable devices for continuous monitoring. Patients report average wait times of under 10 minutes for consultations, compared to hours or days for traditional appointments, while maintaining clinical outcomes equivalent to in-person care for appropriate conditions.

Our satisfaction metrics reflect comprehensive patient experience: 98% would recommend MediConnect to others, 96% rate care quality as excellent or very good, and 99% appreciate the convenience of virtual visits. Physicians using our platform report improved work-life balance and ability to see more patients while maintaining quality through efficient digital workflows and clinical decision support tools.

The platform has proven particularly impactful in rural and underserved areas where physician shortages limit healthcare access. We have facilitated over 15 million consultations, prescribed 8 million medications, and coordinated 2 million specialist referrals. Our chronic disease management programs show 30% better medication adherence and 25% fewer hospital readmissions compared to traditional care models.

Looking ahead, we are expanding into mental health services, chronic disease management, and preventive care programs. Partnerships with major employers and health plans are extending platform access to millions more Americans, advancing our mission of making quality healthcare accessible to everyone, anywhere.',
'https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=800&h=600&fit=crop',
TRUE, 3214, NOW() - INTERVAL '2 days'),

-- News 4: ShopFlow - NO IMAGE  
(4, 44, 'ShopFlow Commerce Processes Record $10B in Transactions During Holiday Season',
'ShopFlow Commerce achieved a major milestone by processing over $10 billion in transactions during the recent holiday shopping season, representing 180% year-over-year growth as more merchants adopt our platform for their e-commerce operations. Peak traffic reached 50,000 transactions per minute without service disruption, demonstrating the scalability and reliability of our infrastructure.

Our platform served over 50,000 active merchants ranging from solo entrepreneurs to enterprises processing millions daily. Total order volume exceeded 100 million across product categories spanning fashion, electronics, home goods, and specialty items. International transactions accounted for 35% of volume, reflecting our global expansion into 120 countries with localized payment methods and multi-currency support.

Technical innovations enabled this success: our machine learning fraud detection blocked $500 million in fraudulent transactions while maintaining a false positive rate under 0.1%, ensuring legitimate purchases process smoothly. Real-time inventory synchronization across sales channels prevented overselling while AI-powered demand forecasting helped merchants optimize stock levels. Mobile purchases comprised 65% of transactions, up from 45% last year, driving our continued investment in mobile optimization.

Merchants using ShopFlow reported average revenue increases of 40% compared to prior year, attributing growth to our suite of tools including abandoned cart recovery, personalized product recommendations, email marketing automation, and social media integration. Our analytics dashboard provides actionable insights on customer behavior, product performance, and marketing effectiveness.

Looking to 2025, we are launching an expanded marketplace connecting ShopFlow merchants with verified suppliers, introducing augmented reality product preview features, and rolling out AI-powered chatbots for customer service automation. These innovations reinforce our mission to empower merchants of all sizes to compete effectively in digital commerce.',
'',
TRUE, 1956, NOW() - INTERVAL '7 days'),

-- News 5: CloudSphere - WITH IMAGE
(5, 49, 'CloudSphere Technologies Achieves 99.999% Uptime Across All Services in 2024',
'CloudSphere Technologies is proud to announce that all our cloud infrastructure services achieved 99.999% uptime throughout 2024, equating to just 5.26 minutes of downtime for the entire year. This industry-leading reliability reflects our unwavering commitment to enterprise-grade infrastructure and operational excellence.

Our achievement resulted from comprehensive investments across multiple dimensions: globally distributed data centers with active-active architectures eliminating single points of failure, automated failover systems detecting and routing around issues within milliseconds, predictive maintenance using machine learning to identify and address potential problems before they impact customers, and rigorous chaos engineering testing that deliberately introduces failures to validate resilience mechanisms.

We operate 35 data centers across 6 continents, each meeting stringent security and redundancy requirements. Traffic routing intelligence directs requests to optimal locations based on latency, load, and health checks, ensuring consistent performance regardless of geographic location or traffic spikes. Our edge network includes 200+ points of presence reducing latency for end users worldwide.

Customer impact of this reliability is substantial: e-commerce platforms processed billions in transactions without revenue-impacting outages, SaaS applications maintained user productivity without service interruptions, and fintech services operated continuously throughout market volatility. Our average API response time remained under 100ms globally throughout the year despite processing over 100 billion requests monthly.

Transparency is core to our operations. We publish real-time status information and detailed incident reports for any service degradation, however minor. Our SLAs include automatic service credits for any downtime, though we issued virtually none in 2024. This track record demonstrates our technical excellence and validates customer trust in CloudSphere for their most critical workloads.',
'https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=800&h=600&fit=crop',
TRUE, 2198, NOW() - INTERVAL '4 days'),

-- News 6: EcoVision - NO IMAGE
(6, 52, 'EcoVision Solutions Helps 100 Companies Achieve Carbon Neutrality in 2024',
'EcoVision Solutions announces that 100 client companies achieved carbon neutrality in 2024 through our comprehensive sustainability programs, representing a collective reduction of 2 million tons of CO2 emissions annually. This milestone demonstrates that environmental responsibility and business success are not just compatible but mutually reinforcing.

Our carbon neutrality program combines three pillars: energy efficiency improvements reducing consumption, renewable energy adoption eliminating emissions, and strategic carbon offsets addressing remaining footprint. We begin with detailed emissions audits using our proprietary measurement platform that tracks scope 1, 2, and 3 emissions with unprecedented accuracy. Clients receive customized roadmaps prioritizing highest-impact interventions with fastest payback periods.

Typical energy efficiency projects include LED lighting retrofits, HVAC optimization, building envelope improvements, and industrial process upgrades. These initiatives reduce energy costs 20-40% while cutting emissions, typically achieving payback within 2-3 years. We finance projects through energy-as-a-service models requiring minimal upfront investment from clients.

Renewable energy solutions encompass on-site solar installations, power purchase agreements for wind energy, and renewable energy certificates. Solar projects utilize our standardized deployment approach reducing costs 30% below market averages. For companies without suitable facilities, we broker competitive PPAs leveraging our aggregated purchasing power across hundreds of clients.

Carbon offsets address unavoidable emissions through high-quality projects we rigorously vet for additionality, permanence, and co-benefits. Our portfolio emphasizes forest conservation, renewable energy in developing countries, and innovative carbon capture technologies. We retire offsets in clients names providing third-party verified certificates of climate impact.

Our 100 carbon neutral clients span manufacturing, technology, financial services, and retail sectors with combined annual revenues exceeding $50 billion. Their commitment demonstrates that sustainability leadership drives competitive advantage through brand enhancement, employee attraction, customer loyalty, and regulatory preparedness.',
'',
TRUE, 1673, NOW() - INTERVAL '6 days'),

-- News 7: BrandBoost - NO IMAGE
(7, 56, 'BrandBoost Digital Named Marketing Agency of the Year by AdTech Innovation Awards',
'BrandBoost Digital has been named Marketing Agency of the Year at the prestigious AdTech Innovation Awards, recognizing our groundbreaking campaign strategies, measurable client results, and pioneering use of artificial intelligence in digital marketing. This accolade validates our data-driven approach and creative excellence that consistently delivers exceptional ROI for clients.

Award judges highlighted our campaign for a consumer electronics brand that generated 300% ROI through personalized video advertising dynamically customized for viewer demographics, interests, and browsing behavior. The campaign reached 50 million people across digital channels with creative that resonated individually, achieving engagement rates triple industry benchmarks and directly attributing $100 million in sales.

Our marketing technology platform integrates data from dozens of sources providing unified customer views that inform strategy and execution. Machine learning models predict optimal channel mix, timing, and messaging for each customer segment. Real-time optimization automatically shifts budgets to highest-performing tactics, maximizing campaign effectiveness. Clients receive transparent analytics dashboards tracking every dollar from impression through conversion.

Client success stories span industries: a B2B software company increased qualified leads 200% while reducing cost-per-lead 40%, a retail brand achieved record holiday sales through influencer partnerships and social commerce, a healthcare provider filled appointment slots through targeted local campaigns, and a financial services firm built brand awareness among millennials through content marketing and community building.

Our team has grown by 40% this year to meet increasing demand, bringing together strategists, designers, copywriters, videographers, and data scientists who collaborate seamlessly to produce integrated campaigns. We have invested significantly in marketing technology, developing AI-powered tools for content optimization, audience targeting, and campaign personalization.

Client retention rates of 96% reflect the value we deliver through transparent reporting, proactive communication, and genuine partnership. We treat client success as our success, aligning our strategies with business objectives and adapting quickly to market changes and emerging opportunities.',
'',
TRUE, 967, NOW() - INTERVAL '6 days'),

-- News 8: GlobalFlow - WITH IMAGE
(8, 58, 'GlobalFlow Logistics Opens Advanced Automated Warehouse Facility in Singapore',
'GlobalFlow Logistics has inaugurated a state-of-the-art automated warehouse facility in Singapore, featuring robotics, AI-powered inventory management, and advanced material handling systems. This 500,000 square foot facility represents a $150 million investment in next-generation logistics infrastructure serving the Asia-Pacific region.

The facility utilizes autonomous mobile robots for goods-to-person picking, automated guided vehicles for material transport, and AI systems optimizing storage locations and retrieval sequences. Processing capacity reaches 100,000 packages daily with 99.8% accuracy, while energy-efficient design reduces power consumption by 40% compared to conventional warehouses.

Strategic location near Singapore port and airport enables rapid distribution throughout Southeast Asia, with same-day delivery capability to 15 million people and next-day delivery to 200 million people. Advanced tracking technology provides real-time visibility from warehouse receipt through final delivery, giving customers unprecedented transparency.

Sustainability features include solar panels generating 30% of facility power, rainwater harvesting systems, and electric vehicle charging infrastructure for our delivery fleet. We have committed to transitioning our entire Asia-Pacific delivery fleet to electric vehicles by 2027, significantly reducing our carbon footprint.

The facility creates 300 local jobs, with extensive training programs preparing workers for technology-enabled logistics operations. We partner with local universities to develop talent pipelines and advance logistics education. This facility positions GlobalFlow as the logistics partner of choice for e-commerce companies and manufacturers expanding in the rapidly growing Asian markets.',
'https://images.unsplash.com/photo-1586528116311-ad8dd3c8310d?w=800&h=600&fit=crop',
TRUE, 1401, NOW() - INTERVAL '8 days'),

-- News 9: ShieldGuard - NO IMAGE
(9, 59, 'ShieldGuard Security Thwarts Major Ransomware Attack Protecting 50,000 Endpoints',
'ShieldGuard Security successfully defended a Fortune 500 client against a sophisticated ransomware attack that could have encrypted critical data across 50,000 endpoints and caused hundreds of millions in damages and business disruption. Our advanced threat detection systems identified the attack within minutes of initial breach attempts, enabling rapid response that contained the threat before any data encryption occurred.

The attack employed advanced techniques including zero-day exploits, social engineering, and lateral movement through network infrastructure. Our AI-powered behavioral analysis detected anomalous activity that traditional signature-based security tools missed. Automated response protocols immediately isolated affected systems, revoked compromised credentials, and deployed patches addressing the exploited vulnerabilities.

Our security operations center, staffed 24/7 by certified experts, coordinated the response effort, conducting forensic analysis to understand attack vectors and ensure complete threat elimination. We identified the threat actors as a known cybercrime group responsible for previous high-profile attacks, providing intelligence to law enforcement agencies.

This incident highlights the evolving sophistication of cyber threats and the critical importance of proactive security measures. Organizations face constant attacks from well-funded criminal groups employing advanced tools and techniques. Our multilayered defense approach combining prevention, detection, and response proved effective where single-point solutions would have failed.

We are sharing threat intelligence with the cybersecurity community to help other organizations protect against similar attacks. Our incident response playbook and lessons learned are being incorporated into security training programs, helping organizations strengthen their security postures against emerging threats.',
'',
TRUE, 2034, NOW() - INTERVAL '9 days'),

-- News 10: LearnSphere - WITH IMAGE
(10, 60, 'LearnSphere EdTech Platform Reaches 2 Million Students Milestone with 95% Satisfaction Rating',
'LearnSphere EdTech celebrates reaching 2 million active students on our learning platform, with learners from 120 countries accessing quality education through our innovative technology. This milestone reflects growing global demand for accessible, flexible, and effective online learning solutions that accommodate diverse learning styles and schedules.

Our AI-powered personalized learning engine adapts content difficulty, pace, and format based on individual student performance and preferences, ensuring optimal learning outcomes. Students using our platform show 40% faster skill acquisition compared to traditional methods, with knowledge retention rates exceeding 85% six months after course completion.

The platform offers 5,000+ courses spanning technology, business, healthcare, creative arts, and personal development. Content is created by subject matter experts and includes video lectures, interactive simulations, hands-on projects, and peer collaboration opportunities. Our mobile app enables learning anywhere, with offline capabilities ensuring access even without internet connectivity.

Student satisfaction ratings of 95% reflect the quality and accessibility of our educational offerings. Success stories include career changers who secured new employment within three months of course completion, professionals who earned promotions after acquiring new skills, and students in developing regions who gained access to world-class education previously unavailable to them.

We have established partnerships with 500 universities and corporations, offering credit-bearing courses, professional certifications, and custom training programs. Our corporate training solutions help organizations upskill their workforce, with measurable improvements in productivity and employee satisfaction. Scholarship programs have provided free access to 100,000 students from underserved communities.',
'https://images.unsplash.com/photo-1501504905252-473c47e087f8?w=800&h=600&fit=crop',
TRUE, 1789, NOW() - INTERVAL '10 days');


-- ============================================
-- STEP 6: UPDATE DEPARTMENT MANAGERS
-- ============================================

-- NeuralTech AI department managers
UPDATE departments SET manager_id = 1 WHERE id = 1;   -- Sarah Chen manages ML Research
UPDATE departments SET manager_id = 9 WHERE id = 2;   -- Robert Anderson manages Engineering
UPDATE departments SET manager_id = 18 WHERE id = 3;  -- Rachel Thomas manages Product
UPDATE departments SET manager_id = 23 WHERE id = 4;  -- Brian Young manages Sales
UPDATE departments SET manager_id = 28 WHERE id = 5;  -- Sophia Baker manages HR

-- Other organizations
UPDATE departments SET manager_id = 31 WHERE id = 6;  -- Quantum Capital
UPDATE departments SET manager_id = 39 WHERE id = 10; -- MediConnect
UPDATE departments SET manager_id = 44 WHERE id = 15; -- ShopFlow
UPDATE departments SET manager_id = 49 WHERE id = 19; -- CloudSphere
UPDATE departments SET manager_id = 52 WHERE id = 24; -- EcoVision
UPDATE departments SET manager_id = 56 WHERE id = 28; -- BrandBoost
UPDATE departments SET manager_id = 58 WHERE id = 33; -- GlobalFlow
UPDATE departments SET manager_id = 59 WHERE id = 37; -- ShieldGuard
UPDATE departments SET manager_id = 60 WHERE id = 41; -- LearnSphere


-- ============================================
-- STEP 7: INSERT ASSIGNMENTS (NeuralTech AI only)
-- ============================================

INSERT INTO assignments (organization_id, department_id, created_by, title, description, category, deadline, priority, status) VALUES

-- ML RESEARCH ASSIGNMENTS
(1, 1, 1, 'Develop Novel Sparse Attention Mechanism', 'Research and implement a novel sparse attention mechanism that reduces computational complexity from O(n²) to O(n log n) while maintaining model performance. Focus on learnable sparsity patterns and dynamic attention allocation based on input characteristics.', 'Research', NOW() + INTERVAL '45 days', 'high', 'in_progress'),

(1, 1, 1, 'Benchmark NeuralGPT-5 Against GPT-4 and Claude', 'Conduct comprehensive benchmarking across 50+ standardized tasks including mathematical reasoning, creative writing, code generation, multilingual understanding, and logical inference. Document results in technical report with statistical significance analysis.', 'Analysis', NOW() + INTERVAL '30 days', 'high', 'in_progress'),

(1, 1, 2, 'Investigate Few-Shot Learning Improvements', 'Research techniques to improve few-shot learning capabilities. Explore meta-learning approaches, prompt engineering strategies, and in-context learning optimization. Target 25% improvement on few-shot benchmarks.', 'Research', NOW() + INTERVAL '60 days', 'medium', 'pending'),

(1, 1, 3, 'Write Paper on Efficient Transformer Architectures', 'Prepare research paper documenting our novel efficient transformer architecture for submission to NeurIPS. Include theoretical analysis, experimental results, ablation studies, and comparison with state-of-the-art models.', 'Publication', NOW() + INTERVAL '40 days', 'high', 'pending'),

(1, 1, 2, 'Implement Distributed Training Infrastructure', 'Build distributed training system supporting 1000+ GPUs across multiple data centers. Implement gradient accumulation, mixed precision training, and checkpoint sharding. Target 90% scaling efficiency.', 'Infrastructure', NOW() + INTERVAL '50 days', 'high', 'in_progress'),

(1, 1, 2, 'Optimize Model Inference Latency', 'Reduce API response time by 40% through model quantization, KV-cache optimization, speculative decoding, and infrastructure improvements. Target sub-150ms p95 latency for standard queries.', 'Performance', NOW() + INTERVAL '35 days', 'high', 'in_progress'),

(1, 1, 3, 'Build API Rate Limiting System', 'Design and implement robust API rate limiting with tiered access (free, pro, enterprise). Include JWT authentication, API key management, usage analytics dashboard, and automated throttling.', 'Backend', NOW() + INTERVAL '25 days', 'high', 'pending'),

(1, 1, 3, 'Develop Model Monitoring Dashboard', 'Create comprehensive monitoring dashboard tracking model performance metrics, API health, error rates, latency distributions, and user feedback. Include alerting for anomalies and degradation.', 'Tooling', NOW() + INTERVAL '30 days', 'medium', 'pending'),

(1, 1, 2, 'Implement Multi-Region Deployment', 'Deploy API infrastructure across 5 geographic regions with automatic failover, health checking, and intelligent request routing. Ensure sub-100ms latency for 95% of global users.', 'Infrastructure', NOW() + INTERVAL '45 days', 'high', 'pending'),

(1, 3, 2, 'Define Q1 2025 Product Roadmap', 'Develop detailed product roadmap for Q1 2025 including feature priorities, resource allocation, timeline estimates, and success metrics. Gather input from engineering, sales, and customer feedback.', 'Strategy', NOW() + INTERVAL '14 days', 'high', 'in_progress'),

(1, 3, 3, 'Conduct Enterprise Customer Research', 'Interview 20 enterprise customers to understand pain points, feature requests, and use cases. Synthesize findings into actionable product recommendations and prioritization framework.', 'Research', NOW() + INTERVAL '21 days', 'high', 'pending'),

(1, 3, 2, 'Design Fine-Tuning Product Experience', 'Design end-to-end user experience for fine-tuning feature allowing customers to customize models on their data. Include dataset upload, training configuration, evaluation metrics, and deployment.', 'Product Design', NOW() + INTERVAL '28 days', 'medium', 'pending'),

(1, 3, 3, 'Analyze API Usage Patterns', 'Deep dive into API usage data to identify common patterns, power users, feature adoption rates, and optimization opportunities. Create data-driven recommendations for product improvements.', 'Analysis', NOW() + INTERVAL '18 days', 'medium', 'in_progress'),

-- ENGINEERING ASSIGNMENTS
(1, 2, 9, 'Implement Distributed Training Infrastructure', 'Build distributed training system supporting 1000+ GPUs across multiple data centers. Implement gradient accumulation, mixed precision training, and checkpoint sharding. Target 90% scaling efficiency.', 'Infrastructure', NOW() + INTERVAL '50 days', 'high', 'in_progress'),

(1, 2, 10, 'Optimize Model Inference Latency', 'Reduce API response time by 40% through model quantization, KV-cache optimization, speculative decoding, and infrastructure improvements. Target sub-150ms p95 latency for standard queries.', 'Performance', NOW() + INTERVAL '35 days', 'high', 'in_progress'),

(1, 2, 10, 'Build API Rate Limiting System', 'Design and implement robust API rate limiting with tiered access (free, pro, enterprise). Include JWT authentication, API key management, usage analytics dashboard, and automated throttling.', 'Backend', NOW() + INTERVAL '25 days', 'high', 'pending'),

(1, 2, 11, 'Develop Model Monitoring Dashboard', 'Create comprehensive monitoring dashboard tracking model performance metrics, API health, error rates, latency distributions, and user feedback. Include alerting for anomalies and degradation.', 'Tooling', NOW() + INTERVAL '30 days', 'medium', 'pending'),

(1, 2, 12, 'Implement Multi-Region Deployment', 'Deploy API infrastructure across 5 geographic regions with automatic failover, health checking, and intelligent request routing. Ensure sub-100ms latency for 95% of global users.', 'Infrastructure', NOW() + INTERVAL '45 days', 'high', 'pending'),

-- PRODUCT MANAGEMENT ASSIGNMENTS
(1, 3, 18, 'Define Q1 2025 Product Roadmap', 'Develop detailed product roadmap for Q1 2025 including feature priorities, resource allocation, timeline estimates, and success metrics. Gather input from engineering, sales, and customer feedback.', 'Strategy', NOW() + INTERVAL '14 days', 'high', 'in_progress'),

(1, 3, 18, 'Conduct Enterprise Customer Research', 'Interview 20 enterprise customers to understand pain points, feature requests, and use cases. Synthesize findings into actionable product recommendations and prioritization framework.', 'Research', NOW() + INTERVAL '21 days', 'high', 'pending'),

(1, 3, 19, 'Design Fine-Tuning Product Experience', 'Design end-to-end user experience for fine-tuning feature allowing customers to customize models on their data. Include dataset upload, training configuration, evaluation metrics, and deployment.', 'Product Design', NOW() + INTERVAL '28 days', 'medium', 'pending'),

(1, 3, 20, 'Analyze API Usage Patterns', 'Deep dive into API usage data to identify common patterns, power users, feature adoption rates, and optimization opportunities. Create data-driven recommendations for product improvements.', 'Analysis', NOW() + INTERVAL '18 days', 'medium', 'in_progress'),

-- SALES & BD ASSIGNMENTS
(1, 4, 23, 'Close $5M Enterprise Deal with Fortune 500', 'Lead enterprise sales process for Fortune 500 financial services company. Conduct technical demos, negotiate contract terms, coordinate security review, and finalize custom pricing agreement.', 'Sales', NOW() + INTERVAL '20 days', 'high', 'in_progress'),

(1, 4, 24, 'Develop Partnership with Microsoft Azure', 'Negotiate strategic partnership with Microsoft to offer NeuralGPT through Azure Marketplace. Define revenue sharing, technical integration requirements, co-marketing initiatives, and support model.', 'Business Development', NOW() + INTERVAL '60 days', 'high', 'pending'),

(1, 4, 25, 'Create Sales Enablement Materials', 'Develop comprehensive sales enablement package including product decks, ROI calculators, case studies, competitive analysis, objection handling guides, and demo scripts.', 'Marketing', NOW() + INTERVAL '25 days', 'medium', 'pending'),

-- HR ASSIGNMENTS
(1, 5, 28, 'Launch Campus Recruiting Program', 'Design and execute campus recruiting strategy targeting top CS programs. Coordinate with universities, organize info sessions, conduct technical interviews, and extend offers to promising candidates.', 'Recruiting', NOW() + INTERVAL '90 days', 'high', 'in_progress'),

(1, 5, 29, 'Implement Performance Review System', 'Roll out new performance review system with 360-degree feedback, competency frameworks, calibration process, and career development planning. Train managers on effective feedback delivery.', 'HR Operations', NOW() + INTERVAL '30 days', 'medium', 'pending'),

(1, 5, 28, 'Develop AI Ethics Training Program', 'Create mandatory training program on AI ethics, responsible development practices, bias mitigation, and safety considerations. Deliver to all technical staff by end of quarter.', 'Training', NOW() + INTERVAL '45 days', 'medium', 'pending');


-- ============================================
-- STEP 8: INSERT ASSIGNMENT USERS (NeuralTech AI only)
-- ============================================

INSERT INTO assignment_users (assignment_id, user_id) VALUES
-- Sparse Attention research
(1, 1), (1, 2), (1, 3),(1, 4),

-- Benchmarking
(2, 1), (2, 3),(2,7),(2,4),

-- Few-shot learning
(3, 2),(3, 3),(3, 4),(3, 5),(3, 6),(3, 7),(3, 8),

-- Research paper
(4, 1), (4, 2),(4, 4),(4, 5),(4, 6),(4, 8),

(5, 2), (5, 3),(5, 4),

(6, 2),(6, 4),(6, 6),(6, 7),

(7, 1), (7, 2), (7, 3),(7, 4),

(8, 1), (8, 2),(8, 4),(8, 6),(8, 7),

(9, 1), (9, 3),(9, 4),

(10, 2),(10, 4),(10, 5),(10, 6),(10, 7),

(11, 1), (11, 2), (11, 3),(11, 4),

(12, 2),(12, 4),(12, 5),(12, 6),(12, 8),

(13, 3),

-- Distributed training
(14, 9), (14, 10), (14, 12),

-- Inference optimization
(15, 10), (15, 13), (15, 14),

-- API rate limiting
(16, 11), (16, 15),

-- Monitoring dashboard
(17, 11), (17, 16),

-- Multi-region deployment
(18, 10), (18, 12), (18, 17),

-- Product roadmap
(19, 18), (19, 19);

-- ============================================
-- STEP 9: INSERT ASSIGNMENT SUBMISSIONS (NeuralTech AI only)
-- ============================================

INSERT INTO assignment_submissions (assignment_id, user_id, notes, submitted_at) VALUES

(3, 1, 'Completed 2 comprehensive benchmarking across 52 standardized tasks. NeuralGPT-5 results: Outperforms GPT-4 on 38 tasks (73%), matches on 10 tasks (19%), underperforms on 4 tasks (8%). Key strengths: 18% better on mathematical reasoning, 22% faster on code generation, 15% more accurate on multilingual tasks. Areas for improvement: Creative writing parity, some edge cases in logical reasoning. Full technical report with statistical analysis available in research drive. Recommended next steps: targeted improvements on underperforming tasks, deeper analysis of failure modes.', NOW() - INTERVAL '3 days'),

(5, 1, 'Completed comprehensive benchmarking across 52 standardized tasks. NeuralGPT-5 results: Outperforms GPT-4 on 38 tasks (73%), matches on 10 tasks (19%), underperforms on 4 tasks (8%). Key strengths: 18% better on mathematical reasoning, 22% faster on code generation, 15% more accurate on multilingual tasks. Areas for improvement: Creative writing parity, some edge cases in logical reasoning. Full technical report with statistical analysis available in research drive. Recommended next steps: targeted improvements on underperforming tasks, deeper analysis of failure modes.', NOW() - INTERVAL '3 days'),

(10, 1, 'Usage analysis complete. Key findings: 68% of API calls are for text generation, 22% for code completion, 10% for summarization. Top 10% of users account for 65% of total volume. Average session length: 12 minutes. Peak usage: 2-4pm EST. Mobile adoption growing 40% MoM. Identified optimization opportunity: 30% of queries could use smaller model variant with cost savings. Churn risk indicator: users with <100 API calls in first week have 60% higher churn. Recommendations: proactive onboarding for low-engagement users, promote appropriate model selection, consider usage-based pricing tiers.', NOW() - INTERVAL '1 day'),

(12, 1, 'Sales enablement package completed and distributed to entire sales team. Deliverables: (1) Master product deck with customizable slides, (2) ROI calculator showing 3-5x productivity gains, (3) 5 detailed case studies across industries, (4) Competitive battlecard for GPT-4/Claude/Gemini, (5) Objection handling playbook covering 20 common concerns, (6) Demo script with technical walkthrough. Early feedback extremely positive - sales team reports 40% reduction in deal cycle time and increased confidence in technical discussions. Next iteration will include video tutorials and industry-specific variations.', NOW() - INTERVAL '2 days'),

(2, 2, 'Completed comprehensive benchmarking across 52 standardized tasks. NeuralGPT-5 results: Outperforms GPT-4 on 38 tasks (73%), matches on 10 tasks (19%), underperforms on 4 tasks (8%). Key strengths: 18% better on mathematical reasoning, 22% faster on code generation, 15% more accurate on multilingual tasks. Areas for improvement: Creative writing parity, some edge cases in logical reasoning. Full technical report with statistical analysis available in research drive. Recommended next steps: targeted improvements on underperforming tasks, deeper analysis of failure modes.', NOW() - INTERVAL '3 days'),

(9, 2, 'Usage analysis complete. Key findings: 68% of API calls are for text generation, 22% for code completion, 10% for summarization. Top 10% of users account for 65% of total volume. Average session length: 12 minutes. Peak usage: 2-4pm EST. Mobile adoption growing 40% MoM. Identified optimization opportunity: 30% of queries could use smaller model variant with cost savings. Churn risk indicator: users with <100 API calls in first week have 60% higher churn. Recommendations: proactive onboarding for low-engagement users, promote appropriate model selection, consider usage-based pricing tiers.', NOW() - INTERVAL '1 day'),

(13, 2, 'Sales enablement package completed and distributed to entire sales team. Deliverables: (1) Master product deck with customizable slides, (2) ROI calculator showing 3-5x productivity gains, (3) 5 detailed case studies across industries, (4) Competitive battlecard for GPT-4/Claude/Gemini, (5) Objection handling playbook covering 20 common concerns, (6) Demo script with technical walkthrough. Early feedback extremely positive - sales team reports 40% reduction in deal cycle time and increased confidence in technical discussions. Next iteration will include video tutorials and industry-specific variations.', NOW() - INTERVAL '2 days'),

(4, 3, 'Completed comprehensive benchmarking across 52 standardized tasks. NeuralGPT-5 results: Outperforms GPT-4 on 38 tasks (73%), matches on 10 tasks (19%), underperforms on 4 tasks (8%). Key strengths: 18% better on mathematical reasoning, 22% faster on code generation, 15% more accurate on multilingual tasks. Areas for improvement: Creative writing parity, some edge cases in logical reasoning. Full technical report with statistical analysis available in research drive. Recommended next steps: targeted improvements on underperforming tasks, deeper analysis of failure modes.', NOW() - INTERVAL '3 days'),

(6, 3, 'Usage analysis complete. Key findings: 68% of API calls are for text generation, 22% for code completion, 10% for summarization. Top 10% of users account for 65% of total volume. Average session length: 12 minutes. Peak usage: 2-4pm EST. Mobile adoption growing 40% MoM. Identified optimization opportunity: 30% of queries could use smaller model variant with cost savings. Churn risk indicator: users with <100 API calls in first week have 60% higher churn. Recommendations: proactive onboarding for low-engagement users, promote appropriate model selection, consider usage-based pricing tiers.', NOW() - INTERVAL '1 day'),

(12, 3, 'Sales enablement package completed and distributed to entire sales team. Deliverables: (1) Master product deck with customizable slides, (2) ROI calculator showing 3-5x productivity gains, (3) 5 detailed case studies across industries, (4) Competitive battlecard for GPT-4/Claude/Gemini, (5) Objection handling playbook covering 20 common concerns, (6) Demo script with technical walkthrough. Early feedback extremely positive - sales team reports 40% reduction in deal cycle time and increased confidence in technical discussions. Next iteration will include video tutorials and industry-specific variations.', NOW() - INTERVAL '2 days'),

(13, 4, 'Sales enablement package completed and distributed to entire sales team. Deliverables: (1) Master product deck with customizable slides, (2) ROI calculator showing 3-5x productivity gains, (3) 5 detailed case studies across industries, (4) Competitive battlecard for GPT-4/Claude/Gemini, (5) Objection handling playbook covering 20 common concerns, (6) Demo script with technical walkthrough. Early feedback extremely positive - sales team reports 40% reduction in deal cycle time and increased confidence in technical discussions. Next iteration will include video tutorials and industry-specific variations.', NOW() - INTERVAL '2 days'),

(5, 5, 'Usage analysis complete. Key findings: 68% of API calls are for text generation, 22% for code completion, 10% for summarization. Top 10% of users account for 65% of total volume. Average session length: 12 minutes. Peak usage: 2-4pm EST. Mobile adoption growing 40% MoM. Identified optimization opportunity: 30% of queries could use smaller model variant with cost savings. Churn risk indicator: users with <100 API calls in first week have 60% higher churn. Recommendations: proactive onboarding for low-engagement users, promote appropriate model selection, consider usage-based pricing tiers.', NOW() - INTERVAL '1 day'),

(7, 5, 'Sales enablement package completed and distributed to entire sales team. Deliverables: (1) Master product deck with customizable slides, (2) ROI calculator showing 3-5x productivity gains, (3) 5 detailed case studies across industries, (4) Competitive battlecard for GPT-4/Claude/Gemini, (5) Objection handling playbook covering 20 common concerns, (6) Demo script with technical walkthrough. Early feedback extremely positive - sales team reports 40% reduction in deal cycle time and increased confidence in technical discussions. Next iteration will include video tutorials and industry-specific variations.', NOW() - INTERVAL '2 days'),

(9, 5, 'Sales enablement package completed and distributed to entire sales team. Deliverables: (1) Master product deck with customizable slides, (2) ROI calculator showing 3-5x productivity gains, (3) 5 detailed case studies across industries, (4) Competitive battlecard for GPT-4/Claude/Gemini, (5) Objection handling playbook covering 20 common concerns, (6) Demo script with technical walkthrough. Early feedback extremely positive - sales team reports 40% reduction in deal cycle time and increased confidence in technical discussions. Next iteration will include video tutorials and industry-specific variations.', NOW() - INTERVAL '2 days'),

(1, 6, 'Sales enablement package completed and distributed to entire sales team. Deliverables: (1) Master product deck with customizable slides, (2) ROI calculator showing 3-5x productivity gains, (3) 5 detailed case studies across industries, (4) Competitive battlecard for GPT-4/Claude/Gemini, (5) Objection handling playbook covering 20 common concerns, (6) Demo script with technical walkthrough. Early feedback extremely positive - sales team reports 40% reduction in deal cycle time and increased confidence in technical discussions. Next iteration will include video tutorials and industry-specific variations.', NOW() - INTERVAL '2 days'),

(2, 6, 'Sales enablement package completed and distributed to entire sales team. Deliverables: (1) Master product deck with customizable slides, (2) ROI calculator showing 3-5x productivity gains, (3) 5 detailed case studies across industries, (4) Competitive battlecard for GPT-4/Claude/Gemini, (5) Objection handling playbook covering 20 common concerns, (6) Demo script with technical walkthrough. Early feedback extremely positive - sales team reports 40% reduction in deal cycle time and increased confidence in technical discussions. Next iteration will include video tutorials and industry-specific variations.', NOW() - INTERVAL '2 days'),

(1, 8, 'Sales enablement package completed and distributed to entire sales team. Deliverables: (1) Master product deck with customizable slides, (2) ROI calculator showing 3-5x productivity gains, (3) 5 detailed case studies across industries, (4) Competitive battlecard for GPT-4/Claude/Gemini, (5) Objection handling playbook covering 20 common concerns, (6) Demo script with technical walkthrough. Early feedback extremely positive - sales team reports 40% reduction in deal cycle time and increased confidence in technical discussions. Next iteration will include video tutorials and industry-specific variations.', NOW() - INTERVAL '2 days'),

(2, 8, 'Sales enablement package completed and distributed to entire sales team. Deliverables: (1) Master product deck with customizable slides, (2) ROI calculator showing 3-5x productivity gains, (3) 5 detailed case studies across industries, (4) Competitive battlecard for GPT-4/Claude/Gemini, (5) Objection handling playbook covering 20 common concerns, (6) Demo script with technical walkthrough. Early feedback extremely positive - sales team reports 40% reduction in deal cycle time and increased confidence in technical discussions. Next iteration will include video tutorials and industry-specific variations.', NOW() - INTERVAL '2 days'),

(6, 8, 'Sales enablement package completed and distributed to entire sales team. Deliverables: (1) Master product deck with customizable slides, (2) ROI calculator showing 3-5x productivity gains, (3) 5 detailed case studies across industries, (4) Competitive battlecard for GPT-4/Claude/Gemini, (5) Objection handling playbook covering 20 common concerns, (6) Demo script with technical walkthrough. Early feedback extremely positive - sales team reports 40% reduction in deal cycle time and increased confidence in technical discussions. Next iteration will include video tutorials and industry-specific variations.', NOW() - INTERVAL '2 days'),

(8, 8, 'Sales enablement package completed and distributed to entire sales team. Deliverables: (1) Master product deck with customizable slides, (2) ROI calculator showing 3-5x productivity gains, (3) 5 detailed case studies across industries, (4) Competitive battlecard for GPT-4/Claude/Gemini, (5) Objection handling playbook covering 20 common concerns, (6) Demo script with technical walkthrough. Early feedback extremely positive - sales team reports 40% reduction in deal cycle time and increased confidence in technical discussions. Next iteration will include video tutorials and industry-specific variations.', NOW() - INTERVAL '2 days');

-- ============================================
-- STEP 10: INSERT ASSIGNMENT RELOCATIONS (NeuralTech AI only)
-- ============================================

INSERT INTO assignment_relocations (assignment_id, department_id, from_user_id, to_user_id, reason, status, requested_at) VALUES

(4, 1, 8,7, 'Inference optimization project has grown in scope beyond initial estimates. Currently handling distributed training infrastructure project which is at critical phase requiring 80% of my time. Recommend reassigning to senior engineer with deep systems performance expertise. Christopher Davis has relevant background in GPU optimization and could take ownership. Can provide 2-week knowledge transfer and remain available for consultation.', 'pending', NOW() - INTERVAL '2 days'),

(13, 1, 3, 4, 'Fine-tuning UX design requires specialized ML product experience that goes beyond my current skill set. While I can contribute to general product flow, the technical complexity of training configuration, hyperparameter selection, and evaluation metrics needs someone with deeper ML background. Stephanie Clark recently completed ML product management course and has worked closely with research team. Recommend reassigning to her with my support on general UX patterns.', 'pending', NOW() - INTERVAL '1 day'),

(6, 1, 4,8, 'Inference optimization project has grown in scope beyond initial estimates. Currently handling distributed training infrastructure project which is at critical phase requiring 80% of my time. Recommend reassigning to senior engineer with deep systems performance expertise. Christopher Davis has relevant background in GPU optimization and could take ownership. Can provide 2-week knowledge transfer and remain available for consultation.', 'pending', NOW() - INTERVAL '2 days'),

(9, 1, 2,5, 'Fine-tuning UX design requires specialized ML product experience that goes beyond my current skill set. While I can contribute to general product flow, the technical complexity of training configuration, hyperparameter selection, and evaluation metrics needs someone with deeper ML background. Stephanie Clark recently completed ML product management course and has worked closely with research team. Recommend reassigning to her with my support on general UX patterns.', 'pending', NOW() - INTERVAL '1 day'),

(12, 1, 4,7, '2Fine-tuning UX design requires specialized ML product experience that goes beyond my current skill set. While I can contribute to general product flow, the technical complexity of training configuration, hyperparameter selection, and evaluation metrics needs someone with deeper ML background. Stephanie Clark recently completed ML product management course and has worked closely with research team. Recommend reassigning to her with my support on general UX patterns.', 'pending', NOW() - INTERVAL '1 day');


-- ============================================
-- STEP 11: INSERT EVENTS (NeuralTech AI only)
-- ============================================

INSERT INTO events (organization_id, created_by, department_id, title, description, event_date, location, is_personal) VALUES

-- ML Research events
(1, 1, 1, 'ML Research Weekly Sync', 'Weekly team meeting to discuss research progress, recent paper reviews, experiment results, and upcoming conference deadlines. All ML researchers expected to attend and share updates.', NOW() + INTERVAL '2 days' + TIME '10:00:00', 'Conference Room 3A', FALSE),

(1, 2, 1, 'NeurIPS Paper Reading Group', 'Bi-weekly paper discussion group covering recent NeurIPS acceptances. This week: "Efficient Transformers: A Survey" and "Scaling Laws for Neural Language Models". Come prepared with questions and critiques.', NOW() + INTERVAL '5 days' + TIME '14:00:00', 'Research Library', FALSE),

(1, 1, NULL, 'NeurIPS Conference Travel', 'Personal calendar block: NeurIPS conference in New Orleans. Presenting paper "Sparse Attention for Efficient Language Models" on Dec 12. Schedule 1-on-1s with collaborators from Stanford and MIT.', NOW() + INTERVAL '30 days' + TIME '09:00:00', 'New Orleans, LA', TRUE),

(1, 3, 1, 'Experiment Review - GPT-5 Architecture', 'Deep dive into recent experiments on sparse attention architecture. Review training curves, loss plateaus, downstream task performance, and ablation studies. Decide on next iteration priorities.', NOW() + INTERVAL '4 days' + TIME '15:00:00', 'Research Pod A', FALSE),

-- Engineering events
(1, 9, 2, 'Engineering All-Hands', 'Monthly engineering organization meeting. Agenda: Q4 roadmap review, infrastructure updates, security initiatives, new hire introductions, and Q&A with leadership. Pizza provided.', NOW() + INTERVAL '10 days' + TIME '16:00:00', 'Main Auditorium', FALSE),

(1, 10, 2, 'Sprint Planning - Inference Optimization', 'Two-week sprint planning for inference optimization initiative. Review backlog, break down user stories, estimate complexity, assign tasks, and identify dependencies. Target: 40% latency reduction.', NOW() + INTERVAL '3 days' + TIME '14:00:00', 'Engineering Pod B', FALSE),

(1, 11, 2, 'Architecture Review - Multi-Region Deployment', 'Technical design review for multi-region deployment architecture. Topics: data replication strategy, failover mechanisms, geo-routing logic, monitoring approach, and cost analysis. Prepare diagrams.', NOW() + INTERVAL '6 days' + TIME '13:00:00', 'Conference Room Tech', FALSE),

(1, 10, NULL, 'AWS re:Invent Conference', 'Personal reminder: Attending AWS re:Invent in Las Vegas. Sessions on EKS optimization, observability best practices, and cost management. Networking with cloud infrastructure community.', NOW() + INTERVAL '45 days' + TIME '08:00:00', 'Las Vegas, NV', TRUE),

(1, 12, 2, 'Daily Engineering Stand-up', 'Daily 15-minute sync for infrastructure team. Quick updates on progress, blockers, and priorities. Keep it brief and actionable.', NOW() + INTERVAL '1 day' + TIME '09:30:00', 'Engineering Space', FALSE),

-- Product events
(1, 18, 3, 'Product Strategy Offsite', 'Full-day product strategy session. Topics: 2025 vision, competitive landscape analysis, feature prioritization, pricing strategy, and go-to-market planning. Bring strategic thinking hats.', NOW() + INTERVAL '12 days' + TIME '09:00:00', 'Offsite Venue - TBD', FALSE),

(1, 19, 3, 'Customer Advisory Board Meeting', 'Quarterly meeting with enterprise customer advisory board. Present product roadmap, gather feedback on proposed features, discuss pain points, and identify expansion opportunities. 10 customers attending.', NOW() + INTERVAL '18 days' + TIME '11:00:00', 'Virtual - Zoom', FALSE),

(1, 20, 3, 'User Research Synthesis Workshop', 'Workshop to synthesize findings from 20 enterprise customer interviews. Use affinity mapping to identify patterns, prioritize insights, and translate into actionable product recommendations.', NOW() + INTERVAL '7 days' + TIME '14:00:00', 'Product War Room', FALSE),

(1, 18, NULL, 'AAAI Conference - Product Track', 'Personal note: Speaking at AAAI on product management for AI systems. Presentation: "Bridging Research and Product: Lessons from Building NeuralGPT". Prep slides by next week.', NOW() + INTERVAL '60 days' + TIME '10:00:00', 'Vancouver, BC', TRUE),

-- Sales & BD events
(1, 23, 4, 'Sales Team Weekly Pipeline Review', 'Weekly pipeline review covering all active enterprise deals. Review stage progression, identify blockers, strategize on closing tactics, and forecast Q4 revenue. Come prepared with deal updates.', NOW() + INTERVAL '3 days' + TIME '10:00:00', 'Sales Conference Room', FALSE),

(1, 24, 4, 'Enterprise Demo - Fortune 500 Financial', 'Critical product demo for Fortune 500 financial services prospect. Showcase enterprise features, security controls, compliance certifications, and custom deployment options. Technical team on standby.', NOW() + INTERVAL '5 days' + TIME '15:00:00', 'Virtual - Teams', FALSE),

(1, 23, NULL, 'SaaStr Conference - Networking', 'Personal calendar block: Attending SaaStr Annual. Focus on networking with enterprise SaaS leaders, attending sales strategy sessions, and promoting NeuralTech in vendor area.', NOW() + INTERVAL '50 days' + TIME '08:00:00', 'San Francisco, CA', TRUE),

(1, 26, 4, 'Partnership Meeting - Microsoft Azure', 'Strategic partnership discussion with Microsoft Azure team. Agenda: marketplace integration timeline, revenue sharing terms, co-marketing opportunities, and technical requirements. Legal team joining.', NOW() + INTERVAL '15 days' + TIME '13:00:00', 'Microsoft Office - Redmond', FALSE),

-- HR events
(1, 28, 5, 'Leadership Team Meeting', 'Monthly leadership team meeting with all department heads. Topics: hiring progress, org health metrics, performance review timeline, diversity initiatives, and budget planning for 2025.', NOW() + INTERVAL '8 days' + TIME '09:00:00', 'Executive Board Room', FALSE),

(1, 29, 5, 'New Hire Orientation', 'Full-day orientation for December new hires (5 employees). Cover company mission, values, benefits, systems access, security training, team introductions, and welcome lunch with exec team.', NOW() + INTERVAL '14 days' + TIME '09:00:00', 'Training Room A', FALSE),

(1, 28, 5, 'Campus Recruiting - MIT Career Fair', 'MIT CS career fair - full day recruitment event. Set up booth, conduct initial screenings, attend info session, host networking dinner with top candidates. Target: 15 qualified candidates.', NOW() + INTERVAL '25 days' + TIME '10:00:00', 'MIT Campus', FALSE),

(1, 30, 5, 'DEI Committee Meeting', 'Monthly diversity, equity, and inclusion committee meeting. Review hiring pipeline diversity, discuss employee resource group activities, plan cultural events, and address concerns raised in surveys.', NOW() + INTERVAL '11 days' + TIME '16:00:00', 'HR Meeting Room', FALSE),

-- Company-wide events
(1, 1, NULL, 'Company All-Hands Meeting', 'Quarterly company all-hands. CEO presents business results, product demos latest features, sales shares major wins, research previews upcoming publications. Open Q&A session. Mandatory attendance.', NOW() + INTERVAL '20 days' + TIME '16:00:00', 'Main Auditorium + Virtual', FALSE),

(1, 9, NULL, 'Holiday Party 2024', 'Annual company holiday party! Dinner, drinks, games, awards ceremony, and year-end celebration. Bring your plus-one. Business casual attire. RSVP by Dec 1st.', NOW() + INTERVAL '40 days' + TIME '18:00:00', 'The Foundry SF', FALSE);


-- ============================================
-- STEP 12: INSERT ATTENDANCE RECORDS (NeuralTech AI only)
-- ============================================

INSERT INTO attendance (user_id, organization_id, department_id, date, check_in_time, check_out_time, is_complete) VALUES

-- Last 5 business days attendance for all 30 NeuralTech employees

-- CURRENT_DATE - 4 (5 days ago)
(1, 1, 1, CURRENT_DATE - 4, (CURRENT_DATE - 4) + TIME '08:45:00', (CURRENT_DATE - 4) + TIME '18:30:00', TRUE),
(2, 1, 1, CURRENT_DATE - 4, (CURRENT_DATE - 4) + TIME '09:00:00', (CURRENT_DATE - 4) + TIME '19:00:00', TRUE),
(3, 1, 1, CURRENT_DATE - 4, (CURRENT_DATE - 4) + TIME '08:55:00', (CURRENT_DATE - 4) + TIME '18:15:00', TRUE),
(4, 1, 1, CURRENT_DATE - 4, (CURRENT_DATE - 4) + TIME '09:10:00', (CURRENT_DATE - 4) + TIME '18:45:00', TRUE),
(5, 1, 1, CURRENT_DATE - 4, (CURRENT_DATE - 4) + TIME '08:50:00', (CURRENT_DATE - 4) + TIME '18:00:00', TRUE),
(6, 1, 1, CURRENT_DATE - 4, (CURRENT_DATE - 4) + TIME '09:05:00', (CURRENT_DATE - 4) + TIME '18:30:00', TRUE),
(7, 1, 1, CURRENT_DATE - 4, (CURRENT_DATE - 4) + TIME '09:15:00', (CURRENT_DATE - 4) + TIME '18:20:00', TRUE),
(8, 1, 1, CURRENT_DATE - 4, (CURRENT_DATE - 4) + TIME '09:20:00', (CURRENT_DATE - 4) + TIME '18:40:00', TRUE),

(9, 1, 2, CURRENT_DATE - 4, (CURRENT_DATE - 4) + TIME '08:30:00', (CURRENT_DATE - 4) + TIME '18:00:00', TRUE),
(10, 1, 2, CURRENT_DATE - 4, (CURRENT_DATE - 4) + TIME '08:45:00', (CURRENT_DATE - 4) + TIME '18:15:00', TRUE),
(11, 1, 2, CURRENT_DATE - 4, (CURRENT_DATE - 4) + TIME '08:55:00', (CURRENT_DATE - 4) + TIME '18:30:00', TRUE),
(12, 1, 2, CURRENT_DATE - 4, (CURRENT_DATE - 4) + TIME '09:00:00', (CURRENT_DATE - 4) + TIME '18:45:00', TRUE),
(13, 1, 2, CURRENT_DATE - 4, (CURRENT_DATE - 4) + TIME '09:05:00', (CURRENT_DATE - 4) + TIME '19:00:00', TRUE),
(14, 1, 2, CURRENT_DATE - 4, (CURRENT_DATE - 4) + TIME '08:57:00', (CURRENT_DATE - 4) + TIME '18:20:00', TRUE),
(15, 1, 2, CURRENT_DATE - 4, (CURRENT_DATE - 4) + TIME '09:10:00', (CURRENT_DATE - 4) + TIME '18:50:00', TRUE),
(16, 1, 2, CURRENT_DATE - 4, (CURRENT_DATE - 4) + TIME '09:15:00', (CURRENT_DATE - 4) + TIME '18:40:00', TRUE),
(17, 1, 2, CURRENT_DATE - 4, (CURRENT_DATE - 4) + TIME '09:20:00', (CURRENT_DATE - 4) + TIME '18:55:00', TRUE),

(18, 1, 3, CURRENT_DATE - 4, (CURRENT_DATE - 4) + TIME '08:40:00', (CURRENT_DATE - 4) + TIME '17:45:00', TRUE),
(19, 1, 3, CURRENT_DATE - 4, (CURRENT_DATE - 4) + TIME '09:00:00', (CURRENT_DATE - 4) + TIME '18:00:00', TRUE),
(20, 1, 3, CURRENT_DATE - 4, (CURRENT_DATE - 4) + TIME '09:05:00', (CURRENT_DATE - 4) + TIME '18:10:00', TRUE),
(21, 1, 3, CURRENT_DATE - 4, (CURRENT_DATE - 4) + TIME '09:15:00', (CURRENT_DATE - 4) + TIME '18:30:00', TRUE),
(22, 1, 3, CURRENT_DATE - 4, (CURRENT_DATE - 4) + TIME '09:20:00', (CURRENT_DATE - 4) + TIME '18:15:00', TRUE),

(23, 1, 4, CURRENT_DATE - 4, (CURRENT_DATE - 4) + TIME '08:30:00', (CURRENT_DATE - 4) + TIME '19:00:00', TRUE),
(24, 1, 4, CURRENT_DATE - 4, (CURRENT_DATE - 4) + TIME '08:45:00', (CURRENT_DATE - 4) + TIME '18:30:00', TRUE),
(25, 1, 4, CURRENT_DATE - 4, (CURRENT_DATE - 4) + TIME '09:00:00', (CURRENT_DATE - 4) + TIME '18:45:00', TRUE),
(26, 1, 4, CURRENT_DATE - 4, (CURRENT_DATE - 4) + TIME '09:10:00', (CURRENT_DATE - 4) + TIME '18:20:00', TRUE),
(27, 1, 4, CURRENT_DATE - 4, (CURRENT_DATE - 4) + TIME '09:15:00', (CURRENT_DATE - 4) + TIME '18:40:00', TRUE),

(28, 1, 5, CURRENT_DATE - 4, (CURRENT_DATE - 4) + TIME '08:35:00', (CURRENT_DATE - 4) + TIME '17:30:00', TRUE),
(29, 1, 5, CURRENT_DATE - 4, (CURRENT_DATE - 4) + TIME '08:50:00', (CURRENT_DATE - 4) + TIME '17:45:00', TRUE),
(30, 1, 5, CURRENT_DATE - 4, (CURRENT_DATE - 4) + TIME '09:00:00', (CURRENT_DATE - 4) + TIME '17:50:00', TRUE),

-- CURRENT_DATE - 3
(1, 1, 1, CURRENT_DATE - 3, (CURRENT_DATE - 3) + TIME '08:50:00', (CURRENT_DATE - 3) + TIME '18:20:00', TRUE),
(2, 1, 1, CURRENT_DATE - 3, (CURRENT_DATE - 3) + TIME '09:05:00', (CURRENT_DATE - 3) + TIME '19:15:00', TRUE),
(3, 1, 1, CURRENT_DATE - 3, (CURRENT_DATE - 3) + TIME '09:00:00', (CURRENT_DATE - 3) + TIME '18:30:00', TRUE),
(4, 1, 1, CURRENT_DATE - 3, (CURRENT_DATE - 3) + TIME '09:15:00', (CURRENT_DATE - 3) + TIME '18:40:00', TRUE),
(5, 1, 1, CURRENT_DATE - 3, (CURRENT_DATE - 3) + TIME '08:45:00', (CURRENT_DATE - 3) + TIME '18:10:00', TRUE),
(6, 1, 1, CURRENT_DATE - 3, (CURRENT_DATE - 3) + TIME '09:10:00', (CURRENT_DATE - 3) + TIME '18:25:00', TRUE),
(7, 1, 1, CURRENT_DATE - 3, (CURRENT_DATE - 3) + TIME '09:20:00', (CURRENT_DATE - 3) + TIME '18:35:00', TRUE),
(8, 1, 1, CURRENT_DATE - 3, (CURRENT_DATE - 3) + TIME '09:25:00', (CURRENT_DATE - 3) + TIME '18:50:00', TRUE),

(9, 1, 2, CURRENT_DATE - 3, (CURRENT_DATE - 3) + TIME '08:35:00', (CURRENT_DATE - 3) + TIME '18:05:00', TRUE),
(10, 1, 2, CURRENT_DATE - 3, (CURRENT_DATE - 3) + TIME '08:50:00', (CURRENT_DATE - 3) + TIME '18:20:00', TRUE),
(11, 1, 2, CURRENT_DATE - 3, (CURRENT_DATE - 3) + TIME '09:00:00', (CURRENT_DATE - 3) + TIME '18:35:00', TRUE),
(12, 1, 2, CURRENT_DATE - 3, (CURRENT_DATE - 3) + TIME '09:05:00', (CURRENT_DATE - 3) + TIME '18:50:00', TRUE),
(13, 1, 2, CURRENT_DATE - 3, (CURRENT_DATE - 3) + TIME '09:10:00', (CURRENT_DATE - 3) + TIME '19:05:00', TRUE),
(14, 1, 2, CURRENT_DATE - 3, (CURRENT_DATE - 3) + TIME '08:55:00', (CURRENT_DATE - 3) + TIME '18:25:00', TRUE),
(15, 1, 2, CURRENT_DATE - 3, (CURRENT_DATE - 3) + TIME '09:15:00', (CURRENT_DATE - 3) + TIME '18:55:00', TRUE),
(16, 1, 2, CURRENT_DATE - 3, (CURRENT_DATE - 3) + TIME '09:20:00', (CURRENT_DATE - 3) + TIME '18:45:00', TRUE),
(17, 1, 2, CURRENT_DATE - 3, (CURRENT_DATE - 3) + TIME '09:25:00', (CURRENT_DATE - 3) + TIME '19:00:00', TRUE),

(18, 1, 3, CURRENT_DATE - 3, (CURRENT_DATE - 3) + TIME '08:45:00', (CURRENT_DATE - 3) + TIME '17:50:00', TRUE),
(19, 1, 3, CURRENT_DATE - 3, (CURRENT_DATE - 3) + TIME '09:05:00', (CURRENT_DATE - 3) + TIME '18:05:00', TRUE),
(20, 1, 3, CURRENT_DATE - 3, (CURRENT_DATE - 3) + TIME '09:10:00', (CURRENT_DATE - 3) + TIME '18:15:00', TRUE),
(21, 1, 3, CURRENT_DATE - 3, (CURRENT_DATE - 3) + TIME '09:20:00', (CURRENT_DATE - 3) + TIME '18:35:00', TRUE),
(22, 1, 3, CURRENT_DATE - 3, (CURRENT_DATE - 3) + TIME '09:25:00', (CURRENT_DATE - 3) + TIME '18:20:00', TRUE),

(23, 1, 4, CURRENT_DATE - 3, (CURRENT_DATE - 3) + TIME '08:35:00', (CURRENT_DATE - 3) + TIME '19:05:00', TRUE),
(24, 1, 4, CURRENT_DATE - 3, (CURRENT_DATE - 3) + TIME '08:50:00', (CURRENT_DATE - 3) + TIME '18:35:00', TRUE),
(25, 1, 4, CURRENT_DATE - 3, (CURRENT_DATE - 3) + TIME '09:05:00', (CURRENT_DATE - 3) + TIME '18:50:00', TRUE),
(26, 1, 4, CURRENT_DATE - 3, (CURRENT_DATE - 3) + TIME '09:15:00', (CURRENT_DATE - 3) + TIME '18:25:00', TRUE),
(27, 1, 4, CURRENT_DATE - 3, (CURRENT_DATE - 3) + TIME '09:20:00', (CURRENT_DATE - 3) + TIME '18:45:00', TRUE),

(28, 1, 5, CURRENT_DATE - 3, (CURRENT_DATE - 3) + TIME '08:40:00', (CURRENT_DATE - 3) + TIME '17:35:00', TRUE),
(29, 1, 5, CURRENT_DATE - 3, (CURRENT_DATE - 3) + TIME '08:55:00', (CURRENT_DATE - 3) + TIME '17:50:00', TRUE),
(30, 1, 5, CURRENT_DATE - 3, (CURRENT_DATE - 3) + TIME '09:05:00', (CURRENT_DATE - 3) + TIME '17:55:00', TRUE),

-- CURRENT_DATE - 2
(1, 1, 1, CURRENT_DATE - 2, (CURRENT_DATE - 2) + TIME '08:40:00', (CURRENT_DATE - 2) + TIME '18:25:00', TRUE),
(2, 1, 1, CURRENT_DATE - 2, (CURRENT_DATE - 2) + TIME '08:55:00', (CURRENT_DATE - 2) + TIME '19:10:00', TRUE),
(3, 1, 1, CURRENT_DATE - 2, (CURRENT_DATE - 2) + TIME '08:50:00', (CURRENT_DATE - 2) + TIME '18:20:00', TRUE),
(4, 1, 1, CURRENT_DATE - 2, (CURRENT_DATE - 2) + TIME '09:05:00', (CURRENT_DATE - 2) + TIME '18:35:00', TRUE),
(5, 1, 1, CURRENT_DATE - 2, (CURRENT_DATE - 2) + TIME '08:55:00', (CURRENT_DATE - 2) + TIME '18:05:00', TRUE),
(6, 1, 1, CURRENT_DATE - 2, (CURRENT_DATE - 2) + TIME '09:00:00', (CURRENT_DATE - 2) + TIME '18:20:00', TRUE),
(7, 1, 1, CURRENT_DATE - 2, (CURRENT_DATE - 2) + TIME '09:10:00', (CURRENT_DATE - 2) + TIME '18:30:00', TRUE),
(8, 1, 1, CURRENT_DATE - 2, (CURRENT_DATE - 2) + TIME '09:15:00', (CURRENT_DATE - 2) + TIME '18:45:00', TRUE),

(9, 1, 2, CURRENT_DATE - 2, (CURRENT_DATE - 2) + TIME '08:40:00', (CURRENT_DATE - 2) + TIME '18:10:00', TRUE),
(10, 1, 2, CURRENT_DATE - 2, (CURRENT_DATE - 2) + TIME '08:55:00', (CURRENT_DATE - 2) + TIME '18:25:00', TRUE),
(11, 1, 2, CURRENT_DATE - 2, (CURRENT_DATE - 2) + TIME '09:05:00', (CURRENT_DATE - 2) + TIME '18:40:00', TRUE),
(12, 1, 2, CURRENT_DATE - 2, (CURRENT_DATE - 2) + TIME '09:10:00', (CURRENT_DATE - 2) + TIME '18:55:00', TRUE),
(13, 1, 2, CURRENT_DATE - 2, (CURRENT_DATE - 2) + TIME '09:15:00', (CURRENT_DATE - 2) + TIME '19:10:00', TRUE),
(14, 1, 2, CURRENT_DATE - 2, (CURRENT_DATE - 2) + TIME '09:00:00', (CURRENT_DATE - 2) + TIME '18:30:00', TRUE),
(15, 1, 2, CURRENT_DATE - 2, (CURRENT_DATE - 2) + TIME '09:20:00', (CURRENT_DATE - 2) + TIME '19:00:00', TRUE),
(16, 1, 2, CURRENT_DATE - 2, (CURRENT_DATE - 2) + TIME '09:25:00', (CURRENT_DATE - 2) + TIME '18:50:00', TRUE),
(17, 1, 2, CURRENT_DATE - 2, (CURRENT_DATE - 2) + TIME '09:30:00', (CURRENT_DATE - 2) + TIME '19:05:00', TRUE),

(18, 1, 3, CURRENT_DATE - 2, (CURRENT_DATE - 2) + TIME '08:50:00', (CURRENT_DATE - 2) + TIME '17:55:00', TRUE),
(19, 1, 3, CURRENT_DATE - 2, (CURRENT_DATE - 2) + TIME '09:10:00', (CURRENT_DATE - 2) + TIME '18:10:00', TRUE),
(20, 1, 3, CURRENT_DATE - 2, (CURRENT_DATE - 2) + TIME '09:15:00', (CURRENT_DATE - 2) + TIME '18:20:00', TRUE),
(21, 1, 3, CURRENT_DATE - 2, (CURRENT_DATE - 2) + TIME '09:25:00', (CURRENT_DATE - 2) + TIME '18:40:00', TRUE),
(22, 1, 3, CURRENT_DATE - 2, (CURRENT_DATE - 2) + TIME '09:30:00', (CURRENT_DATE - 2) + TIME '18:25:00', TRUE),

(23, 1, 4, CURRENT_DATE - 2, (CURRENT_DATE - 2) + TIME '08:40:00', (CURRENT_DATE - 2) + TIME '19:10:00', TRUE),
(24, 1, 4, CURRENT_DATE - 2, (CURRENT_DATE - 2) + TIME '08:55:00', (CURRENT_DATE - 2) + TIME '18:40:00', TRUE),
(25, 1, 4, CURRENT_DATE - 2, (CURRENT_DATE - 2) + TIME '09:10:00', (CURRENT_DATE - 2) + TIME '18:55:00', TRUE),
(26, 1, 4, CURRENT_DATE - 2, (CURRENT_DATE - 2) + TIME '09:20:00', (CURRENT_DATE - 2) + TIME '18:30:00', TRUE),
(27, 1, 4, CURRENT_DATE - 2, (CURRENT_DATE - 2) + TIME '09:25:00', (CURRENT_DATE - 2) + TIME '18:50:00', TRUE),

(28, 1, 5, CURRENT_DATE - 2, (CURRENT_DATE - 2) + TIME '08:45:00', (CURRENT_DATE - 2) + TIME '17:40:00', TRUE),
(29, 1, 5, CURRENT_DATE - 2, (CURRENT_DATE - 2) + TIME '09:00:00', (CURRENT_DATE - 2) + TIME '17:55:00', TRUE),
(30, 1, 5, CURRENT_DATE - 2, (CURRENT_DATE - 2) + TIME '09:10:00', (CURRENT_DATE - 2) + TIME '18:00:00', TRUE),

-- CURRENT_DATE - 1 (Yesterday)
(1, 1, 1, CURRENT_DATE - 1, (CURRENT_DATE - 1) + TIME '08:45:00', (CURRENT_DATE - 1) + TIME '18:30:00', TRUE),
(2, 1, 1, CURRENT_DATE - 1, (CURRENT_DATE - 1) + TIME '09:00:00', (CURRENT_DATE - 1) + TIME '19:05:00', TRUE),
(3, 1, 1, CURRENT_DATE - 1, (CURRENT_DATE - 1) + TIME '08:55:00', (CURRENT_DATE - 1) + TIME '18:25:00', TRUE),
(4, 1, 1, CURRENT_DATE - 1, (CURRENT_DATE - 1) + TIME '09:10:00', (CURRENT_DATE - 1) + TIME '18:40:00', TRUE),
(5, 1, 1, CURRENT_DATE - 1, (CURRENT_DATE - 1) + TIME '09:00:00', (CURRENT_DATE - 1) + TIME '18:10:00', TRUE),
(6, 1, 1, CURRENT_DATE - 1, (CURRENT_DATE - 1) + TIME '09:05:00', (CURRENT_DATE - 1) + TIME '18:25:00', TRUE),
(7, 1, 1, CURRENT_DATE - 1, (CURRENT_DATE - 1) + TIME '09:15:00', (CURRENT_DATE - 1) + TIME '18:35:00', TRUE),
(8, 1, 1, CURRENT_DATE - 1, (CURRENT_DATE - 1) + TIME '09:20:00', (CURRENT_DATE - 1) + TIME '18:50:00', TRUE),

(9, 1, 2, CURRENT_DATE - 1, (CURRENT_DATE - 1) + TIME '08:35:00', (CURRENT_DATE - 1) + TIME '18:05:00', TRUE),
(10, 1, 2, CURRENT_DATE - 1, (CURRENT_DATE - 1) + TIME '08:50:00', (CURRENT_DATE - 1) + TIME '18:20:00', TRUE),
(11, 1, 2, CURRENT_DATE - 1, (CURRENT_DATE - 1) + TIME '09:00:00', (CURRENT_DATE - 1) + TIME '18:35:00', TRUE),
(12, 1, 2, CURRENT_DATE - 1, (CURRENT_DATE - 1) + TIME '09:05:00', (CURRENT_DATE - 1) + TIME '18:50:00', TRUE),
(13, 1, 2, CURRENT_DATE - 1, (CURRENT_DATE - 1) + TIME '09:10:00', (CURRENT_DATE - 1) + TIME '19:05:00', TRUE),
(14, 1, 2, CURRENT_DATE - 1, (CURRENT_DATE - 1) + TIME '08:55:00', (CURRENT_DATE - 1) + TIME '18:25:00', TRUE),
(15, 1, 2, CURRENT_DATE - 1, (CURRENT_DATE - 1) + TIME '09:15:00', (CURRENT_DATE - 1) + TIME '18:55:00', TRUE),
(16, 1, 2, CURRENT_DATE - 1, (CURRENT_DATE - 1) + TIME '09:20:00', (CURRENT_DATE - 1) + TIME '18:45:00', TRUE),
(17, 1, 2, CURRENT_DATE - 1, (CURRENT_DATE - 1) + TIME '09:25:00', (CURRENT_DATE - 1) + TIME '19:00:00', TRUE),

(18, 1, 3, CURRENT_DATE - 1, (CURRENT_DATE - 1) + TIME '08:45:00', (CURRENT_DATE - 1) + TIME '17:50:00', TRUE),
(19, 1, 3, CURRENT_DATE - 1, (CURRENT_DATE - 1) + TIME '09:05:00', (CURRENT_DATE - 1) + TIME '18:05:00', TRUE),
(20, 1, 3, CURRENT_DATE - 1, (CURRENT_DATE - 1) + TIME '09:10:00', (CURRENT_DATE - 1) + TIME '18:15:00', TRUE),
(21, 1, 3, CURRENT_DATE - 1, (CURRENT_DATE - 1) + TIME '09:20:00', (CURRENT_DATE - 1) + TIME '18:35:00', TRUE),
(22, 1, 3, CURRENT_DATE - 1, (CURRENT_DATE - 1) + TIME '09:25:00', (CURRENT_DATE - 1) + TIME '18:20:00', TRUE),

(23, 1, 4, CURRENT_DATE - 1, (CURRENT_DATE - 1) + TIME '08:35:00', (CURRENT_DATE - 1) + TIME '19:05:00', TRUE),
(24, 1, 4, CURRENT_DATE - 1, (CURRENT_DATE - 1) + TIME '08:50:00', (CURRENT_DATE - 1) + TIME '18:35:00', TRUE),
(25, 1, 4, CURRENT_DATE - 1, (CURRENT_DATE - 1) + TIME '09:05:00', (CURRENT_DATE - 1) + TIME '18:50:00', TRUE),
(26, 1, 4, CURRENT_DATE - 1, (CURRENT_DATE - 1) + TIME '09:15:00', (CURRENT_DATE - 1) + TIME '18:25:00', TRUE),
(27, 1, 4, CURRENT_DATE - 1, (CURRENT_DATE - 1) + TIME '09:20:00', (CURRENT_DATE - 1) + TIME '18:45:00', TRUE),

(28, 1, 5, CURRENT_DATE - 1, (CURRENT_DATE - 1) + TIME '08:40:00', (CURRENT_DATE - 1) + TIME '17:35:00', TRUE),
(29, 1, 5, CURRENT_DATE - 1, (CURRENT_DATE - 1) + TIME '08:55:00', (CURRENT_DATE - 1) + TIME '17:50:00', TRUE),
(30, 1, 5, CURRENT_DATE - 1, (CURRENT_DATE - 1) + TIME '09:05:00', (CURRENT_DATE - 1) + TIME '17:55:00', TRUE),

-- CURRENT_DATE (Today - checked in but not checked out yet)
(1, 1, 1, CURRENT_DATE, CURRENT_TIMESTAMP - INTERVAL '3 hours', NULL, FALSE),
(2, 1, 1, CURRENT_DATE, CURRENT_TIMESTAMP - INTERVAL '2.5 hours', NULL, FALSE),
(3, 1, 1, CURRENT_DATE, CURRENT_TIMESTAMP - INTERVAL '2 hours 45 minutes', NULL, FALSE),
(4, 1, 1, CURRENT_DATE, CURRENT_TIMESTAMP - INTERVAL '2 hours 30 minutes', NULL, FALSE),
(5, 1, 1, CURRENT_DATE, CURRENT_TIMESTAMP - INTERVAL '3 hours 10 minutes', NULL, FALSE),
(6, 1, 1, CURRENT_DATE, CURRENT_TIMESTAMP - INTERVAL '2 hours 55 minutes', NULL, FALSE),
(7, 1, 1, CURRENT_DATE, CURRENT_TIMESTAMP - INTERVAL '2 hours 20 minutes', NULL, FALSE),
(8, 1, 1, CURRENT_DATE, CURRENT_TIMESTAMP - INTERVAL '2 hours 15 minutes', NULL, FALSE),

(9, 1, 2, CURRENT_DATE, CURRENT_TIMESTAMP - INTERVAL '3 hours 30 minutes', NULL, FALSE),
(10, 1, 2, CURRENT_DATE, CURRENT_TIMESTAMP - INTERVAL '3 hours 15 minutes', NULL, FALSE),
(11, 1, 2, CURRENT_DATE, CURRENT_TIMESTAMP - INTERVAL '3 hours', NULL, FALSE),
(12, 1, 2, CURRENT_DATE, CURRENT_TIMESTAMP - INTERVAL '2 hours 55 minutes', NULL, FALSE),
(13, 1, 2, CURRENT_DATE, CURRENT_TIMESTAMP - INTERVAL '2 hours 50 minutes', NULL, FALSE),
(14, 1, 2, CURRENT_DATE, CURRENT_TIMESTAMP - INTERVAL '3 hours 5 minutes', NULL, FALSE),
(15, 1, 2, CURRENT_DATE, CURRENT_TIMESTAMP - INTERVAL '2 hours 45 minutes', NULL, FALSE),
(16, 1, 2, CURRENT_DATE, CURRENT_TIMESTAMP - INTERVAL '2 hours 40 minutes', NULL, FALSE),
(17, 1, 2, CURRENT_DATE, CURRENT_TIMESTAMP - INTERVAL '2 hours 35 minutes', NULL, FALSE),

(18, 1, 3, CURRENT_DATE, CURRENT_TIMESTAMP - INTERVAL '3 hours 20 minutes', NULL, FALSE),
(19, 1, 3, CURRENT_DATE, CURRENT_TIMESTAMP - INTERVAL '3 hours', NULL, FALSE),
(20, 1, 3, CURRENT_DATE, CURRENT_TIMESTAMP - INTERVAL '2 hours 55 minutes', NULL, FALSE),
(21, 1, 3, CURRENT_DATE, CURRENT_TIMESTAMP - INTERVAL '2 hours 45 minutes', NULL, FALSE),
(22, 1, 3, CURRENT_DATE, CURRENT_TIMESTAMP - INTERVAL '2 hours 40 minutes', NULL, FALSE),

(23, 1, 4, CURRENT_DATE, CURRENT_TIMESTAMP - INTERVAL '3 hours 30 minutes', NULL, FALSE),
(24, 1, 4, CURRENT_DATE, CURRENT_TIMESTAMP - INTERVAL '3 hours 15 minutes', NULL, FALSE),
(25, 1, 4, CURRENT_DATE, CURRENT_TIMESTAMP - INTERVAL '3 hours', NULL, FALSE),
(26, 1, 4, CURRENT_DATE, CURRENT_TIMESTAMP - INTERVAL '2 hours 50 minutes', NULL, FALSE),
(27, 1, 4, CURRENT_DATE, CURRENT_TIMESTAMP - INTERVAL '2 hours 45 minutes', NULL, FALSE),

(28, 1, 5, CURRENT_DATE, CURRENT_TIMESTAMP - INTERVAL '3 hours 25 minutes', NULL, FALSE),
(29, 1, 5, CURRENT_DATE, CURRENT_TIMESTAMP - INTERVAL '3 hours 10 minutes', NULL, FALSE),
(30, 1, 5, CURRENT_DATE, CURRENT_TIMESTAMP - INTERVAL '3 hours', NULL, FALSE);


-- ============================================
-- STEP 13: INSERT DEPARTMENT ATTENDANCE STATS (NeuralTech AI only)
-- ============================================

INSERT INTO department_attendance_stats (department_id, month, total_employees, total_attendances, attendance_percentage) VALUES

-- Current month stats for all NeuralTech departments
(1, DATE_TRUNC('month', CURRENT_DATE), 8, 160, 93.00),  -- ML Research: 8 people x 20 workdays = 160
(2, DATE_TRUNC('month', CURRENT_DATE), 9, 171, 95.00),   -- Engineering: 9 people x 20 workdays = 180, 171 attended
(3, DATE_TRUNC('month', CURRENT_DATE), 5, 98, 98.00),    -- Product: 5 people x 20 workdays = 100, 98 attended
(4, DATE_TRUNC('month', CURRENT_DATE), 5, 100, 100.00),  -- Sales: 5 people x 20 workdays = 100
(5, DATE_TRUNC('month', CURRENT_DATE), 3, 57, 95.00),    -- HR: 3 people x 20 workdays = 60, 57 attended

-- Previous month stats
(1, DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month'), 8, 168, 105.00),  -- ML Research
(2, DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month'), 9, 180, 100.00),  -- Engineering
(3, DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month'), 5, 102, 102.00),  -- Product
(4, DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month'), 5, 105, 105.00),  -- Sales
(5, DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month'), 3, 60, 100.00),   -- HR

-- Two months ago
(1, DATE_TRUNC('month', CURRENT_DATE - INTERVAL '2 months'), 8, 164, 102.50),
(2, DATE_TRUNC('month', CURRENT_DATE - INTERVAL '2 months'), 9, 176, 97.78),
(3, DATE_TRUNC('month', CURRENT_DATE - INTERVAL '2 months'), 5, 100, 100.00),
(4, DATE_TRUNC('month', CURRENT_DATE - INTERVAL '2 months'), 5, 98, 98.00),
(5, DATE_TRUNC('month', CURRENT_DATE - INTERVAL '2 months'), 3, 58, 96.67);


-- ============================================
-- STEP 14: INSERT NEW HIRES TRACKING (NeuralTech AI only)
-- ============================================

INSERT INTO new_hires_tracking (organization_id, hires_this_month, hires_this_year, month, year) VALUES

-- Current year tracking for NeuralTech AI
(1, 3, 30, DATE_TRUNC('month', CURRENT_DATE), EXTRACT(YEAR FROM CURRENT_DATE)),

-- Previous months this year
(1, 2, 27, DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month'), EXTRACT(YEAR FROM CURRENT_DATE)),
(1, 4, 25, DATE_TRUNC('month', CURRENT_DATE - INTERVAL '2 months'), EXTRACT(YEAR FROM CURRENT_DATE)),
(1, 3, 21, DATE_TRUNC('month', CURRENT_DATE - INTERVAL '3 months'), EXTRACT(YEAR FROM CURRENT_DATE)),
(1, 2, 18, DATE_TRUNC('month', CURRENT_DATE - INTERVAL '4 months'), EXTRACT(YEAR FROM CURRENT_DATE)),
(1, 5, 16, DATE_TRUNC('month', CURRENT_DATE - INTERVAL '5 months'), EXTRACT(YEAR FROM CURRENT_DATE)),
(1, 2, 11, DATE_TRUNC('month', CURRENT_DATE - INTERVAL '6 months'), EXTRACT(YEAR FROM CURRENT_DATE)),
(1, 3, 9, DATE_TRUNC('month', CURRENT_DATE - INTERVAL '7 months'), EXTRACT(YEAR FROM CURRENT_DATE)),
(1, 1, 6, DATE_TRUNC('month', CURRENT_DATE - INTERVAL '8 months'), EXTRACT(YEAR FROM CURRENT_DATE)),
(1, 2, 5, DATE_TRUNC('month', CURRENT_DATE - INTERVAL '9 months'), EXTRACT(YEAR FROM CURRENT_DATE)),
(1, 2, 3, DATE_TRUNC('month', CURRENT_DATE - INTERVAL '10 months'), EXTRACT(YEAR FROM CURRENT_DATE)),
(1, 1, 1, DATE_TRUNC('month', CURRENT_DATE - INTERVAL '11 months'), EXTRACT(YEAR FROM CURRENT_DATE));


-- ============================================
-- STEP 15: INSERT USER STATISTICS (NeuralTech AI only)
-- ============================================

INSERT INTO user_statistics (user_id, total_assignments, completed_assignments, pending_assignments, total_work_hours) VALUES

-- ML Research Department
(1, 28, 22, 6, 2147.50),   -- Sarah Chen - Research Director
(2, 32, 25, 7, 2389.75),   -- Michael Rodriguez - Senior ML Researcher
(3, 25, 19, 6, 1876.25),   -- Priya Patel - Senior ML Researcher
(4, 21, 16, 5, 1654.00),   -- David Kim - ML Researcher
(5, 19, 15, 4, 1523.50),   -- Amanda Thompson - ML Researcher
(6, 18, 14, 4, 1445.75),   -- James Lee - ML Researcher
(7, 15, 12, 3, 1234.25),   -- Emily Garcia - Research Engineer
(8, 12, 10, 2, 989.50),    -- Daniel Nguyen - Research Engineer

-- Engineering Department
(9, 35, 28, 7, 2634.75),   -- Robert Anderson - VP Engineering
(10, 31, 24, 7, 2398.50),  -- Lisa Martinez - Engineering Manager
(11, 29, 22, 7, 2187.25),  -- Kevin Brown - Engineering Manager
(12, 26, 20, 6, 1998.00),  -- Jennifer Wilson - Senior Engineer
(13, 24, 19, 5, 1876.50),  -- Christopher Davis - Senior Engineer
(14, 23, 18, 5, 1789.75),  -- Michelle Taylor - Senior Engineer
(15, 19, 15, 4, 1534.25),  -- Andrew Moore - Software Engineer
(16, 18, 14, 4, 1445.00),  -- Jessica White - Software Engineer
(17, 16, 13, 3, 1298.50),  -- Matthew Harris - Software Engineer

-- Product Management Department
(18, 27, 21, 6, 2056.50),  -- Rachel Thomas - Head of Product
(19, 24, 19, 5, 1887.75),  -- Brandon Jackson - Product Manager
(20, 22, 17, 5, 1723.25),  -- Stephanie Clark - Product Manager
(21, 17, 13, 4, 1376.50),  -- Ryan Lewis - Product Analyst
(22, 15, 12, 3, 1198.75),  -- Nicole Walker - Product Analyst

-- Sales & BD Department
(23, 29, 23, 6, 2234.50),  -- Brian Young - VP Sales
(24, 26, 21, 5, 2045.75),  -- Laura King - Enterprise AE
(25, 24, 19, 5, 1923.25),  -- Nathan Scott - Enterprise AE
(26, 21, 17, 4, 1689.50),  -- Emma Green - BD Manager
(27, 19, 15, 4, 1534.00),  -- Tyler Adams - BD Manager

-- HR Department
(28, 23, 18, 5, 1834.75),  -- Sophia Baker - CPO
(29, 20, 16, 4, 1598.50),  -- Jacob Nelson - HR Manager
(30, 16, 13, 3, 1287.25);  -- Olivia Carter - HR Specialist


-- ============================================
-- STEP 16: INSERT JOIN REQUESTS (NeuralTech AI only)
-- ============================================

-- First, create pending applicants for NeuralTech AI
INSERT INTO users (first_name, last_name, email, password_hash, avatar_color, organization_id, is_approved, is_active) VALUES

-- ML Research applicants
('Alex', 'Johnson', 'alex.johnson@neuraltech.com', 'alex123', '#9b59b6', 1, FALSE, TRUE),
('Maria', 'Santos', 'maria.santos@neuraltech.com', 'maria123', '#e74c3c', 1, FALSE, TRUE),
('Thomas', 'Chen', 'thomas.chen@neuraltech.com', 'thomas123', '#3498db', 1, FALSE, TRUE),
('Sophie', 'Anderson', 'sophie.anderson@neuraltech.com', 'sophie123', '#27ae60', 1, FALSE, TRUE),
('James', 'Williams', 'james.williams@neuraltech.com', 'james123', '#f39c12', 1, FALSE, TRUE),
('Emily', 'Brown', 'emily.brown@neuraltech.com', 'emily123', '#1abc9c', 1, FALSE, TRUE),
('Ryan', 'Miller', 'ryan.miller@neuraltech.com', 'ryan123', '#e67e22', 1, FALSE, TRUE),

-- Product applicants
('Isabella', 'Garcia', 'isabella.garcia@neuraltech.com', 'isabella123', '#95a5a6', 1, FALSE, TRUE),
('Lucas', 'Martinez', 'lucas.martinez@neuraltech.com', 'lucas123', '#34495e', 1, FALSE, TRUE),

-- Sales applicants
('Ava', 'Taylor', 'ava.taylor@neuraltech.com', 'ava123', '#e91e63', 1, FALSE, TRUE),
('Ethan', 'Davis', 'ethan.davis@neuraltech.com', 'ethan123', '#9c27b0', 1, FALSE, TRUE),

-- HR applicant
('Mia', 'Rodriguez', 'mia.rodriguez@neuraltech.com', 'mia123', '#00bcd4', 1, FALSE, TRUE);


-- Now create join requests for these applicants
INSERT INTO join_requests (user_id, organization_id, requested_department_id, requested_role_id, status, requested_at) VALUES

-- ML Research requests
(40, 1, 1, 3, 'pending', NOW() - INTERVAL '5 days'),   -- Alex Johnson -> ML Researcher
(41, 1, 1, 4, 'pending', NOW() - INTERVAL '3 days'),   -- Maria Santos -> Research Engineer
(42, 1, 1, 2, 'pending', NOW() - INTERVAL '7 days'),   -- Thomas Chen -> Senior ML Researcher
(43, 1, 1, 8, 'pending', NOW() - INTERVAL '2 days'),   -- Sophie Anderson -> Software Engineer
(44, 1, 1, 8, 'pending', NOW() - INTERVAL '4 days'),   -- James Williams -> Software Engineer
(45, 1, 1, 7, 'pending', NOW() - INTERVAL '6 days'),   -- Emily Brown -> Senior Engineer
(46, 1, 1, 8, 'pending', NOW() - INTERVAL '1 day'),    -- Ryan Miller -> Software Engineer

-- Product requests
(47, 1, 3, 11, 'pending', NOW() - INTERVAL '3 days'),  -- Isabella Garcia -> Product Analyst
(48, 1, 3, 10, 'pending', NOW() - INTERVAL '8 days'),  -- Lucas Martinez -> Product Manager

-- Sales requests
(49, 1, 4, 14, 'pending', NOW() - INTERVAL '4 days'),  -- Ava Taylor -> BD Manager
(50, 1, 4, 13, 'pending', NOW() - INTERVAL '2 days'),  -- Ethan Davis -> Enterprise AE

-- HR request
(51, 1, 5, 17, 'pending', NOW() - INTERVAL '6 days');  -- Mia Rodriguez -> HR Specialist

-- Update employees_count for each organization
UPDATE organizations SET 
  employees_count = (
    SELECT COUNT(*) 
    FROM users 
    WHERE organization_id = organizations.id 
    AND is_approved = TRUE
  );

-- Update departments_count for each organization
UPDATE organizations SET 
  departments_count = (
    SELECT COUNT(*) 
    FROM departments 
    WHERE organization_id = organizations.id
  );





-- ============================================
-- 2: FINAL VERIFICATION QUERIES
-- ============================================
-- ============================================
SELECT id, name, departments_count, employees_count
FROM organizations
ORDER BY id;

SELECT id, organization_id, name, manager_id
FROM departments
ORDER BY organization_id;

SELECT id, organization_id, department_id, name, can_accept_requests, can_post_news, can_post_event, can_assign_tasks, can_receive_tasks, can_view_statistics, can_hire, can_reassign_tasks, can_modify_department_look
FROM roles
ORDER BY organization_id,department_id;

SELECT id, organization_id, department_id, first_name, last_name, email, role_id
FROM users
ORDER BY organization_id,department_id;
-- Show summary
SELECT 'Organization counts updated successfully!' AS message;