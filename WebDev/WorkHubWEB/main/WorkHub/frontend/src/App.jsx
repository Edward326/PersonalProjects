import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import ProtectedRoute from './components/ProtectedRoute.jsx';
// Public Pages
import WelcomePage from './components/pages/WelcomePage';
import ChooseRegister from './components/Auth/ChooseRegister';
import LoginPage from './components/Auth/LoginPage';
import EmployeeRegister from './components/Auth/EmployeeRegister';
import OrganizationRegister from './components/Auth/OrganizationRegister';
// Protected Pages
import HomePage from './components/pages/HomePage';
import RequestsPage from './components/pages/RequestsPage';
import AssignmentsPage from './components/pages/AssignmentPage';
import OrganizationPage from './components/pages/OrganizationPage';
import StatisticsPage from './components/pages/StatisticsPage';
import SettingsPage from './components/pages/SettingsPage';

function App() {
  return (
    <Router>
      <Routes>
        {/* Public Routes */}
        <Route path="/welcome" element={<WelcomePage />} />
        <Route path="/choose-register" element={<ChooseRegister />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/employee-register" element={<EmployeeRegister />} />
        <Route path="/organization-register" element={<OrganizationRegister />} />
        
        {/* Protected Routes - All require authentication */}
        <Route 
          path="/" 
          element={
            <ProtectedRoute>
              <HomePage />
            </ProtectedRoute>
          } 
        />
        
        <Route 
          path="/requests" 
          element={
            <ProtectedRoute>
              <RequestsPage />
            </ProtectedRoute>
          } 
        />
        
        <Route 
          path="/assignments" 
          element={
            <ProtectedRoute>
              <AssignmentsPage />
            </ProtectedRoute>
          } 
        />
        
        <Route 
          path="/organization" 
          element={
            <ProtectedRoute>
              <OrganizationPage />
            </ProtectedRoute>
          } 
        />
        
        <Route 
          path="/statistics" 
          element={
            <ProtectedRoute>
              <StatisticsPage />
            </ProtectedRoute>
          } 
        />
        
        <Route 
          path="/settings" 
          element={
            <ProtectedRoute>
              <SettingsPage />
            </ProtectedRoute>
          } 
        />
      </Routes>
    </Router>
  );
}

export default App;