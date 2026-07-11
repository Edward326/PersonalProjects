// frontend/src/services/api.js
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3000/api';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Organizations
export const getOrganizations = () => api.get('/organizations');
export const getOrganization = (id) => api.get(`/organizations/${id}`);

// News
export const getPublicNews = (limit = 10, offset = 0) => 
  api.get(`/news/public?limit=${limit}&offset=${offset}`);

// Export the api instance
export default api;