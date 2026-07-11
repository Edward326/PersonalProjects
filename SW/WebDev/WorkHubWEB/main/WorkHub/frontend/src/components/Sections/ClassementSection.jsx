import React, { useState, useEffect } from 'react';
import RankingTable from '../Cards/RankingTable';
import '../../styles/WelcomePage.css';

const API_BASE_URL = 'http://localhost:3000/api';

export default function ClassementSection() {
  const [activeRanking, setActiveRanking] = useState('employees');
  const [filterByIndustry, setFilterByIndustry] = useState(false);
  const [selectedIndustry, setSelectedIndustry] = useState('all');
  const [companies, setCompanies] = useState([]);
  const [industries, setIndustries] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchCompanies();
  }, []);

  const fetchCompanies = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/organizations`);
      const result = await response.json();
      
      if (result.success) {
        setCompanies(result.data);
        const uniqueIndustries = [...new Set(result.data.map(c => c.industry))];
        setIndustries(uniqueIndustries);
      }
    } catch (error) {
      console.error('Error fetching companies:', error);
    } finally {
      setLoading(false);
    }
  };

  const rankingOptions = [
    { id: 'employees', label: 'By Employees', icon: '👥' },
    { id: 'departments', label: 'By Departments', icon: '🏢' },
    { id: 'industryStats', label: 'Industry Stats', icon: '📈' }
  ];

  const getRankingData = () => {
    if (activeRanking === 'industryStats') {
      return industries.map(ind => ({
        industry: ind,
        count: companies.filter(c => c.industry === ind).length
      })).sort((a, b) => b.count - a.count);
    }
    
    let data = [...companies];
    
    if (filterByIndustry && selectedIndustry !== 'all') {
      data = data.filter(c => c.industry === selectedIndustry);
    }
    
    // Sort by the active ranking field in descending order
    if (activeRanking === 'employees') {
      data.sort((a, b) => (b.employees_count || 0) - (a.employees_count || 0));
    } else if (activeRanking === 'departments') {
      data.sort((a, b) => (b.departments_count || 0) - (a.departments_count || 0));
    }
    
    return data;
  };

  if (loading) {
    return <div className="loading">Loading rankings...</div>;
  }

  return (
    <div className="classement-section">
      <div className="classement-header">
        <h2>Rankings & Classements</h2>
      </div>

      {activeRanking !== 'industryStats' && (
        <div className="industry-filter-container">
          <button 
            className={`industry-filter-btn ${filterByIndustry ? 'active' : ''}`}
            onClick={() => setFilterByIndustry(!filterByIndustry)}
          >
            {filterByIndustry ? '🔍 Filtered by Industry' : '🌐 Global View'}
          </button>
          
          {filterByIndustry && (
            <select 
              className="industry-select show"
              value={selectedIndustry}
              onChange={(e) => setSelectedIndustry(e.target.value)}
            >
              <option value="all">All Industries</option>
              {industries.map(ind => (
                <option key={ind} value={ind}>{ind}</option>
              ))}
            </select>
          )}
        </div>
      )}

      <div className="ranking-tabs">
        {rankingOptions.map(option => (
          <button
            key={option.id}
            className={`ranking-tab ${activeRanking === option.id ? 'active' : ''}`}
            onClick={() => {
              setActiveRanking(option.id);
              setFilterByIndustry(false);
            }}
          >
            <span className="tab-icon">{option.icon}</span>
            <span className="tab-label">{option.label}</span>
          </button>
        ))}
      </div>

      <RankingTable 
        data={getRankingData()}
        type={activeRanking}
      />
    </div>
  );
}