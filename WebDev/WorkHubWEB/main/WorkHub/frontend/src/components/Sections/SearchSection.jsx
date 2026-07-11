import React, { useState, useEffect } from 'react';
import CompanyCard from '../Cards/CompanyCard';
import '../../styles/WelcomePage.css';

const API_BASE_URL = 'http://localhost:3000/api';

export default function SearchSection({ initialSearch }) {
  const [searchQuery, setSearchQuery] = useState(initialSearch || '');
  const [companies, setCompanies] = useState([]);
  const [selectedCompany, setSelectedCompany] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (initialSearch) {
      handleSearch(initialSearch);
    }
  }, [initialSearch]);

  useEffect(() => {
    if (selectedCompany) {
      const handleEsc = (e) => {
        if (e.key === 'Escape') {
          handleBack();
        }
      };
      window.addEventListener('keydown', handleEsc);
      return () => window.removeEventListener('keydown', handleEsc);
    }
  }, [selectedCompany]);

  const handleSearch = async (query) => {
    setSearchQuery(query);
    if (query.trim() === '') {
      setCompanies([]);
      setSelectedCompany(null);
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/organizations`);
      const result = await response.json();
      
      if (result.success) {
        const filtered = result.data.filter(c =>
          c.name.toLowerCase().includes(query.toLowerCase()) ||
          c.industry.toLowerCase().includes(query.toLowerCase())
        );
        setCompanies(filtered);
      }
    } catch (error) {
      console.error('Error searching companies:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectCompany = (company) => {
    setSearchQuery(company.name);
    setSelectedCompany(company);
    setCompanies([]);
  };

  const handleBack = () => {
    setSelectedCompany(null);
    setSearchQuery('');
  };

  return (
    <div className="search-section">
      <div className="search-header">
        <h2>Find Companies</h2>
        <p>Search and explore companies on WorkHub</p>
      </div>

      {!selectedCompany && (
        <div className="search-container">
          <input
            type="text"
            className="search-input"
            placeholder="Search by company name or industry..."
            value={searchQuery}
            onChange={(e) => handleSearch(e.target.value)}
          />
        </div>
      )}

      {loading && <div className="loading">Searching...</div>}

      {searchQuery && companies.length === 0 && !loading && !selectedCompany && (
        <div className="no-results">No companies found</div>
      )}

      {selectedCompany ? (
        <CompanyCard 
          company={selectedCompany}
          onBack={handleBack}
        />
      ) : (
        <div className="companies-list">
          {companies.map(company => (
            <div
              key={company.id}
              className="company-preview"
              onClick={() => handleSelectCompany(company)}
            >
              <div className="preview-logo">
                <img src={company.logo_url} alt={company.name} />
              </div>
              <div className="preview-info">
                <h3>{company.name}</h3>
                <p>{company.industry}</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}