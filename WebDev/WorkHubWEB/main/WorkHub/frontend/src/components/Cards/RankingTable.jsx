import React from 'react';

export default function RankingTable({ data, type }) {
  if (type === 'industryStats') {
    // Sort industry stats by count descending
    const sortedData = [...data].sort((a, b) => b.count - a.count);
    
    return (
      <div className="ranking-table industry-stats">
        <div className="stats-grid">
          {sortedData.map((stat, idx) => (
            <div key={idx} className="stat-card">
              <h4>{stat.industry}</h4>
              <div className="stat-number">{stat.count}</div>
              <p>Companies</p>
            </div>
          ))}
        </div>
      </div>
    );
  }

  const getColumnLabel = () => {
    const labels = {
      departments: 'Departments',
      employees: 'Employees'
    };
    return labels[type] || 'Value';
  };

  const getColumnValue = (company) => {
    switch(type) {
      case 'departments':
        return company.departments_count || 0;
      case 'employees':
        return company.employees_count || 0;
      default:
        return '';
    }
  };

  const getSortValue = (company) => {
    switch(type) {
      case 'departments':
        return company.departments_count || 0;
      case 'employees':
        return company.employees_count || 0;
      default:
        return 0;
    }
  };

  // Sort data in descending order based on the current type
  const sortedData = [...data].sort((a, b) => getSortValue(b) - getSortValue(a));

  return (
    <div className="ranking-table">
      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Company</th>
            <th>Industry</th>
            <th>{getColumnLabel()}</th>
          </tr>
        </thead>
        <tbody>
          {sortedData.map((company, idx) => (
            <tr key={company.id} className={idx === 0 ? 'first-place' : ''}>
              <td className="rank">#{idx + 1}</td>
              <td className="company-name">{company.name}</td>
              <td className="industry">{company.industry}</td>
              <td className="value">{getColumnValue(company)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}