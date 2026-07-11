import React from 'react'
import { useState, useRef } from 'react';
import Sidebar from '../Shared/Sidebar/Sidebar_Welcome';
import Header from '../Shared/Header/Header_Welcome';
import NewsSection from '../Sections/NewsSection_Welcome';
import ClassementSection from '../Sections/ClassementSection';
import SearchSection from '../Sections/SearchSection';
import AboutSection from '../Sections/AboutSection';
import '../../styles/WelcomePage.css';

export default function WelcomePage() {
  const [activeSection, setActiveSection] = useState('home');
  const [sidebarExpanded, setSidebarExpanded] = useState(false);
  const [searchContext, setSearchContext] = useState(null);
  const mainContentRef = useRef(null);

  const handleCompanyClick = (companyName, companyId) => {
    setSearchContext({ name: companyName, id: companyId });
    setActiveSection('search');
  };

  const renderContent = () => {
    switch(activeSection) {
      case 'home':
        return <NewsSection onCompanyClick={handleCompanyClick} />;
      case 'classement':
        return <ClassementSection />;
      case 'search':
        return <SearchSection initialSearch={searchContext?.name} />;
      case 'about':
        return <AboutSection />;
      default:
        return <NewsSection onCompanyClick={handleCompanyClick} />;
    }
  };

  return (
    <div className="home-page-container">
      <Header />
      <div className="home-page-body">
        {/* Sidebar-ul și Handle-ul sunt generate de componenta Sidebar */}
        <Sidebar 
          activeSection={activeSection} 
          setActiveSection={setActiveSection}
          sidebarExpanded={sidebarExpanded}
          setSidebarExpanded={setSidebarExpanded}
        />
        <main className="main-content" ref={mainContentRef}>
          <div className="content-wrapper">
            {renderContent()}
          </div>
        </main>
      </div>
    </div>
  );
}