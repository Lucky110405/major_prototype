import { useState } from 'react';
import Layout from './components/Layout';
import IngestPage from './pages/IngestPage';
import QueryPage from './pages/QueryPage';
import AnalysisAgentPage from './pages/AnalysisAgentPage';

type Page = 'ingest' | 'query' | 'agent';

function App() {
  const [currentPage, setCurrentPage] = useState<Page>('ingest');

  const renderPage = () => {
    switch (currentPage) {
      case 'ingest':
        return <IngestPage />;
      case 'query':
        return <QueryPage />;
      case 'agent':
        return <AnalysisAgentPage />;
      default:
        return <IngestPage />;
    }
  };

  return (
    <Layout currentPage={currentPage} onNavigate={setCurrentPage}>
      {renderPage()}
    </Layout>
  );
}

export default App;
