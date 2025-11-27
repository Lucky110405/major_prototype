import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.min.css';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Ingest from './pages/Ingest';
import Query from './pages/Query';
import Agents from './pages/Agents';

function App() {
  return (
    <Router>
      <div className="App">
        <Navbar />
        <div className="container mt-4">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/ingest" element={<Ingest />} />
            <Route path="/query" element={<Query />} />
            <Route path="/agents" element={<Agents />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;