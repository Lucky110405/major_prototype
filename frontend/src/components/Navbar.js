import React from 'react';
import { Link } from 'react-router-dom';

const Navbar = () => {
  return (
    <nav className="navbar navbar-expand-lg navbar-dark bg-dark">
      <div className="container">
        <Link className="navbar-brand" to="/">CMM BI Framework</Link>
        <div className="navbar-nav">
          <Link className="nav-link" to="/">Home</Link>
          <Link className="nav-link" to="/ingest">Ingest</Link>
          <Link className="nav-link" to="/query">Query</Link>
          <Link className="nav-link" to="/agents">Agents</Link>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;