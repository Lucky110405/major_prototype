import React, { useState } from 'react';
import axios from 'axios';

const Query = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.get(`/query/?q=${encodeURIComponent(query)}`);
      setResults(response.data.results || []);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div>
      <h2>Query Data</h2>
      <form onSubmit={handleSubmit}>
        <div className="mb-3">
          <input
            type="text"
            className="form-control"
            placeholder="Enter your query"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
        <button type="submit" className="btn btn-primary">Query</button>
      </form>
      <div className="mt-4">
        <h3>Results</h3>
        {results.length > 0 ? (
          <ul className="list-group">
            {results.map((result, index) => (
              <li key={index} className="list-group-item">
                <strong>ID:</strong> {result.id}<br />
                <strong>Score:</strong> {result.score}<br />
                <strong>Metadata:</strong> {JSON.stringify(result.metadata)}
              </li>
            ))}
          </ul>
        ) : (
          <p>No results found.</p>
        )}
      </div>
    </div>
  );
};

export default Query;