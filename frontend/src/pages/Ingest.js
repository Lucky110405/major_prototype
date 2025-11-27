import React, { useState } from 'react';
import axios from 'axios';

const Ingest = () => {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState('');

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('/ingest/auto', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setMessage('Ingestion successful: ' + JSON.stringify(response.data));
    } catch (error) {
      setMessage('Error: ' + error.response?.data?.message || error.message);
    }
  };

  return (
    <div>
      <h2>Data Ingestion</h2>
      <form onSubmit={handleSubmit}>
        <div className="mb-3">
          <input type="file" className="form-control" onChange={handleFileChange} />
        </div>
        <button type="submit" className="btn btn-primary">Ingest</button>
      </form>
      {message && <div className="alert alert-info mt-3">{message}</div>}
    </div>
  );
};

export default Ingest;