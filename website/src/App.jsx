import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router';
import Home from './pages/Home';
import Upload from './pages/Upload.jsx';
import Navbar from "./Navbar";

function App() {
    return (
        <Router>
            <Navbar />
            <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/Upload" element={<Upload />} />
            </Routes>
        </Router>
    );
}

export default App;