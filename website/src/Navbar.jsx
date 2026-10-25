import React, { useState, useEffect } from 'react';
import { Link } from 'react-router';
import './Navbar.css';

export default function Navbar() {
    const [darkMode, setDarkMode] = useState(false);

    useEffect(() => {
        if(darkMode)
            document.body.classList.add('dark-mode');
        else
            document.body.classList.remove('dark-mode');
    }, [darkMode]);

    return (
        <nav>
            <Link className='nav-item' to='/'>
                Home
            </Link>
            <Link className='nav-item' to='/Upload'>
                Upload
            </Link>
            <div className="nav-right">
                <div className="dark-mode-toggle" onClick={() => setDarkMode(!darkMode)}>
                    {darkMode ? "☀️" : "🌙"}
                </div>
            </div>
        </nav>
    )
}