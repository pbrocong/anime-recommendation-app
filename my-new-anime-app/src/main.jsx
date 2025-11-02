import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'

// Ensure this line exists so global styles are applied.
import './index.css'

/**
 * React entry point:
 * - Creates a root on the #root element
 * - Renders <App /> within React.StrictMode for extra checks in development
 */
ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)