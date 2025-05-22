// LoadingSpinner.tsx
import React from 'react';

interface Props {
  message: string;
}

export const LoadingSpinner: React.FC<Props> = ({ message }) => (
  <div style={{
    position: 'fixed',
    top: 0, left: 0, width: '100%', height: '100%',
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    display: 'flex', justifyContent: 'center', alignItems: 'center',
    zIndex: 9999
  }}>
    <div style={{ textAlign: 'center', color: '#fff' }}>
      <div style={{
        border: '5px solid #f3f3f3',
        borderTop: '5px solid #ffdf00',
        borderRadius: '50%',
        width: '50px', height: '50px',
        margin: '0 auto 20px',
        animation: 'spin 2s linear infinite'
      }} />
      <p>{message}</p>
    </div>
    <style>
      {`@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`}
    </style>
  </div>
);
