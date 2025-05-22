// PredictionPanel.tsx
import React from 'react';

interface Props {
  prediction: number;
  date: string;
  isCreating: boolean;
  onCreate: () => void;
  theme?: string;
}

export const PredictionPanel: React.FC<Props> = ({ prediction, date, isCreating, onCreate, theme = 'dark' }) => {
  return (
    <div style={{
      maxWidth: '800px',
      margin: '1rem auto',
      padding: '1rem',
      color: theme === 'dark' ? '#fff' : '#000',
      borderRadius: '8px',
      textAlign: 'center',
    }}>
      <div
        style={{
          padding: '1rem 0',
          marginBottom: '1rem',
          fontSize: '1.25rem',
          fontWeight: 'bold',
          backgroundColor: prediction > 0.5 ? '#f44336' : '#4caf50',
          color: '#fff',
          borderRadius: '8px',
        }}
      >
        {prediction > 0.5
          ? 'Yes, there is a likelihood of a wildfire.'
          : 'No, you are safe. No wildfire detected.'}
        <div>
          Probability: <strong>{(prediction * 100).toFixed(2)}%</strong>
        </div>
      </div>

      {prediction > 0.5 && (
        <button
          onClick={onCreate}
          disabled={isCreating}
          style={{
            padding: '0.75rem 1.5rem',
            backgroundColor: 'red',
            color: '#fff',
            border: 'none',
            borderRadius: '5px',
            cursor: isCreating ? 'not-allowed' : 'pointer',
            fontSize: '1rem',
            fontWeight: 'bold',
            opacity: isCreating ? 0.7 : 1
          }}
        >
          {isCreating ? 'Creating Environment...' : 'Go to Tactics'}
        </button>
      )}
    </div>
  );
};
