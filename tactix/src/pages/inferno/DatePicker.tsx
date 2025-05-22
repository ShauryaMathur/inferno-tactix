// DatePicker.tsx
import React from 'react';

interface Props {
  value: string;
  onChange: (val: string) => void;
  min?: string;
  max?: string;
}

export const DatePicker: React.FC<Props> = ({ value, onChange, min, max }) => {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange(e.target.value);
  };

  return (
    <input
      type="date"
      value={value}
      onChange={handleChange}
      min={min}
      max={max}
    />
  );
};
